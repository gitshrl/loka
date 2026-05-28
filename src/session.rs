use anyhow::{Context, Result, anyhow};
use rusqlite::types::Type;
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use serde_json::Value;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
use uuid::Uuid;

use crate::messages::Role;

#[derive(Debug)]
pub struct SessionStore {
    conn: Connection,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SessionSummary {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub turn_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SearchHit {
    pub session_id: String,
    pub title: String,
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolCallStatus {
    Running,
    Completed,
    Failed,
}

impl ToolCallStatus {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }

    #[must_use]
    pub fn from_db(value: &str) -> Option<Self> {
        match value {
            "running" => Some(Self::Running),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolCallRecord {
    pub id: String,
    pub session_id: String,
    pub name: String,
    pub input: Value,
    pub status: ToolCallStatus,
    pub output: Option<Value>,
    pub error: Option<String>,
    pub created_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SessionTurn {
    pub role: Role,
    pub content: String,
    pub created_at: String,
}

impl SessionStore {
    /// Opens the session database inside the configured state directory.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory cannot be created, the database cannot be opened,
    /// or migrations fail.
    pub fn open(state_dir: impl AsRef<Path>) -> Result<Self> {
        fs::create_dir_all(state_dir.as_ref()).with_context(|| {
            format!(
                "create session state directory {}",
                state_dir.as_ref().display()
            )
        })?;
        Self::open_database(state_dir.as_ref().join("sessions.sqlite3"))
    }

    /// Opens a session database at an explicit path.
    ///
    /// # Errors
    ///
    /// Returns an error when the database cannot be opened or migrations fail.
    pub fn open_database(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref())
            .with_context(|| format!("open session database {}", path.as_ref().display()))?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    /// Opens an in-memory session database for tests and short-lived workflows.
    ///
    /// # Errors
    ///
    /// Returns an error when the `SQLite` connection cannot be created or migrations fail.
    pub fn in_memory() -> Result<Self> {
        let store = Self {
            conn: Connection::open_in_memory().context("open in-memory session database")?,
        };
        store.migrate()?;
        Ok(store)
    }

    /// Creates a new session and returns its id.
    ///
    /// # Errors
    ///
    /// Returns an error when timestamp formatting fails or `SQLite` rejects the insert.
    pub fn create_session(&self, title: &str) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let now = now_rfc3339()?;
        self.conn
            .execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)",
                params![id, normalize_title(title), now, now],
            )
            .context("create session")?;
        Ok(id)
    }

    /// Appends one turn to an existing session.
    ///
    /// # Errors
    ///
    /// Returns an error when timestamp formatting fails, the session id does not exist,
    /// or `SQLite` rejects the insert/update.
    pub fn append_turn(&self, session_id: &str, role: Role, content: &str) -> Result<()> {
        let created_at = now_rfc3339()?;
        let changed = self.conn.execute(
            "INSERT INTO turns (session_id, role, content, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![session_id, role.as_str(), content, created_at],
        )?;

        if changed != 1 {
            return Err(anyhow!("session turn insert affected {changed} rows"));
        }

        self.conn.execute(
            "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
            params![created_at, session_id],
        )?;

        Ok(())
    }

    /// Checks whether a session id exists.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` rejects the lookup.
    pub fn session_exists(&self, session_id: &str) -> Result<bool> {
        let exists = self
            .conn
            .query_row(
                "SELECT 1 FROM sessions WHERE id = ?1",
                params![session_id],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        Ok(exists)
    }

    /// Lists turns for a session in creation order.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` rejects the query or persisted role data is invalid.
    pub fn session_turns(&self, session_id: &str) -> Result<Vec<SessionTurn>> {
        let mut statement = self.conn.prepare(
            "SELECT role, content, created_at
             FROM turns
             WHERE session_id = ?1
             ORDER BY id ASC",
        )?;

        let rows = statement.query_map(params![session_id], |row| {
            let role: String = row.get(0)?;
            let role = decode_role(0, &role)?;
            Ok(SessionTurn {
                role,
                content: row.get(1)?,
                created_at: row.get(2)?,
            })
        })?;

        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("list session turns")
    }

    /// Records a tool call before execution and returns its durable id.
    ///
    /// # Errors
    ///
    /// Returns an error when timestamp formatting, JSON serialization, or the insert fails.
    pub fn record_tool_call_started(
        &self,
        session_id: &str,
        name: &str,
        input: &Value,
    ) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let now = now_rfc3339()?;
        let input_json = serde_json::to_string(input).context("encode tool input")?;

        self.conn
            .execute(
                "INSERT INTO tool_calls
                    (id, session_id, name, input_json, status, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    id,
                    session_id,
                    name,
                    input_json,
                    ToolCallStatus::Running.as_str(),
                    now
                ],
            )
            .context("record tool call start")?;

        Ok(id)
    }

    /// Marks a tool call completed and stores the structured output.
    ///
    /// # Errors
    ///
    /// Returns an error when timestamp formatting, JSON serialization, or the update fails.
    pub fn record_tool_call_completed(&self, id: &str, output: &Value) -> Result<()> {
        let now = now_rfc3339()?;
        let output_json = serde_json::to_string(output).context("encode tool output")?;

        let changed = self
            .conn
            .execute(
                "UPDATE tool_calls
                 SET status = ?1, output_json = ?2, error = NULL, completed_at = ?3
                 WHERE id = ?4",
                params![ToolCallStatus::Completed.as_str(), output_json, now, id],
            )
            .context("record tool call completion")?;
        if changed != 1 {
            return Err(anyhow!("tool call {id} does not exist"));
        }
        Ok(())
    }

    /// Marks a tool call failed and stores the failure text.
    ///
    /// # Errors
    ///
    /// Returns an error when timestamp formatting or the update fails.
    pub fn record_tool_call_failed(&self, id: &str, error: &str) -> Result<()> {
        let now = now_rfc3339()?;

        let changed = self
            .conn
            .execute(
                "UPDATE tool_calls
                 SET status = ?1, output_json = NULL, error = ?2, completed_at = ?3
                 WHERE id = ?4",
                params![ToolCallStatus::Failed.as_str(), error, now, id],
            )
            .context("record tool call failure")?;
        if changed != 1 {
            return Err(anyhow!("tool call {id} does not exist"));
        }
        Ok(())
    }

    /// Lists persisted tool calls for a session in creation order.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` rejects the query, persisted status data is invalid,
    /// or persisted JSON cannot be decoded.
    pub fn session_tool_calls(&self, session_id: &str) -> Result<Vec<ToolCallRecord>> {
        let mut statement = self.conn.prepare(
            "SELECT id, session_id, name, input_json, status, output_json, error, created_at,
                    completed_at
             FROM tool_calls
             WHERE session_id = ?1
             ORDER BY rowid ASC",
        )?;

        let rows = statement.query_map(params![session_id], |row| {
            let input_json: String = row.get(3)?;
            let status: String = row.get(4)?;
            let output_json: Option<String> = row.get(5)?;
            Ok(ToolCallRecord {
                id: row.get(0)?,
                session_id: row.get(1)?,
                name: row.get(2)?,
                input: decode_json(3, &input_json)?,
                status: decode_tool_call_status(4, &status)?,
                output: output_json
                    .as_deref()
                    .map(|value| decode_json(5, value))
                    .transpose()?,
                error: row.get(6)?,
                created_at: row.get(7)?,
                completed_at: row.get(8)?,
            })
        })?;

        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("list session tool calls")
    }

    /// Lists recently updated sessions.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` rejects the query.
    pub fn list_sessions(&self, limit: u16) -> Result<Vec<SessionSummary>> {
        let mut statement = self.conn.prepare(
            "SELECT
                s.id,
                s.title,
                s.created_at,
                s.updated_at,
                COUNT(t.id) AS turn_count
             FROM sessions s
             LEFT JOIN turns t ON t.session_id = s.id
             GROUP BY s.id
             ORDER BY s.updated_at DESC
             LIMIT ?1",
        )?;

        let rows = statement.query_map(params![i64::from(limit)], |row| {
            let turn_count: i64 = row.get(4)?;
            Ok(SessionSummary {
                id: row.get(0)?,
                title: row.get(1)?,
                created_at: row.get(2)?,
                updated_at: row.get(3)?,
                turn_count: turn_count.max(0).cast_unsigned(),
            })
        })?;

        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("list sessions")
    }

    /// Searches indexed session turns.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` rejects the full-text query or persisted role data is invalid.
    pub fn search(&self, query: &str, limit: u16) -> Result<Vec<SearchHit>> {
        let Some(fts_query) = build_fts_query(query) else {
            return Ok(Vec::new());
        };

        let mut statement = self.conn.prepare(
            "SELECT
                t.session_id,
                s.title,
                t.role,
                t.content
             FROM turns_fts
             JOIN turns t ON t.id = turns_fts.rowid
             JOIN sessions s ON s.id = t.session_id
             WHERE turns_fts MATCH ?1
             ORDER BY bm25(turns_fts)
             LIMIT ?2",
        )?;

        let rows = statement.query_map(params![fts_query, i64::from(limit)], |row| {
            let role: String = row.get(2)?;
            let role = decode_role(2, &role)?;
            Ok(SearchHit {
                session_id: row.get(0)?,
                title: row.get(1)?,
                role,
                content: row.get(3)?,
            })
        })?;

        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("search sessions")
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "
                PRAGMA foreign_keys = ON;
                PRAGMA busy_timeout = 5000;
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = NORMAL;

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session_id ON turns(session_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);

                CREATE TABLE IF NOT EXISTS tool_calls (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
                    output_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_tool_calls_session_id
                    ON tool_calls(session_id, created_at);

                CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                    session_id UNINDEXED,
                    role UNINDEXED,
                    content,
                    content='turns',
                    content_rowid='id'
                );

                CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
                    INSERT INTO turns_fts(rowid, session_id, role, content)
                    VALUES (new.id, new.session_id, new.role, new.content);
                END;

                CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
                    INSERT INTO turns_fts(turns_fts, rowid, session_id, role, content)
                    VALUES ('delete', old.id, old.session_id, old.role, old.content);
                END;

                CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON turns BEGIN
                    INSERT INTO turns_fts(turns_fts, rowid, session_id, role, content)
                    VALUES ('delete', old.id, old.session_id, old.role, old.content);
                    INSERT INTO turns_fts(rowid, session_id, role, content)
                    VALUES (new.id, new.session_id, new.role, new.content);
                END;
                ",
            )
            .context("migrate session database")
    }
}

fn decode_role(column: usize, role: &str) -> rusqlite::Result<Role> {
    Role::from_db(role).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            column,
            Type::Text,
            Box::new(InvalidRole(role.to_string())),
        )
    })
}

fn decode_tool_call_status(column: usize, status: &str) -> rusqlite::Result<ToolCallStatus> {
    ToolCallStatus::from_db(status).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            column,
            Type::Text,
            Box::new(InvalidToolCallStatus(status.to_string())),
        )
    })
}

fn decode_json(column: usize, value: &str) -> rusqlite::Result<Value> {
    serde_json::from_str(value).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(column, Type::Text, Box::new(error))
    })
}

#[derive(Debug)]
struct InvalidRole(String);

impl fmt::Display for InvalidRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid session role {}", self.0)
    }
}

impl Error for InvalidRole {}

#[derive(Debug)]
struct InvalidToolCallStatus(String);

impl fmt::Display for InvalidToolCallStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid tool call status {}", self.0)
    }
}

impl Error for InvalidToolCallStatus {}

fn now_rfc3339() -> Result<String> {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .context("format timestamp")
}

fn normalize_title(title: &str) -> String {
    let title = title.trim();
    if title.is_empty() {
        return "untitled session".to_string();
    }

    title.chars().take(96).collect()
}

fn build_fts_query(query: &str) -> Option<String> {
    let terms = query
        .split_whitespace()
        .map(str::trim)
        .filter(|term| !term.is_empty())
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect::<Vec<_>>();

    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" AND "))
    }
}
