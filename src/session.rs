use anyhow::{Context, Result, anyhow};
use rusqlite::types::Type;
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
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

#[derive(Debug)]
struct InvalidRole(String);

impl fmt::Display for InvalidRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid session role {}", self.0)
    }
}

impl Error for InvalidRole {}

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
