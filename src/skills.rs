use anyhow::{Context, Result, anyhow};
use rusqlite::types::Type;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SkillStatus {
    Proposed,
    Enabled,
    Disabled,
}

impl SkillStatus {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Proposed => "proposed",
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
        }
    }

    #[must_use]
    pub fn from_db(value: &str) -> Option<Self> {
        match value {
            "proposed" => Some(Self::Proposed),
            "enabled" => Some(Self::Enabled),
            "disabled" => Some(Self::Disabled),
            _ => None,
        }
    }
}

impl fmt::Display for SkillStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillDraft {
    pub name: String,
    pub trigger: String,
    pub instructions: String,
    pub required_tools: Vec<String>,
    pub safety_notes: Vec<String>,
    pub examples: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Skill {
    pub id: String,
    pub name: String,
    pub trigger: String,
    pub instructions: String,
    pub required_tools: Vec<String>,
    pub safety_notes: Vec<String>,
    pub examples: Vec<String>,
    pub status: SkillStatus,
    pub created_at: String,
    pub updated_at: String,
}

impl Skill {
    #[must_use]
    pub fn prompt_block(&self) -> String {
        let mut output = String::with_capacity(self.instructions.len() + 256);
        output.push_str("Skill: ");
        output.push_str(&self.name);
        output.push_str("\nTrigger: ");
        output.push_str(&self.trigger);
        output.push_str("\nInstructions:\n");
        output.push_str(&self.instructions);

        if !self.required_tools.is_empty() {
            output.push_str("\nRequired tools:\n");
            for tool in &self.required_tools {
                output.push_str("- ");
                output.push_str(tool);
                output.push('\n');
            }
        }

        if !self.safety_notes.is_empty() {
            output.push_str("\nSafety notes:\n");
            for note in &self.safety_notes {
                output.push_str("- ");
                output.push_str(note);
                output.push('\n');
            }
        }

        if !self.examples.is_empty() {
            output.push_str("\nExamples:\n");
            for example in &self.examples {
                output.push_str("- ");
                output.push_str(example);
                output.push('\n');
            }
        }

        output
    }
}

#[derive(Debug)]
pub struct SkillStore {
    conn: Connection,
}

impl SkillStore {
    /// Opens the skill database inside the configured state directory.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory cannot be created, database cannot open, or migrations fail.
    pub fn open(state_dir: impl AsRef<Path>) -> Result<Self> {
        fs::create_dir_all(state_dir.as_ref()).with_context(|| {
            format!(
                "create skill state directory {}",
                state_dir.as_ref().display()
            )
        })?;
        Self::open_database(state_dir.as_ref().join("skills.sqlite3"))
    }

    /// Opens a skill database at an explicit path.
    ///
    /// # Errors
    ///
    /// Returns an error when the database cannot open or migrations fail.
    pub fn open_database(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref())
            .with_context(|| format!("open skill database {}", path.as_ref().display()))?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    /// Opens an in-memory skill store for tests.
    ///
    /// # Errors
    ///
    /// Returns an error when the `SQLite` connection cannot be created or migrations fail.
    pub fn in_memory() -> Result<Self> {
        let store = Self {
            conn: Connection::open_in_memory().context("open in-memory skill database")?,
        };
        store.migrate()?;
        Ok(store)
    }

    /// Creates a proposed skill.
    ///
    /// # Errors
    ///
    /// Returns an error when validation fails, timestamp formatting fails, JSON encoding fails,
    /// or `SQLite` rejects the insert.
    pub fn propose(&self, draft: &SkillDraft) -> Result<Skill> {
        validate_draft(draft)?;
        let id = Uuid::new_v4().to_string();
        let now = now_rfc3339()?;
        self.conn.execute(
            "INSERT INTO skills (
                id, name, trigger, instructions, required_tools, safety_notes, examples, status, created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                id,
                draft.name.trim(),
                draft.trigger.trim(),
                draft.instructions.trim(),
                encode_vec(&draft.required_tools)?,
                encode_vec(&draft.safety_notes)?,
                encode_vec(&draft.examples)?,
                SkillStatus::Proposed.as_str(),
                now,
                now,
            ],
        )?;

        self.get(&id)?
            .ok_or_else(|| anyhow!("created skill {id} was not found"))
    }

    /// Enables a proposed or disabled skill.
    ///
    /// # Errors
    ///
    /// Returns an error when timestamp formatting fails, `SQLite` rejects the update,
    /// or the skill id does not exist.
    pub fn enable(&self, id: &str) -> Result<Skill> {
        self.set_status(id, SkillStatus::Enabled)
    }

    /// Looks up a skill by id.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` rejects the lookup or stored JSON/status data is invalid.
    pub fn get(&self, id: &str) -> Result<Option<Skill>> {
        self.conn
            .query_row(
                "SELECT
                    id, name, trigger, instructions, required_tools, safety_notes, examples, status, created_at, updated_at
                 FROM skills
                 WHERE id = ?1",
                params![id],
                row_to_skill,
            )
            .optional()
            .context("get skill")
    }

    /// Lists skills, optionally filtered by status.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` rejects the query or stored JSON/status data is invalid.
    pub fn list(&self, status: Option<SkillStatus>) -> Result<Vec<Skill>> {
        if let Some(status) = status {
            let mut statement = self.conn.prepare(
                "SELECT
                    id, name, trigger, instructions, required_tools, safety_notes, examples, status, created_at, updated_at
                 FROM skills
                 WHERE status = ?1
                 ORDER BY updated_at DESC",
            )?;
            let rows = statement.query_map(params![status.as_str()], row_to_skill)?;
            return rows
                .collect::<rusqlite::Result<Vec<_>>>()
                .context("list skills");
        }

        let mut statement = self.conn.prepare(
            "SELECT
                id, name, trigger, instructions, required_tools, safety_notes, examples, status, created_at, updated_at
             FROM skills
             ORDER BY updated_at DESC",
        )?;
        let rows = statement.query_map([], row_to_skill)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("list skills")
    }

    /// Finds enabled skills whose trigger appears in the prompt.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` rejects the query or stored JSON/status data is invalid.
    pub fn enabled_for_prompt(&self, prompt: &str) -> Result<Vec<Skill>> {
        let prompt = prompt.to_lowercase();
        let skills = self.list(Some(SkillStatus::Enabled))?;
        Ok(skills
            .into_iter()
            .filter(|skill| {
                let trigger = skill.trigger.to_lowercase();
                !trigger.is_empty() && prompt.contains(&trigger)
            })
            .collect())
    }

    fn set_status(&self, id: &str, status: SkillStatus) -> Result<Skill> {
        let now = now_rfc3339()?;
        let changed = self.conn.execute(
            "UPDATE skills SET status = ?1, updated_at = ?2 WHERE id = ?3",
            params![status.as_str(), now, id],
        )?;
        if changed != 1 {
            return Err(anyhow!("skill {id} not found"));
        }
        self.get(id)?.ok_or_else(|| anyhow!("skill {id} not found"))
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "
                PRAGMA foreign_keys = ON;
                PRAGMA busy_timeout = 5000;
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = NORMAL;

                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    instructions TEXT NOT NULL,
                    required_tools TEXT NOT NULL,
                    safety_notes TEXT NOT NULL,
                    examples TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_skills_status ON skills(status);
                CREATE INDEX IF NOT EXISTS idx_skills_updated_at ON skills(updated_at);
                ",
            )
            .context("migrate skill database")
    }
}

fn row_to_skill(row: &rusqlite::Row<'_>) -> rusqlite::Result<Skill> {
    let status: String = row.get(7)?;
    let status = SkillStatus::from_db(&status).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            7,
            Type::Text,
            Box::new(InvalidSkillStatus(status.clone())),
        )
    })?;

    Ok(Skill {
        id: row.get(0)?,
        name: row.get(1)?,
        trigger: row.get(2)?,
        instructions: row.get(3)?,
        required_tools: decode_vec(4, &row.get::<_, String>(4)?)?,
        safety_notes: decode_vec(5, &row.get::<_, String>(5)?)?,
        examples: decode_vec(6, &row.get::<_, String>(6)?)?,
        status,
        created_at: row.get(8)?,
        updated_at: row.get(9)?,
    })
}

pub(crate) fn validate_draft(draft: &SkillDraft) -> Result<()> {
    ensure_non_empty("skill name", &draft.name)?;
    ensure_non_empty("skill trigger", &draft.trigger)?;
    ensure_non_empty("skill instructions", &draft.instructions)?;
    Ok(())
}

fn ensure_non_empty(label: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        Err(anyhow!("{label} is required"))
    } else {
        Ok(())
    }
}

fn encode_vec(values: &[String]) -> Result<String> {
    let values = values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    serde_json::to_string(&values).context("encode skill list field")
}

fn decode_vec(column: usize, value: &str) -> rusqlite::Result<Vec<String>> {
    serde_json::from_str(value).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(column, Type::Text, Box::new(error))
    })
}

fn now_rfc3339() -> Result<String> {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .context("format timestamp")
}

#[derive(Debug)]
struct InvalidSkillStatus(String);

impl fmt::Display for InvalidSkillStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid skill status {}", self.0)
    }
}

impl Error for InvalidSkillStatus {}
