use anyhow::{Context, Result, bail};
use futures::future::join_all;
use rusqlite::types::Type;
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;
use std::time::Duration;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
use tokio::time::timeout;
use uuid::Uuid;

use crate::config::AppConfig;
use crate::llm::{ChatRequest, LlmClient};
use crate::messages::{Message, Role};
use crate::session::SessionStore;
use crate::wiki::WikiClient;

const RECALL_LIMIT: u8 = 6;
const RECALL_DEPTH: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AgentProfile {
    Supervisor,
    Planner,
    Researcher,
    Coder,
    Reviewer,
}

impl AgentProfile {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Supervisor => "supervisor",
            Self::Planner => "planner",
            Self::Researcher => "researcher",
            Self::Coder => "coder",
            Self::Reviewer => "reviewer",
        }
    }

    #[must_use]
    pub fn from_db(value: &str) -> Option<Self> {
        match value {
            "supervisor" => Some(Self::Supervisor),
            "planner" => Some(Self::Planner),
            "researcher" => Some(Self::Researcher),
            "coder" => Some(Self::Coder),
            "reviewer" => Some(Self::Reviewer),
            _ => None,
        }
    }
}

impl fmt::Display for AgentProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str((*self).as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerSpec {
    pub profile: AgentProfile,
    pub objective: String,
    pub output_format: String,
    pub tools_allowed: Vec<String>,
    pub max_iterations: u16,
    pub max_tokens: u32,
    pub timeout_seconds: u64,
    pub justification: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiAgentRunRequest {
    pub objective: String,
    pub recall: bool,
    pub max_workers: usize,
    pub workers: Vec<WorkerSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiAgentRunOutput {
    pub run_id: String,
    pub supervisor_session_id: String,
    pub synthesis: String,
    pub workers: Vec<WorkerRunOutput>,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkerRunOutput {
    pub task_id: String,
    pub profile: AgentProfile,
    pub session_id: String,
    pub status: TaskStatus,
    pub summary: String,
    pub error: Option<String>,
    pub tokens_used: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
}

impl TaskStatus {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
        }
    }

    #[must_use]
    pub fn from_db(value: &str) -> Option<Self> {
        match value {
            "pending" => Some(Self::Pending),
            "running" => Some(Self::Running),
            "succeeded" => Some(Self::Succeeded),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str((*self).as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StoredRun {
    pub id: String,
    pub objective: String,
    pub status: TaskStatus,
    pub supervisor_session_id: Option<String>,
    pub synthesis: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub tasks: Vec<StoredTask>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StoredTask {
    pub id: String,
    pub run_id: String,
    pub profile: AgentProfile,
    pub status: TaskStatus,
    pub session_id: String,
    pub summary: Option<String>,
    pub error: Option<String>,
    pub tokens_used: u64,
}

#[derive(Debug)]
pub struct TaskGraphStore {
    conn: Connection,
}

impl TaskGraphStore {
    /// Opens the task graph database inside the configured state directory.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory cannot be created, the database cannot be opened,
    /// or migrations fail.
    pub fn open(state_dir: impl AsRef<Path>) -> Result<Self> {
        fs::create_dir_all(state_dir.as_ref()).with_context(|| {
            format!(
                "create task graph state directory {}",
                state_dir.as_ref().display()
            )
        })?;
        Self::open_database(state_dir.as_ref().join("tasks.sqlite3"))
    }

    /// Opens a task graph database at an explicit path.
    ///
    /// # Errors
    ///
    /// Returns an error when the database cannot be opened or migrations fail.
    pub fn open_database(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref())
            .with_context(|| format!("open task graph database {}", path.as_ref().display()))?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    /// Opens an in-memory task graph database for tests and short-lived workflows.
    ///
    /// # Errors
    ///
    /// Returns an error when the `SQLite` connection cannot be created or migrations fail.
    pub fn in_memory() -> Result<Self> {
        let store = Self {
            conn: Connection::open_in_memory().context("open in-memory task graph database")?,
        };
        store.migrate()?;
        Ok(store)
    }

    /// Returns one persisted run and its tasks.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` rejects the lookup or persisted enum data is invalid.
    pub fn run(&self, run_id: &str) -> Result<Option<StoredRun>> {
        let mut run = self
            .conn
            .query_row(
                "SELECT id, objective, status, supervisor_session_id, synthesis, created_at, updated_at
                 FROM agent_runs
                 WHERE id = ?1",
                params![run_id],
                |row| {
                    let status: String = row.get(2)?;
                    Ok(StoredRun {
                        id: row.get(0)?,
                        objective: row.get(1)?,
                        status: decode_status(2, &status)?,
                        supervisor_session_id: row.get(3)?,
                        synthesis: row.get(4)?,
                        created_at: row.get(5)?,
                        updated_at: row.get(6)?,
                        tasks: Vec::new(),
                    })
                },
            )
            .optional()?;

        if let Some(run) = &mut run {
            run.tasks = self.tasks_for_run(run_id)?;
        }

        Ok(run)
    }

    fn create_run(&self, objective: &str) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let now = now_rfc3339()?;
        self.conn.execute(
            "INSERT INTO agent_runs (id, objective, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, objective.trim(), TaskStatus::Running.as_str(), now, now],
        )?;
        Ok(id)
    }

    fn add_task(&self, run_id: &str, spec: &WorkerSpec, session_id: &str) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let now = now_rfc3339()?;
        let tools_allowed = serde_json::to_string(&spec.tools_allowed)?;
        self.conn.execute(
            "INSERT INTO agent_tasks (
                id, run_id, profile, objective, status, session_id, output_format,
                tools_allowed, max_iterations, max_tokens, timeout_seconds, justification,
                created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                id,
                run_id,
                spec.profile.as_str(),
                spec.objective.trim(),
                TaskStatus::Pending.as_str(),
                session_id,
                spec.output_format.trim(),
                tools_allowed,
                i64::from(spec.max_iterations),
                i64::from(spec.max_tokens),
                i64::try_from(spec.timeout_seconds).context("timeout seconds exceed i64")?,
                spec.justification.trim(),
                now,
                now
            ],
        )?;
        Ok(id)
    }

    fn mark_task_running(&self, task_id: &str) -> Result<()> {
        self.update_task_status(task_id, TaskStatus::Running)
    }

    fn complete_task(&self, task_id: &str, summary: &str, tokens_used: u64) -> Result<()> {
        let now = now_rfc3339()?;
        self.conn.execute(
            "UPDATE agent_tasks
             SET status = ?1, summary = ?2, error = NULL, tokens_used = ?3, updated_at = ?4
             WHERE id = ?5",
            params![
                TaskStatus::Succeeded.as_str(),
                summary.trim(),
                i64::try_from(tokens_used).context("token count exceeds i64")?,
                now,
                task_id
            ],
        )?;
        Ok(())
    }

    fn fail_task(&self, task_id: &str, error: &str) -> Result<()> {
        let now = now_rfc3339()?;
        self.conn.execute(
            "UPDATE agent_tasks
             SET status = ?1, error = ?2, updated_at = ?3
             WHERE id = ?4",
            params![TaskStatus::Failed.as_str(), error, now, task_id],
        )?;
        Ok(())
    }

    fn finish_run(&self, run_id: &str, supervisor_session_id: &str, synthesis: &str) -> Result<()> {
        let now = now_rfc3339()?;
        self.conn.execute(
            "UPDATE agent_runs
             SET status = ?1, supervisor_session_id = ?2, synthesis = ?3, updated_at = ?4
             WHERE id = ?5",
            params![
                TaskStatus::Succeeded.as_str(),
                supervisor_session_id,
                synthesis.trim(),
                now,
                run_id
            ],
        )?;
        Ok(())
    }

    fn fail_run(&self, run_id: &str, error: &str) -> Result<()> {
        let now = now_rfc3339()?;
        self.conn.execute(
            "UPDATE agent_runs
             SET status = ?1, synthesis = ?2, updated_at = ?3
             WHERE id = ?4",
            params![TaskStatus::Failed.as_str(), error, now, run_id],
        )?;
        Ok(())
    }

    fn update_task_status(&self, task_id: &str, status: TaskStatus) -> Result<()> {
        let now = now_rfc3339()?;
        self.conn.execute(
            "UPDATE agent_tasks SET status = ?1, updated_at = ?2 WHERE id = ?3",
            params![status.as_str(), now, task_id],
        )?;
        Ok(())
    }

    fn tasks_for_run(&self, run_id: &str) -> Result<Vec<StoredTask>> {
        let mut statement = self.conn.prepare(
            "SELECT id, run_id, profile, status, session_id, summary, error, tokens_used
             FROM agent_tasks
             WHERE run_id = ?1
             ORDER BY id ASC",
        )?;
        let rows = statement.query_map(params![run_id], |row| {
            let profile: String = row.get(2)?;
            let status: String = row.get(3)?;
            let tokens_used: i64 = row.get(7)?;
            Ok(StoredTask {
                id: row.get(0)?,
                run_id: row.get(1)?,
                profile: decode_profile(2, &profile)?,
                status: decode_status(3, &status)?,
                session_id: row.get(4)?,
                summary: row.get(5)?,
                error: row.get(6)?,
                tokens_used: tokens_used.max(0).cast_unsigned(),
            })
        })?;

        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("list tasks for run")
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "
                PRAGMA foreign_keys = ON;
                PRAGMA busy_timeout = 5000;
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = NORMAL;

                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    supervisor_session_id TEXT,
                    synthesis TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_tasks (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
                    profile TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    output_format TEXT NOT NULL,
                    tools_allowed TEXT NOT NULL,
                    max_iterations INTEGER NOT NULL,
                    max_tokens INTEGER NOT NULL,
                    timeout_seconds INTEGER NOT NULL,
                    justification TEXT NOT NULL,
                    summary TEXT,
                    error TEXT,
                    tokens_used INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_agent_runs_updated_at
                ON agent_runs(updated_at);

                CREATE INDEX IF NOT EXISTS idx_agent_tasks_run_id
                ON agent_tasks(run_id);
                ",
            )
            .context("migrate task graph database")
    }
}

#[derive(Debug)]
pub struct MultiAgentRuntime {
    config: AppConfig,
    llm: LlmClient,
    wiki: WikiClient,
    sessions: SessionStore,
    tasks: TaskGraphStore,
}

impl MultiAgentRuntime {
    #[must_use]
    pub fn new(config: AppConfig, sessions: SessionStore, tasks: TaskGraphStore) -> Self {
        Self {
            llm: LlmClient::new(
                config.pengepul_base_url.clone(),
                config.pengepul_api_key.clone(),
            ),
            wiki: WikiClient::new(config.wiki_base_url.clone()),
            config,
            sessions,
            tasks,
        }
    }

    /// Runs one supervisor with isolated worker sessions.
    ///
    /// # Errors
    ///
    /// Returns an error when the request contract is invalid, persistence fails, recall fails,
    /// or supervisor synthesis fails.
    pub async fn run(&self, request: MultiAgentRunRequest) -> Result<MultiAgentRunOutput> {
        validate_request(&request)?;

        let shared_memory = if request.recall {
            let memory = self
                .wiki
                .rag(&request.objective, RECALL_LIMIT, RECALL_DEPTH)
                .await?;
            Some(memory.markdown.trim().to_string()).filter(|memory| !memory.is_empty())
        } else {
            None
        };

        let run_id = self.tasks.create_run(&request.objective)?;
        let worker_calls = self.prepare_worker_calls(&run_id, &request)?;
        let worker_results = self
            .execute_workers(worker_calls, shared_memory.clone())
            .await?;
        let supervisor_session_id = self
            .sessions
            .create_session(&format!("supervisor: {}", request.objective))?;
        self.sessions
            .append_turn(&supervisor_session_id, Role::User, &request.objective)?;

        let synthesis = self
            .synthesize(&request.objective, &worker_results, shared_memory)
            .await;
        let synthesis = match synthesis {
            Ok(synthesis) => synthesis,
            Err(error) => {
                self.tasks.fail_run(&run_id, &error.to_string())?;
                return Err(error);
            }
        };

        self.sessions
            .append_turn(&supervisor_session_id, Role::Assistant, &synthesis.content)?;
        self.tasks
            .finish_run(&run_id, &supervisor_session_id, &synthesis.content)?;

        Ok(MultiAgentRunOutput {
            run_id,
            supervisor_session_id,
            synthesis: synthesis.content,
            total_tokens: worker_results
                .iter()
                .map(|worker| worker.tokens_used)
                .sum::<u64>()
                + synthesis.tokens_used,
            workers: worker_results,
        })
    }

    fn prepare_worker_calls(
        &self,
        run_id: &str,
        request: &MultiAgentRunRequest,
    ) -> Result<Vec<WorkerCall>> {
        let mut calls = Vec::with_capacity(request.workers.len());
        for spec in &request.workers {
            let session_id = self
                .sessions
                .create_session(&format!("{}: {}", spec.profile, request.objective))?;
            self.sessions
                .append_turn(&session_id, Role::User, &spec.objective)?;
            let task_id = self.tasks.add_task(run_id, spec, &session_id)?;
            self.tasks.mark_task_running(&task_id)?;
            calls.push(WorkerCall {
                task_id,
                session_id,
                spec: spec.clone(),
            });
        }
        Ok(calls)
    }

    async fn execute_workers(
        &self,
        calls: Vec<WorkerCall>,
        shared_memory: Option<String>,
    ) -> Result<Vec<WorkerRunOutput>> {
        let jobs = calls.into_iter().map(|call| {
            let llm = self.llm.clone();
            let model = self.config.model.clone();
            let shared_memory = shared_memory.clone();
            async move { execute_worker(call, llm, model, shared_memory).await }
        });

        let executions = join_all(jobs).await;
        let mut outputs = Vec::with_capacity(executions.len());
        for execution in executions {
            match &execution.error {
                Some(error) => self.tasks.fail_task(&execution.task_id, error)?,
                None => self.tasks.complete_task(
                    &execution.task_id,
                    &execution.summary,
                    execution.tokens_used,
                )?,
            }
            self.sessions.append_turn(
                &execution.session_id,
                Role::Assistant,
                worker_session_output(&execution),
            )?;
            outputs.push(WorkerRunOutput {
                task_id: execution.task_id,
                profile: execution.profile,
                session_id: execution.session_id,
                status: if execution.error.is_some() {
                    TaskStatus::Failed
                } else {
                    TaskStatus::Succeeded
                },
                summary: execution.summary,
                error: execution.error,
                tokens_used: execution.tokens_used,
            });
        }

        Ok(outputs)
    }

    async fn synthesize(
        &self,
        objective: &str,
        workers: &[WorkerRunOutput],
        shared_memory: Option<String>,
    ) -> Result<ModelSummary> {
        let mut messages = Vec::with_capacity(3);
        messages.push(Message::system(supervisor_system_prompt()));
        if let Some(memory) = shared_memory {
            messages.push(Message::system(format!(
                "Shared project memory, read-only:\n\n{memory}"
            )));
        }
        messages.push(Message::user(format!(
            "Objective:\n{objective}\n\nWorker summaries:\n{}",
            format_worker_summaries(workers)
        )));

        let output = self
            .llm
            .chat(ChatRequest {
                model: self.config.model.clone(),
                messages,
            })
            .await?;

        Ok(ModelSummary {
            content: output.content,
            tokens_used: output.usage.map_or(0, |usage| usage.total_tokens),
        })
    }
}

#[derive(Debug, Clone)]
struct WorkerCall {
    task_id: String,
    session_id: String,
    spec: WorkerSpec,
}

#[derive(Debug)]
struct WorkerExecution {
    task_id: String,
    profile: AgentProfile,
    session_id: String,
    summary: String,
    error: Option<String>,
    tokens_used: u64,
}

#[derive(Debug)]
struct ModelSummary {
    content: String,
    tokens_used: u64,
}

async fn execute_worker(
    call: WorkerCall,
    llm: LlmClient,
    model: String,
    shared_memory: Option<String>,
) -> WorkerExecution {
    let timeout_duration = Duration::from_secs(call.spec.timeout_seconds);
    let messages = worker_messages(&call.spec, shared_memory);
    let result = timeout(timeout_duration, llm.chat(ChatRequest { model, messages })).await;

    match result {
        Ok(Ok(output)) => WorkerExecution {
            task_id: call.task_id,
            profile: call.spec.profile,
            session_id: call.session_id,
            summary: output.content,
            error: None,
            tokens_used: output.usage.map_or(0, |usage| usage.total_tokens),
        },
        Ok(Err(error)) => WorkerExecution {
            task_id: call.task_id,
            profile: call.spec.profile,
            session_id: call.session_id,
            summary: String::new(),
            error: Some(error.to_string()),
            tokens_used: 0,
        },
        Err(_) => WorkerExecution {
            task_id: call.task_id,
            profile: call.spec.profile,
            session_id: call.session_id,
            summary: String::new(),
            error: Some(format!(
                "{} worker timed out after {:?}",
                call.spec.profile, timeout_duration
            )),
            tokens_used: 0,
        },
    }
}

fn validate_request(request: &MultiAgentRunRequest) -> Result<()> {
    if request.objective.trim().is_empty() {
        bail!("multi-agent objective is required");
    }
    if request.max_workers == 0 {
        bail!("max workers must be greater than zero");
    }
    if request.workers.is_empty() {
        bail!("at least one worker is required");
    }
    if request.workers.len() > request.max_workers {
        bail!(
            "{} workers requested but max workers is {}",
            request.workers.len(),
            request.max_workers
        );
    }

    for spec in &request.workers {
        validate_worker_spec(spec)?;
    }

    Ok(())
}

fn validate_worker_spec(spec: &WorkerSpec) -> Result<()> {
    if spec.objective.trim().is_empty() {
        bail!("{} worker objective is required", spec.profile);
    }
    if spec.output_format.trim().is_empty() {
        bail!("{} worker output format is required", spec.profile);
    }
    if spec.tools_allowed.is_empty() || spec.tools_allowed.iter().any(|tool| tool.trim().is_empty())
    {
        bail!("{} worker tools_allowed must be explicit", spec.profile);
    }
    if spec.max_iterations == 0 {
        bail!(
            "{} worker max_iterations must be greater than zero",
            spec.profile
        );
    }
    if spec.max_tokens == 0 {
        bail!(
            "{} worker max_tokens must be greater than zero",
            spec.profile
        );
    }
    if spec.timeout_seconds == 0 {
        bail!(
            "{} worker timeout_seconds must be greater than zero",
            spec.profile
        );
    }
    if spec.justification.trim().is_empty() {
        bail!("{} worker justification is required", spec.profile);
    }
    Ok(())
}

fn worker_messages(spec: &WorkerSpec, shared_memory: Option<String>) -> Vec<Message> {
    let mut messages = Vec::with_capacity(3);
    messages.push(Message::system(worker_system_prompt(spec)));
    if let Some(memory) = shared_memory.filter(|memory| !memory.trim().is_empty()) {
        messages.push(Message::system(format!(
            "Shared project memory, read-only:\n\n{}",
            memory.trim()
        )));
    }
    messages.push(Message::user(format!(
        "Worker objective:\n{}\n\nJustification:\n{}\n\nReturn format: {}",
        spec.objective.trim(),
        spec.justification.trim(),
        spec.output_format.trim()
    )));
    messages
}

fn worker_system_prompt(spec: &WorkerSpec) -> String {
    format!(
        "You are a Loka worker.\n\
         Worker profile: {}\n\
         You are an agent-as-tool child, not a handoff owner.\n\
         Do not spawn sub-agents.\n\
         Use a private scratchpad internally, but return only the requested compact output.\n\
         Do not write durable memory. Durable proposals may only be created by the supervisor.\n\
         Allowed tools: {}\n\
         Max iterations: {}\n\
         Max tokens: {}\n\
         Timeout seconds: {}\n\
         Return format: {}",
        spec.profile,
        spec.tools_allowed.join(", "),
        spec.max_iterations,
        spec.max_tokens,
        spec.timeout_seconds,
        spec.output_format.trim()
    )
}

fn supervisor_system_prompt() -> &'static str {
    "You are the Loka supervisor. Supervisor synthesis must merge worker results into one direct answer. Do not include private worker scratchpads. Mention failed workers only when their failure affects confidence. Durable memory proposals are supervisor-only and must not be written unless a separate explicit command asks for it."
}

fn format_worker_summaries(workers: &[WorkerRunOutput]) -> String {
    let mut output = String::new();
    for worker in workers {
        output.push_str("- ");
        output.push_str(worker.profile.as_str());
        output.push_str(" task ");
        output.push_str(&worker.task_id);
        output.push_str(": ");
        match &worker.error {
            Some(error) => {
                output.push_str("failed: ");
                output.push_str(error);
            }
            None => output.push_str(worker.summary.trim()),
        }
        output.push('\n');
    }
    output
}

fn worker_session_output(execution: &WorkerExecution) -> &str {
    execution
        .error
        .as_deref()
        .unwrap_or(execution.summary.as_str())
}

fn decode_profile(column: usize, value: &str) -> rusqlite::Result<AgentProfile> {
    AgentProfile::from_db(value).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            column,
            Type::Text,
            Box::new(InvalidProfile(value.to_string())),
        )
    })
}

fn decode_status(column: usize, value: &str) -> rusqlite::Result<TaskStatus> {
    TaskStatus::from_db(value).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            column,
            Type::Text,
            Box::new(InvalidStatus(value.to_string())),
        )
    })
}

#[derive(Debug)]
struct InvalidProfile(String);

impl fmt::Display for InvalidProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid agent profile {}", self.0)
    }
}

impl Error for InvalidProfile {}

#[derive(Debug)]
struct InvalidStatus(String);

impl fmt::Display for InvalidStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid task status {}", self.0)
    }
}

impl Error for InvalidStatus {}

fn now_rfc3339() -> Result<String> {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .context("format timestamp")
}
