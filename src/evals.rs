use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::messages::Role;
use crate::multi_agent::{AgentProfile, TaskStatus};
use crate::skills::{SkillDraft, validate_draft};

pub const DEFAULT_FIXTURE_DIR: &str = "evals/fixtures";

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct EvalFixture {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(flatten)]
    pub scenario: EvalScenario,
    pub expectations: EvalExpectations,
}

impl EvalFixture {
    #[must_use]
    pub const fn kind(&self) -> EvalKind {
        self.scenario.kind()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum EvalScenario {
    Ask(AskScenario),
    Chat(ChatScenario),
    Learning(LearningScenario),
    SkillCreation(SkillCreationScenario),
    MultiAgent(MultiAgentScenario),
}

impl EvalScenario {
    #[must_use]
    pub const fn kind(&self) -> EvalKind {
        match self {
            Self::Ask(_) => EvalKind::Ask,
            Self::Chat(_) => EvalKind::Chat,
            Self::Learning(_) => EvalKind::Learning,
            Self::SkillCreation(_) => EvalKind::SkillCreation,
            Self::MultiAgent(_) => EvalKind::MultiAgent,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalKind {
    Ask,
    Chat,
    Learning,
    SkillCreation,
    MultiAgent,
}

impl EvalKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::Chat => "chat",
            Self::Learning => "learning",
            Self::SkillCreation => "skill-creation",
            Self::MultiAgent => "multi-agent",
        }
    }
}

impl fmt::Display for EvalKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str((*self).as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct AskScenario {
    pub prompt: String,
    #[serde(default)]
    pub recall: bool,
    #[serde(default)]
    pub system_message: Option<String>,
    #[serde(default)]
    pub memory_context: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ChatScenario {
    pub messages: Vec<String>,
    #[serde(default)]
    pub recall: bool,
    #[serde(default)]
    pub summary_min_turns: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct LearningScenario {
    pub session_id: String,
    pub session_turns: Vec<SessionTurnFixture>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallFixture>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct SkillCreationScenario {
    pub session_id: String,
    pub session_turns: Vec<SessionTurnFixture>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct MultiAgentScenario {
    pub objective: String,
    #[serde(default)]
    pub recall: bool,
    pub max_workers: usize,
    pub workers: Vec<WorkerFixture>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct SessionTurnFixture {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ToolCallFixture {
    pub name: String,
    #[serde(default)]
    pub input: Value,
    pub status: ToolCallFixtureStatus,
    #[serde(default)]
    pub output: Option<Value>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolCallFixtureStatus {
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct WorkerFixture {
    pub profile: AgentProfile,
    pub objective: String,
    pub output_format: String,
    pub tools_allowed: Vec<String>,
    pub max_iterations: u16,
    pub max_tokens: u32,
    pub timeout_seconds: u64,
    pub justification: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct EvalExpectations {
    #[serde(default)]
    pub prompt_markers: Vec<String>,
    #[serde(default)]
    pub forbidden_prompt_markers: Vec<String>,
    #[serde(default)]
    pub output_markers: Vec<String>,
    #[serde(default)]
    pub memory_tags: Vec<String>,
    #[serde(default)]
    pub skill: Option<SkillDraft>,
    #[serde(default)]
    pub workers: Vec<WorkerExpectation>,
    #[serde(default)]
    pub max_prompt_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct WorkerExpectation {
    pub profile: AgentProfile,
    pub status: TaskStatus,
    #[serde(default)]
    pub output_markers: Vec<String>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
}

/// Loads all JSON eval fixtures from a directory in stable path order.
///
/// # Errors
///
/// Returns an error when the directory cannot be read, contains no JSON fixtures,
/// or a fixture cannot be read or decoded.
pub fn load_fixtures(path: impl AsRef<Path>) -> Result<Vec<EvalFixture>> {
    let dir = path.as_ref();
    let mut files = json_files(dir)?;
    files.sort();

    if files.is_empty() {
        bail!("no eval fixture JSON files found in {}", dir.display());
    }

    files
        .iter()
        .map(load_fixture_file)
        .collect::<Result<Vec<_>>>()
}

/// Loads one JSON eval fixture.
///
/// # Errors
///
/// Returns an error when the file cannot be read or decoded.
pub fn load_fixture_file(path: impl AsRef<Path>) -> Result<EvalFixture> {
    let path = path.as_ref();
    let content = fs::read_to_string(path)
        .with_context(|| format!("read eval fixture {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("decode eval fixture {}", path.display()))
}

/// Validates a fixture set and rejects duplicate ids.
///
/// # Errors
///
/// Returns an error when any fixture is structurally invalid or an id is duplicated.
pub fn validate_fixtures(fixtures: &[EvalFixture]) -> Result<()> {
    let mut ids = HashSet::with_capacity(fixtures.len());
    for fixture in fixtures {
        validate_fixture(fixture)?;
        if !ids.insert(fixture.id.as_str()) {
            bail!("duplicate eval fixture id {}", fixture.id);
        }
    }
    Ok(())
}

/// Validates one eval fixture.
///
/// # Errors
///
/// Returns an error when required fields are empty or runtime limits are invalid.
pub fn validate_fixture(fixture: &EvalFixture) -> Result<()> {
    validate_text(&fixture.id, "id")?;
    validate_text(&fixture.title, "title")?;
    validate_strings(&fixture.tags, "tags")?;
    validate_scenario(fixture)?;
    validate_expectations(&fixture.expectations)?;
    Ok(())
}

fn json_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let entries = fs::read_dir(dir)
        .with_context(|| format!("read eval fixture directory {}", dir.display()))?;
    let mut files = Vec::new();
    for entry in entries {
        let path = entry?.path();
        if path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|extension| extension == "json")
        {
            files.push(path);
        }
    }
    Ok(files)
}

fn validate_scenario(fixture: &EvalFixture) -> Result<()> {
    match &fixture.scenario {
        EvalScenario::Ask(scenario) => validate_ask(&fixture.id, scenario),
        EvalScenario::Chat(scenario) => validate_chat(&fixture.id, scenario),
        EvalScenario::Learning(scenario) => validate_learning(&fixture.id, scenario),
        EvalScenario::SkillCreation(scenario) => validate_skill_creation(&fixture.id, scenario),
        EvalScenario::MultiAgent(scenario) => validate_multi_agent(&fixture.id, scenario),
    }
}

fn validate_ask(id: &str, scenario: &AskScenario) -> Result<()> {
    validate_text(&scenario.prompt, &format!("{id}.prompt"))?;
    validate_optional_text(
        scenario.system_message.as_deref(),
        &format!("{id}.system_message"),
    )?;
    validate_strings(&scenario.memory_context, &format!("{id}.memory_context"))
}

fn validate_chat(id: &str, scenario: &ChatScenario) -> Result<()> {
    if scenario.messages.is_empty() {
        bail!("{id}.messages must not be empty");
    }
    validate_strings(&scenario.messages, &format!("{id}.messages"))?;
    if scenario.summary_min_turns.is_some_and(|turns| turns == 0) {
        bail!("{id}.summary_min_turns must be greater than zero");
    }
    Ok(())
}

fn validate_learning(id: &str, scenario: &LearningScenario) -> Result<()> {
    validate_text(&scenario.session_id, &format!("{id}.session_id"))?;
    validate_turns(id, &scenario.session_turns)?;
    for tool_call in &scenario.tool_calls {
        validate_text(&tool_call.name, &format!("{id}.tool_call.name"))?;
        validate_optional_text(tool_call.error.as_deref(), &format!("{id}.tool_call.error"))?;
    }
    Ok(())
}

fn validate_skill_creation(id: &str, scenario: &SkillCreationScenario) -> Result<()> {
    validate_text(&scenario.session_id, &format!("{id}.session_id"))?;
    validate_turns(id, &scenario.session_turns)
}

fn validate_multi_agent(id: &str, scenario: &MultiAgentScenario) -> Result<()> {
    validate_text(&scenario.objective, &format!("{id}.objective"))?;
    if scenario.max_workers == 0 {
        bail!("{id}.max_workers must be greater than zero");
    }
    if scenario.workers.is_empty() {
        bail!("{id}.workers must not be empty");
    }
    if scenario.workers.len() > scenario.max_workers {
        bail!("{id}.workers exceeds max_workers");
    }

    for worker in &scenario.workers {
        validate_worker(id, worker)?;
    }
    Ok(())
}

fn validate_worker(id: &str, worker: &WorkerFixture) -> Result<()> {
    validate_text(&worker.objective, &format!("{id}.worker.objective"))?;
    validate_text(&worker.output_format, &format!("{id}.worker.output_format"))?;
    validate_strings(&worker.tools_allowed, &format!("{id}.worker.tools_allowed"))?;
    validate_text(&worker.justification, &format!("{id}.worker.justification"))?;
    if worker.max_iterations == 0 {
        bail!("{id}.worker.max_iterations must be greater than zero");
    }
    if worker.max_tokens == 0 {
        bail!("{id}.worker.max_tokens must be greater than zero");
    }
    if worker.timeout_seconds == 0 {
        bail!("{id}.worker.timeout_seconds must be greater than zero");
    }
    Ok(())
}

fn validate_turns(id: &str, turns: &[SessionTurnFixture]) -> Result<()> {
    if turns.is_empty() {
        bail!("{id}.session_turns must not be empty");
    }
    for turn in turns {
        validate_text(&turn.content, &format!("{id}.session_turn.content"))?;
    }
    Ok(())
}

fn validate_expectations(expectations: &EvalExpectations) -> Result<()> {
    validate_strings(&expectations.prompt_markers, "expectations.prompt_markers")?;
    validate_strings(
        &expectations.forbidden_prompt_markers,
        "expectations.forbidden_prompt_markers",
    )?;
    validate_strings(&expectations.output_markers, "expectations.output_markers")?;
    validate_strings(&expectations.memory_tags, "expectations.memory_tags")?;

    if let Some(tokens) = expectations.max_prompt_tokens
        && tokens == 0
    {
        bail!("expectations.max_prompt_tokens must be greater than zero");
    }
    if let Some(skill) = &expectations.skill {
        validate_draft(skill)?;
    }
    for worker in &expectations.workers {
        validate_strings(
            &worker.output_markers,
            "expectations.workers.output_markers",
        )?;
        if worker.max_tokens.is_some_and(|tokens| tokens == 0) {
            bail!("expectations.workers.max_tokens must be greater than zero");
        }
    }

    if expectations.prompt_markers.is_empty()
        && expectations.forbidden_prompt_markers.is_empty()
        && expectations.output_markers.is_empty()
        && expectations.memory_tags.is_empty()
        && expectations.skill.is_none()
        && expectations.workers.is_empty()
        && expectations.max_prompt_tokens.is_none()
    {
        bail!("expectations must define at least one assertion");
    }
    Ok(())
}

fn validate_strings(values: &[String], field: &str) -> Result<()> {
    for value in values {
        validate_text(value, field)?;
    }
    Ok(())
}

fn validate_optional_text(value: Option<&str>, field: &str) -> Result<()> {
    if let Some(value) = value {
        validate_text(value, field)?;
    }
    Ok(())
}

fn validate_text(value: &str, field: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field} must not be empty");
    }
    Ok(())
}
