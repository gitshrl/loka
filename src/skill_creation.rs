use anyhow::{Context, Result, anyhow};

use crate::config::AppConfig;
use crate::memory::{MemoryClient, MemoryNoteInput};
use crate::messages::Message;
use crate::model::{ChatRequest, ModelClient};
use crate::session::{SessionStore, SessionTurn};
use crate::skills::{Skill, SkillDraft, SkillStore, validate_draft};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposeSkillFromSessionRequest {
    pub session_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProposeSkillFromSessionOutput {
    ProposalCreated {
        skill: Box<Skill>,
        memory_proposal_id: String,
    },
    NoReusableWorkflow,
}

#[derive(Debug)]
pub struct SkillCreationEngine {
    config: AppConfig,
    model_client: ModelClient,
    memory: MemoryClient,
    sessions: SessionStore,
    skills: SkillStore,
}

impl SkillCreationEngine {
    #[must_use]
    pub fn new(config: AppConfig, sessions: SessionStore, skills: SkillStore) -> Self {
        Self {
            model_client: ModelClient::with_protocol(
                &config.model_base_url,
                config.model_api_key.clone(),
                config.model_protocol,
            ),
            memory: MemoryClient::new(&config.memory_base_url),
            config,
            sessions,
            skills,
        }
    }

    /// Proposes one reusable skill from a persisted session.
    ///
    /// # Errors
    ///
    /// Returns an error when the session is empty, model extraction fails, model output is invalid,
    /// `memory API` rejects the proposal-first note, or the local skill store rejects the draft.
    pub async fn propose_from_session(
        &self,
        request: ProposeSkillFromSessionRequest,
    ) -> Result<ProposeSkillFromSessionOutput> {
        let turns = self.sessions.session_turns(&request.session_id)?;
        if turns.is_empty() {
            return Err(anyhow!("session {} has no turns", request.session_id));
        }

        let extraction = self
            .model_client
            .chat(ChatRequest {
                model: self.config.model.clone(),
                messages: vec![
                    Message::system(skill_creation_system_prompt()),
                    Message::user(format_session_for_skill_creation(
                        &request.session_id,
                        &turns,
                    )),
                ],
            })
            .await?;

        let Some(draft) = parse_skill_draft(&extraction.content)? else {
            return Ok(ProposeSkillFromSessionOutput::NoReusableWorkflow);
        };

        validate_draft(&draft)?;
        let memory_proposal_id = self
            .memory
            .propose_note(MemoryNoteInput {
                title: format!("Skill proposal: {}", draft.name.trim()),
                body: format_skill_proposal_note(&request.session_id, &draft),
                kind: "note".to_string(),
                agent_id: self.config.agent_id.clone(),
                tags: vec![
                    "skill".to_string(),
                    "proposal".to_string(),
                    "session".to_string(),
                ],
            })
            .await?;
        let skill = self.skills.propose(&draft)?;

        Ok(ProposeSkillFromSessionOutput::ProposalCreated {
            skill: Box::new(skill),
            memory_proposal_id,
        })
    }
}

fn parse_skill_draft(content: &str) -> Result<Option<SkillDraft>> {
    let content = content.trim();
    if content.eq_ignore_ascii_case("NONE") {
        return Ok(None);
    }

    serde_json::from_str(content)
        .map(Some)
        .context("parse skill proposal JSON")
}

fn skill_creation_system_prompt() -> &'static str {
    "Create exactly one Loka skill proposal when this session contains a repeated workflow that should become reusable. Return strict JSON with these snake_case keys: name, trigger, instructions, required_tools, safety_notes, examples. Use short concrete triggers. Keep instructions operational and tool-aware. If no reusable workflow exists, return exactly NONE."
}

fn format_session_for_skill_creation(session_id: &str, turns: &[SessionTurn]) -> String {
    let mut output =
        String::with_capacity(256 + turns.iter().map(|turn| turn.content.len()).sum::<usize>());
    output.push_str("Session id: ");
    output.push_str(session_id);
    output.push_str("\n\n");

    for turn in turns {
        output.push_str(turn.role.as_str());
        output.push_str(": ");
        output.push_str(turn.content.trim());
        output.push_str("\n\n");
    }

    output
}

fn format_skill_proposal_note(session_id: &str, draft: &SkillDraft) -> String {
    let mut output = String::with_capacity(
        draft.name.len()
            + draft.trigger.len()
            + draft.instructions.len()
            + draft.required_tools.iter().map(String::len).sum::<usize>()
            + draft.safety_notes.iter().map(String::len).sum::<usize>()
            + draft.examples.iter().map(String::len).sum::<usize>()
            + 512,
    );
    output.push_str("# Skill Proposal\n\n");
    output.push_str("Source session: ");
    output.push_str(session_id);
    output.push_str("\n\nName: ");
    output.push_str(draft.name.trim());
    output.push_str("\nTrigger: ");
    output.push_str(draft.trigger.trim());
    output.push_str("\n\n## Instructions\n");
    output.push_str(draft.instructions.trim());
    output.push('\n');

    push_list(&mut output, "Required tools", &draft.required_tools);
    push_list(&mut output, "Safety notes", &draft.safety_notes);
    push_list(&mut output, "Examples", &draft.examples);

    output
}

fn push_list(output: &mut String, title: &str, values: &[String]) {
    if values.is_empty() {
        return;
    }

    output.push_str("\n## ");
    output.push_str(title);
    output.push('\n');
    for value in values {
        let value = value.trim();
        if !value.is_empty() {
            output.push_str("- ");
            output.push_str(value);
            output.push('\n');
        }
    }
}
