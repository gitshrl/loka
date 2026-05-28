use anyhow::{Result, anyhow};

use crate::config::AppConfig;
use crate::llm::{ChatRequest, LlmClient};
use crate::messages::Message;
use crate::session::{SessionStore, SessionTurn, ToolCallRecord};
use crate::session_context::format_session_context;
use crate::wiki::{NoteInput, PendingProposal, WikiClient};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LearnSessionRequest {
    pub session_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LearnSessionOutput {
    ProposalCreated { proposal_id: String },
    NoDurableKnowledge,
}

#[derive(Debug)]
pub struct LearningEngine {
    config: AppConfig,
    llm: LlmClient,
    wiki: WikiClient,
    sessions: SessionStore,
}

impl LearningEngine {
    #[must_use]
    pub fn new(config: AppConfig, sessions: SessionStore) -> Self {
        Self {
            llm: LlmClient::new(&config.pengepul_base_url, config.pengepul_api_key.clone()),
            wiki: WikiClient::new(&config.wiki_base_url),
            config,
            sessions,
        }
    }

    /// Extracts durable knowledge from a persisted session and writes it as a proposal.
    ///
    /// # Errors
    ///
    /// Returns an error when the session does not exist, turn retrieval fails, model extraction
    /// fails, or `personal-wiki` rejects the proposal-first note write.
    pub async fn learn_session(&self, request: LearnSessionRequest) -> Result<LearnSessionOutput> {
        learn_from_session(&self.config, &self.llm, &self.wiki, &self.sessions, request).await
    }
}

pub(crate) async fn learn_from_session(
    config: &AppConfig,
    llm: &LlmClient,
    wiki: &WikiClient,
    sessions: &SessionStore,
    request: LearnSessionRequest,
) -> Result<LearnSessionOutput> {
    let turns = sessions.session_turns(&request.session_id)?;
    if turns.is_empty() {
        return Err(anyhow!("session {} has no turns", request.session_id));
    }

    let tool_calls = sessions.session_tool_calls(&request.session_id)?;
    let extraction = llm
        .chat(ChatRequest {
            model: config.model.clone(),
            messages: vec![
                Message::system(learning_system_prompt()),
                Message::user(format_session_for_learning(
                    &request.session_id,
                    &turns,
                    &tool_calls,
                )),
            ],
        })
        .await?;

    let body = extraction.content.trim();
    if body.eq_ignore_ascii_case("NONE") {
        return Ok(LearnSessionOutput::NoDurableKnowledge);
    }

    let proposal_id = wiki
        .add_note(NoteInput {
            title: format!("Session learning: {}", request.session_id),
            body: body.to_string(),
            kind: "note".to_string(),
            agent_id: config.agent_id.clone(),
            tags: vec!["learning".to_string(), "session".to_string()],
        })
        .await?;

    Ok(LearnSessionOutput::ProposalCreated { proposal_id })
}

/// Lists pending learning proposals from `personal-wiki`.
///
/// # Errors
///
/// Returns an error when `personal-wiki` cannot list pending proposals.
pub async fn pending_learning_proposals(
    wiki: &WikiClient,
    limit: u16,
) -> Result<Vec<PendingProposal>> {
    let proposals = wiki.pending_proposals(limit).await?;
    Ok(proposals.into_iter().filter(is_learning_proposal).collect())
}

fn is_learning_proposal(proposal: &PendingProposal) -> bool {
    proposal.tags.iter().any(|tag| tag == "learning")
        || proposal.title.starts_with("Session learning:")
}

fn learning_system_prompt() -> &'static str {
    "Extract only durable knowledge from this session: user preferences, project facts, decisions, recurring workflows, or tool failures. Return concise markdown. If there is nothing durable, return exactly NONE."
}

fn format_session_for_learning(
    session_id: &str,
    turns: &[SessionTurn],
    tool_calls: &[ToolCallRecord],
) -> String {
    format_session_context(session_id, turns, tool_calls)
}
