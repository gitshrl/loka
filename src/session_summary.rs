use anyhow::{Result, anyhow};

use crate::config::AppConfig;
use crate::memory::{MemoryClient, MemoryNoteInput};
use crate::messages::Message;
use crate::model::{ChatRequest, ModelClient};
use crate::session::{SessionStore, SessionTurn, ToolCallRecord};
use crate::session_context::format_session_context;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSummaryRequest {
    pub session_id: String,
    pub min_turns: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionSummaryOutput {
    ProposalCreated { proposal_id: String },
    TooShort { turn_count: usize },
}

#[derive(Debug)]
pub struct SessionSummaryEngine {
    config: AppConfig,
    model_client: ModelClient,
    memory: MemoryClient,
    sessions: SessionStore,
}

impl SessionSummaryEngine {
    #[must_use]
    pub fn new(config: AppConfig, sessions: SessionStore) -> Self {
        Self {
            model_client: ModelClient::with_protocol(
                &config.model_base_url,
                config.model_api_key.clone(),
                config.model_protocol,
            ),
            memory: MemoryClient::new(&config.memory_base_url),
            config,
            sessions,
        }
    }

    /// Summarizes one persisted session and writes the summary as a proposal-first memory note.
    ///
    /// # Errors
    ///
    /// Returns an error when the session is empty, model summarization fails, or the memory note
    /// proposal fails.
    pub async fn summarize(&self, request: SessionSummaryRequest) -> Result<SessionSummaryOutput> {
        summarize_session(
            &self.config,
            &self.model_client,
            &self.memory,
            &self.sessions,
            request,
        )
        .await
    }
}

pub(crate) async fn summarize_session(
    config: &AppConfig,
    model_client: &ModelClient,
    memory: &MemoryClient,
    sessions: &SessionStore,
    request: SessionSummaryRequest,
) -> Result<SessionSummaryOutput> {
    let turns = sessions.session_turns(&request.session_id)?;
    if turns.is_empty() {
        return Err(anyhow!("session {} has no turns", request.session_id));
    }
    if turns.len() < request.min_turns {
        return Ok(SessionSummaryOutput::TooShort {
            turn_count: turns.len(),
        });
    }

    let tool_calls = sessions.session_tool_calls(&request.session_id)?;
    let summary = model_client
        .chat(ChatRequest {
            model: config.model.clone(),
            messages: vec![
                Message::system(summary_system_prompt()),
                Message::user(format_session_for_summary(
                    &request.session_id,
                    &turns,
                    &tool_calls,
                )),
            ],
        })
        .await?;

    let proposal_id = memory
        .propose_note(MemoryNoteInput {
            title: format!("Session summary: {}", request.session_id),
            body: summary.content,
            kind: "note".to_string(),
            agent_id: config.agent_id.clone(),
            tags: vec!["summary".to_string(), "session".to_string()],
        })
        .await?;

    Ok(SessionSummaryOutput::ProposalCreated { proposal_id })
}

fn summary_system_prompt() -> &'static str {
    "Summarize this Loka session into durable, compact markdown. Preserve decisions, open questions, tool failures, runtime constraints, and next actions. Do not invent facts. Keep the summary useful for future session search and memory review."
}

fn format_session_for_summary(
    session_id: &str,
    turns: &[SessionTurn],
    tool_calls: &[ToolCallRecord],
) -> String {
    format_session_context(session_id, turns, tool_calls)
}
