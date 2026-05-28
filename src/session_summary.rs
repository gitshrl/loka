use anyhow::{Result, anyhow};

use crate::config::AppConfig;
use crate::llm::{ChatRequest, LlmClient};
use crate::messages::Message;
use crate::session::{SessionStore, SessionTurn, ToolCallRecord};
use crate::session_context::format_session_context;
use crate::wiki::{NoteInput, WikiClient};

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
    llm: LlmClient,
    wiki: WikiClient,
    sessions: SessionStore,
}

impl SessionSummaryEngine {
    #[must_use]
    pub fn new(config: AppConfig, sessions: SessionStore) -> Self {
        Self {
            llm: LlmClient::new(&config.pengepul_base_url, config.pengepul_api_key.clone()),
            wiki: WikiClient::new(&config.wiki_base_url),
            config,
            sessions,
        }
    }

    /// Summarizes one persisted session and writes the summary as a proposal-first wiki note.
    ///
    /// # Errors
    ///
    /// Returns an error when the session is empty, model summarization fails, or the wiki note
    /// proposal fails.
    pub async fn summarize(&self, request: SessionSummaryRequest) -> Result<SessionSummaryOutput> {
        summarize_session(&self.config, &self.llm, &self.wiki, &self.sessions, request).await
    }
}

pub(crate) async fn summarize_session(
    config: &AppConfig,
    llm: &LlmClient,
    wiki: &WikiClient,
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
    let summary = llm
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

    let proposal_id = wiki
        .add_note(NoteInput {
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
