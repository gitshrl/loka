use anyhow::{Result, anyhow};

use crate::config::AppConfig;
use crate::llm::{ChatRequest, LlmClient};
use crate::messages::Message;
use crate::session::{SessionStore, SessionTurn};
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
        let turns = self.sessions.session_turns(&request.session_id)?;
        if turns.is_empty() {
            return Err(anyhow!("session {} has no turns", request.session_id));
        }
        if turns.len() < request.min_turns {
            return Ok(SessionSummaryOutput::TooShort {
                turn_count: turns.len(),
            });
        }

        let summary = self
            .llm
            .chat(ChatRequest {
                model: self.config.model.clone(),
                messages: vec![
                    Message::system(summary_system_prompt()),
                    Message::user(format_session_for_summary(&request.session_id, &turns)),
                ],
            })
            .await?;

        let proposal_id = self
            .wiki
            .add_note(NoteInput {
                title: format!("Session summary: {}", request.session_id),
                body: summary.content,
                kind: "note".to_string(),
                agent_id: self.config.agent_id.clone(),
                tags: vec!["summary".to_string(), "session".to_string()],
            })
            .await?;

        Ok(SessionSummaryOutput::ProposalCreated { proposal_id })
    }
}

fn summary_system_prompt() -> &'static str {
    "Summarize this Loka session into durable, compact markdown. Preserve decisions, open questions, tool failures, runtime constraints, and next actions. Do not invent facts. Keep the summary useful for future session search and memory review."
}

fn format_session_for_summary(session_id: &str, turns: &[SessionTurn]) -> String {
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
