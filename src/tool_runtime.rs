use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::session::SessionStore;
use crate::wiki::{NoteInput, WikiClient};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub input: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ToolResult {
    pub output: Value,
}

#[derive(Debug)]
pub struct ToolRuntime {
    sessions: SessionStore,
    wiki: Option<WikiClient>,
    agent_id: String,
}

#[derive(Debug, Deserialize)]
struct SessionListInput {
    limit: Option<u16>,
}

#[derive(Debug, Deserialize)]
struct SessionSearchInput {
    query: String,
    limit: Option<u16>,
}

#[derive(Debug, Deserialize)]
struct WikiRagInput {
    query: String,
    limit: Option<u8>,
    depth: Option<u8>,
}

#[derive(Debug, Deserialize)]
struct WikiAddNoteInput {
    title: String,
    body: String,
    tags: Option<Vec<String>>,
}

impl ToolRuntime {
    #[must_use]
    pub fn new(sessions: SessionStore) -> Self {
        Self {
            sessions,
            wiki: None,
            agent_id: "loka-agent".to_string(),
        }
    }

    #[must_use]
    pub fn with_wiki(mut self, wiki: WikiClient, agent_id: impl Into<String>) -> Self {
        self.wiki = Some(wiki);
        self.agent_id = agent_id.into();
        self
    }

    /// Executes a supported service-backed tool call.
    ///
    /// # Errors
    ///
    /// Returns an error when tool input is invalid, required services are not configured,
    /// the backing service rejects the request, or the registered tool has no runtime executor yet.
    pub async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        match call.name.as_str() {
            "session_list" => self.execute_session_list(call.input),
            "session_search" => self.execute_session_search(call.input),
            "wiki_rag" => self.execute_wiki_rag(call.input).await,
            "wiki_add_note" => self.execute_wiki_add_note(call.input).await,
            name => Err(anyhow!("tool {name} has no runtime executor")),
        }
    }

    fn execute_session_list(&self, input: Value) -> Result<ToolResult> {
        let input: SessionListInput = serde_json::from_value(input)?;
        let sessions = self.sessions.list_sessions(input.limit.unwrap_or(20))?;
        Ok(ToolResult {
            output: json!({ "sessions": sessions }),
        })
    }

    fn execute_session_search(&self, input: Value) -> Result<ToolResult> {
        let input: SessionSearchInput = serde_json::from_value(input)?;
        let hits = self
            .sessions
            .search(&input.query, input.limit.unwrap_or(20))?;
        Ok(ToolResult {
            output: json!({ "hits": hits }),
        })
    }

    async fn execute_wiki_rag(&self, input: Value) -> Result<ToolResult> {
        let input: WikiRagInput = serde_json::from_value(input)?;
        let wiki = self
            .wiki
            .as_ref()
            .ok_or_else(|| anyhow!("wiki_rag requires personal-wiki configuration"))?;
        let context = wiki
            .rag(
                &input.query,
                input.limit.unwrap_or(6),
                input.depth.unwrap_or(1),
            )
            .await?;
        Ok(ToolResult {
            output: json!({ "context": context }),
        })
    }

    async fn execute_wiki_add_note(&self, input: Value) -> Result<ToolResult> {
        let input: WikiAddNoteInput = serde_json::from_value(input)?;
        let wiki = self
            .wiki
            .as_ref()
            .ok_or_else(|| anyhow!("wiki_add_note requires personal-wiki configuration"))?;
        let proposal_id = wiki
            .add_note(NoteInput {
                title: input.title,
                body: input.body,
                kind: "note".to_string(),
                agent_id: self.agent_id.clone(),
                tags: input.tags.unwrap_or_default(),
            })
            .await?;
        Ok(ToolResult {
            output: json!({ "proposal_id": proposal_id }),
        })
    }
}
