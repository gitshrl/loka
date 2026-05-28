use anyhow::{Context, Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct MemoryClient {
    http: Client,
    base_url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct MemoryRecallOutput {
    pub mode: String,
    pub markdown: String,
}

#[derive(Debug, Serialize)]
struct RecallRequest<'a> {
    query: &'a str,
    limit: u8,
    depth: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPrefetchInput {
    pub query: String,
    pub limit: u8,
    pub depth: u8,
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryTurnInput {
    pub session_id: Option<String>,
    pub user: String,
    pub assistant: String,
    pub agent_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySessionEndInput {
    pub session_id: String,
    pub agent_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryShutdownInput {
    pub agent_id: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PrefetchRequest {
    query: String,
    limit: u8,
    depth: u8,
    session_id: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TurnRequest {
    session_id: Option<String>,
    user: String,
    assistant: String,
    agent_id: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SessionEndRequest {
    session_id: String,
    agent_id: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ShutdownRequest {
    agent_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SessionEndResponse {
    proposal_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryNoteInput {
    pub title: String,
    pub body: String,
    pub kind: String,
    pub agent_id: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct NoteRequest {
    title: String,
    body: String,
    kind: String,
    agent_id: String,
    tags: Vec<String>,
    mode: WriteMode,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
enum WriteMode {
    Propose,
}

#[derive(Debug, Deserialize)]
struct NoteResponse {
    mode: String,
    proposal: Option<NoteProposal>,
}

#[derive(Debug, Deserialize)]
struct NoteProposal {
    id: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct PendingProposal {
    pub id: String,
    pub title: String,
    pub kind: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PendingProposalsQuery {
    status: ProposalStatus,
    limit: u16,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
enum ProposalStatus {
    Pending,
}

#[derive(Debug, Deserialize)]
struct PendingProposalsResponse {
    proposals: Vec<PendingProposal>,
}

impl MemoryClient {
    #[must_use]
    pub fn new(base_url: impl AsRef<str>) -> Self {
        Self {
            http: Client::new(),
            base_url: base_url.as_ref().trim_end_matches('/').to_string(),
        }
    }

    /// Fetches relevant memory context from `memory API`.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP request fails, the service returns a non-success status,
    /// or the response cannot be decoded.
    pub async fn recall(&self, query: &str, limit: u8, depth: u8) -> Result<MemoryRecallOutput> {
        let response = self
            .http
            .post(format!("{}/api/rag", self.base_url))
            .json(&RecallRequest {
                query,
                limit,
                depth,
            })
            .send()
            .await
            .context("send memory API rag request")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("memory API rag failed with {status}: {body}"));
        }

        response
            .json()
            .await
            .context("parse memory API rag response")
    }

    /// Prefetches session-scoped memory context from `memory API`.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP request fails, the service returns a non-success status,
    /// or the response cannot be decoded.
    pub async fn prefetch(&self, input: MemoryPrefetchInput) -> Result<MemoryRecallOutput> {
        let response = self
            .http
            .post(format!("{}/api/memory/prefetch", self.base_url))
            .json(&PrefetchRequest {
                query: input.query,
                limit: input.limit,
                depth: input.depth,
                session_id: input.session_id,
            })
            .send()
            .await
            .context("send memory API prefetch request")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("memory API prefetch failed with {status}: {body}"));
        }

        response
            .json()
            .await
            .context("parse memory API prefetch response")
    }

    /// Syncs a completed user/assistant turn to `memory API`.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP request fails or the service returns a non-success status.
    pub async fn sync_turn(&self, input: MemoryTurnInput) -> Result<()> {
        let response = self
            .http
            .post(format!("{}/api/memory/turns", self.base_url))
            .json(&TurnRequest {
                session_id: input.session_id,
                user: input.user,
                assistant: input.assistant,
                agent_id: input.agent_id,
            })
            .send()
            .await
            .context("send memory API turn sync request")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("memory API turn sync failed with {status}: {body}"));
        }

        Ok(())
    }

    /// Extracts learnings at the end of a session.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP request fails, the service returns a non-success status,
    /// or the response cannot be decoded.
    pub async fn end_session(&self, input: MemorySessionEndInput) -> Result<Option<String>> {
        let response = self
            .http
            .post(format!("{}/api/memory/session-end", self.base_url))
            .json(&SessionEndRequest {
                session_id: input.session_id,
                agent_id: input.agent_id,
            })
            .send()
            .await
            .context("send memory API session end request")?;

        let status = response.status();
        if status == reqwest::StatusCode::NO_CONTENT {
            return Ok(None);
        }
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "memory API session end failed with {status}: {body}"
            ));
        }

        let body: SessionEndResponse = response
            .json()
            .await
            .context("parse memory API session end response")?;

        Ok(body
            .proposal_id
            .filter(|proposal_id| !proposal_id.trim().is_empty()))
    }

    /// Flushes memory-provider shutdown work.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP request fails or the service returns a non-success status.
    pub async fn shutdown(&self, input: MemoryShutdownInput) -> Result<()> {
        let response = self
            .http
            .post(format!("{}/api/memory/shutdown", self.base_url))
            .json(&ShutdownRequest {
                agent_id: input.agent_id,
            })
            .send()
            .await
            .context("send memory API shutdown request")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("memory API shutdown failed with {status}: {body}"));
        }

        Ok(())
    }

    /// Creates a proposal-first note in `memory API`.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP request fails, the service returns a non-success status,
    /// the response cannot be decoded, or the service does not return a proposal id.
    pub async fn propose_note(&self, input: MemoryNoteInput) -> Result<String> {
        let response = self
            .http
            .post(format!("{}/api/notes", self.base_url))
            .json(&NoteRequest {
                title: input.title,
                body: input.body,
                kind: input.kind,
                agent_id: input.agent_id,
                tags: input.tags,
                mode: WriteMode::Propose,
            })
            .send()
            .await
            .context("send memory API note request")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("memory API note failed with {status}: {body}"));
        }

        let body: NoteResponse = response
            .json()
            .await
            .context("parse memory API note response")?;

        if body.mode != "propose" {
            return Err(anyhow!(
                "memory API returned unexpected note write mode: {}",
                body.mode
            ));
        }

        body.proposal
            .map(|proposal| proposal.id)
            .filter(|id| !id.trim().is_empty())
            .ok_or_else(|| anyhow!("memory API returned no proposal id"))
    }

    /// Lists pending proposal records from `memory API`.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP request fails, the service returns a non-success status,
    /// or the response cannot be decoded.
    pub async fn pending_proposals(&self, limit: u16) -> Result<Vec<PendingProposal>> {
        let response = self
            .http
            .get(format!("{}/api/proposals", self.base_url))
            .query(&PendingProposalsQuery {
                status: ProposalStatus::Pending,
                limit,
            })
            .send()
            .await
            .context("send memory API proposal list request")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "memory API proposal list failed with {status}: {body}"
            ));
        }

        let body: PendingProposalsResponse = response
            .json()
            .await
            .context("parse memory API proposal list response")?;

        Ok(body.proposals)
    }
}
