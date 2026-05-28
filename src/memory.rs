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
