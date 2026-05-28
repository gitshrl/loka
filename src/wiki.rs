use anyhow::{Context, Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct WikiClient {
    http: Client,
    base_url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RagOutput {
    pub mode: String,
    pub markdown: String,
}

#[derive(Debug, Serialize)]
struct RagRequest<'a> {
    query: &'a str,
    limit: u8,
    depth: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoteInput {
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
    proposal: Option<Proposal>,
}

#[derive(Debug, Deserialize)]
struct Proposal {
    id: String,
}

impl WikiClient {
    #[must_use]
    pub fn new(base_url: impl AsRef<str>) -> Self {
        Self {
            http: Client::new(),
            base_url: base_url.as_ref().trim_end_matches('/').to_string(),
        }
    }

    /// Fetches relevant memory context from `personal-wiki`.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP request fails, the service returns a non-success status,
    /// or the response cannot be decoded.
    pub async fn rag(&self, query: &str, limit: u8, depth: u8) -> Result<RagOutput> {
        let response = self
            .http
            .post(format!("{}/api/rag", self.base_url))
            .json(&RagRequest {
                query,
                limit,
                depth,
            })
            .send()
            .await
            .context("send personal-wiki rag request")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("personal-wiki rag failed with {status}: {body}"));
        }

        response
            .json()
            .await
            .context("parse personal-wiki rag response")
    }

    /// Creates a proposal-first note in `personal-wiki`.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP request fails, the service returns a non-success status,
    /// the response cannot be decoded, or the service does not return a proposal id.
    pub async fn add_note(&self, input: NoteInput) -> Result<String> {
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
            .context("send personal-wiki note request")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("personal-wiki note failed with {status}: {body}"));
        }

        let body: NoteResponse = response
            .json()
            .await
            .context("parse personal-wiki note response")?;

        if body.mode != "propose" {
            return Err(anyhow!(
                "personal-wiki returned unexpected note write mode: {}",
                body.mode
            ));
        }

        body.proposal
            .map(|proposal| proposal.id)
            .filter(|id| !id.trim().is_empty())
            .ok_or_else(|| anyhow!("personal-wiki returned no proposal id"))
    }
}
