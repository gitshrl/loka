use anyhow::{Context, Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::messages::Message;

#[derive(Debug, Clone)]
pub struct LlmClient {
    http: Client,
    base_url: String,
    api_key: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatOutput {
    pub content: String,
    pub usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct TokenUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

#[derive(Debug, Serialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<Message>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<TokenUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    content: Option<String>,
}

impl LlmClient {
    #[must_use]
    pub fn new(base_url: impl AsRef<str>, api_key: impl Into<String>) -> Self {
        Self {
            http: Client::new(),
            base_url: base_url.as_ref().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
        }
    }

    /// Sends one OpenAI-compatible chat completion request through `pengepul`.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP request fails, `pengepul` returns a non-success status,
    /// the response cannot be decoded, or the response contains no assistant text.
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatOutput> {
        let response = self
            .http
            .post(format!("{}/v1/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&OpenAiChatRequest {
                model: request.model,
                messages: request.messages,
            })
            .send()
            .await
            .context("send chat completion request to pengepul")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("pengepul request failed with {status}: {body}"));
        }

        let body: OpenAiChatResponse = response
            .json()
            .await
            .context("parse pengepul chat completion response")?;

        let content = body
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .map(|content| content.trim().to_string())
            .filter(|content| !content.is_empty())
            .ok_or_else(|| anyhow!("pengepul returned no assistant content"))?;

        Ok(ChatOutput {
            content,
            usage: body.usage,
        })
    }
}
