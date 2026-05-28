use anyhow::{Context, Result, anyhow};
use futures::StreamExt;
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

#[derive(Debug, Serialize)]
struct OpenAiChatStreamRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
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

#[derive(Debug, Deserialize)]
struct OpenAiChatStreamChunk {
    choices: Vec<OpenAiStreamChoice>,
    usage: Option<TokenUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiStreamDelta,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamDelta {
    content: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamLine {
    Continue,
    Done,
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

    /// Sends one OpenAI-compatible streaming chat completion request through `pengepul`.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP stream fails, `pengepul` returns a non-success status,
    /// a stream event cannot be decoded, the callback fails, or no assistant text is produced.
    pub async fn chat_stream<F>(&self, request: ChatRequest, mut on_delta: F) -> Result<ChatOutput>
    where
        F: FnMut(&str) -> Result<()>,
    {
        let response = self
            .http
            .post(format!("{}/v1/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&OpenAiChatStreamRequest {
                model: request.model,
                messages: request.messages,
                stream: true,
            })
            .send()
            .await
            .context("send streaming chat completion request to pengepul")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "pengepul streaming request failed with {status}: {body}"
            ));
        }

        let mut state = StreamAccumulator::default();
        let mut bytes = response.bytes_stream();
        while let Some(chunk) = bytes.next().await {
            state.push_bytes(&chunk.context("read pengepul stream chunk")?, &mut on_delta)?;
            if state.done {
                break;
            }
        }
        state.finish(&mut on_delta)
    }
}

#[derive(Debug, Default)]
struct StreamAccumulator {
    pending: Vec<u8>,
    content: String,
    usage: Option<TokenUsage>,
    done: bool,
}

impl StreamAccumulator {
    fn push_bytes<F>(&mut self, bytes: &[u8], on_delta: &mut F) -> Result<()>
    where
        F: FnMut(&str) -> Result<()>,
    {
        self.pending.extend_from_slice(bytes);
        while let Some(index) = self.pending.iter().position(|byte| *byte == b'\n') {
            let line = self.pending.drain(..=index).collect::<Vec<_>>();
            if self.process_line(&line, on_delta)? == StreamLine::Done {
                self.done = true;
                break;
            }
        }
        Ok(())
    }

    fn finish<F>(mut self, on_delta: &mut F) -> Result<ChatOutput>
    where
        F: FnMut(&str) -> Result<()>,
    {
        if !self.pending.is_empty() && !self.done {
            let line = std::mem::take(&mut self.pending);
            self.process_line(&line, on_delta)?;
        }
        if self.content.trim().is_empty() {
            return Err(anyhow!("pengepul returned no streamed assistant content"));
        }
        Ok(ChatOutput {
            content: self.content,
            usage: self.usage,
        })
    }

    fn process_line<F>(&mut self, line: &[u8], on_delta: &mut F) -> Result<StreamLine>
    where
        F: FnMut(&str) -> Result<()>,
    {
        let line = std::str::from_utf8(line)
            .context("decode pengepul stream line")?
            .trim();
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            return Ok(StreamLine::Continue);
        };
        if data.is_empty() {
            return Ok(StreamLine::Continue);
        }
        if data == "[DONE]" {
            return Ok(StreamLine::Done);
        }

        let chunk: OpenAiChatStreamChunk =
            serde_json::from_str(data).context("parse pengepul stream event")?;
        if let Some(usage) = chunk.usage {
            self.usage = Some(usage);
        }
        for choice in chunk.choices {
            if let Some(delta) = choice.delta.content
                && !delta.is_empty()
            {
                on_delta(&delta)?;
                self.content.push_str(&delta);
            }
        }
        Ok(StreamLine::Continue)
    }
}
