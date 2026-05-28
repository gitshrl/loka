use anyhow::{Context, Result, anyhow};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::config::ModelProtocol;
use crate::messages::Message;
use crate::messages::Role;

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone)]
pub struct ModelClient {
    http: Client,
    base_url: String,
    api_key: String,
    protocol: ModelProtocol,
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

#[derive(Debug, Serialize)]
struct AnthropicMessageRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessageStreamRequest {
    model: String,
    max_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageResponse {
    content: Vec<AnthropicContentBlock>,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}

impl From<AnthropicUsage> for TokenUsage {
    fn from(value: AnthropicUsage) -> Self {
        Self {
            prompt_tokens: value.input_tokens,
            completion_tokens: value.output_tokens,
            total_tokens: value.input_tokens + value.output_tokens,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicStreamEvent {
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { delta: AnthropicStreamDelta },
    #[serde(rename = "message_delta")]
    MessageDelta { usage: Option<AnthropicUsage> },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicStreamDelta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(other)]
    Other,
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

impl ModelClient {
    #[must_use]
    pub fn new(base_url: impl AsRef<str>, api_key: impl Into<String>) -> Self {
        Self::with_protocol(base_url, api_key, ModelProtocol::OpenAiCompatible)
    }

    #[must_use]
    pub fn with_protocol(
        base_url: impl AsRef<str>,
        api_key: impl Into<String>,
        protocol: ModelProtocol,
    ) -> Self {
        Self {
            http: Client::new(),
            base_url: base_url.as_ref().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            protocol,
        }
    }

    /// Sends one chat request through the configured model API protocol.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP request fails, the model API returns a non-success status,
    /// the response cannot be decoded, or the response contains no assistant text.
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatOutput> {
        match self.protocol {
            ModelProtocol::OpenAiCompatible => self.chat_openai(request).await,
            ModelProtocol::AnthropicCompatible => self.chat_anthropic(request).await,
        }
    }

    async fn chat_openai(&self, request: ChatRequest) -> Result<ChatOutput> {
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
            .context("send chat completion request to model API")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("model API request failed with {status}: {body}"));
        }

        let body: OpenAiChatResponse = response
            .json()
            .await
            .context("parse model API chat completion response")?;

        let content = body
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .map(|content| content.trim().to_string())
            .filter(|content| !content.is_empty())
            .ok_or_else(|| anyhow!("model API returned no assistant content"))?;

        Ok(ChatOutput {
            content,
            usage: body.usage,
        })
    }

    async fn chat_anthropic(&self, request: ChatRequest) -> Result<ChatOutput> {
        let request = anthropic_request(request);
        let response = self
            .http
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&request)
            .send()
            .await
            .context("send chat request to model API")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("model API request failed with {status}: {body}"));
        }

        let body: AnthropicMessageResponse = response
            .json()
            .await
            .context("parse model API chat response")?;

        let content = anthropic_text(body.content)?;
        Ok(ChatOutput {
            content,
            usage: body.usage.map(Into::into),
        })
    }

    /// Sends one streaming chat request through the configured model API protocol.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP stream fails, the model API returns a non-success status,
    /// a stream event cannot be decoded, the callback fails, or no assistant text is produced.
    pub async fn chat_stream<F>(&self, request: ChatRequest, on_delta: F) -> Result<ChatOutput>
    where
        F: FnMut(&str) -> Result<()>,
    {
        match self.protocol {
            ModelProtocol::OpenAiCompatible => self.chat_stream_openai(request, on_delta).await,
            ModelProtocol::AnthropicCompatible => {
                self.chat_stream_anthropic(request, on_delta).await
            }
        }
    }

    async fn chat_stream_openai<F>(
        &self,
        request: ChatRequest,
        mut on_delta: F,
    ) -> Result<ChatOutput>
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
            .context("send streaming chat completion request to model API")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "model API streaming request failed with {status}: {body}"
            ));
        }

        let mut state = StreamAccumulator::default();
        let mut bytes = response.bytes_stream();
        while let Some(chunk) = bytes.next().await {
            state.push_bytes(
                &chunk.context("read model API stream chunk")?,
                &mut on_delta,
            )?;
            if state.done {
                break;
            }
        }
        state.finish(&mut on_delta)
    }

    async fn chat_stream_anthropic<F>(
        &self,
        request: ChatRequest,
        mut on_delta: F,
    ) -> Result<ChatOutput>
    where
        F: FnMut(&str) -> Result<()>,
    {
        let request = anthropic_stream_request(request);
        let response = self
            .http
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&request)
            .send()
            .await
            .context("send streaming chat request to model API")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "model API streaming request failed with {status}: {body}"
            ));
        }

        let mut state = AnthropicStreamAccumulator::default();
        let mut bytes = response.bytes_stream();
        while let Some(chunk) = bytes.next().await {
            state.push_bytes(
                &chunk.context("read model API stream chunk")?,
                &mut on_delta,
            )?;
            if state.done {
                break;
            }
        }
        state.finish(&mut on_delta)
    }
}

fn anthropic_request(request: ChatRequest) -> AnthropicMessageRequest {
    let (system, messages) = anthropic_messages(request.messages);
    AnthropicMessageRequest {
        model: request.model,
        max_tokens: DEFAULT_MAX_TOKENS,
        system,
        messages,
    }
}

fn anthropic_stream_request(request: ChatRequest) -> AnthropicMessageStreamRequest {
    let (system, messages) = anthropic_messages(request.messages);
    AnthropicMessageStreamRequest {
        model: request.model,
        max_tokens: DEFAULT_MAX_TOKENS,
        stream: true,
        system,
        messages,
    }
}

fn anthropic_messages(messages: Vec<Message>) -> (Option<String>, Vec<AnthropicMessage>) {
    let mut system = Vec::new();
    let mut anthropic_messages = Vec::with_capacity(messages.len());

    for message in messages {
        match message.role {
            Role::System => system.push(message.content),
            Role::User => anthropic_messages.push(AnthropicMessage {
                role: "user",
                content: message.content,
            }),
            Role::Assistant => anthropic_messages.push(AnthropicMessage {
                role: "assistant",
                content: message.content,
            }),
            Role::Tool => anthropic_messages.push(AnthropicMessage {
                role: "user",
                content: format!("Tool result:\n{}", message.content),
            }),
        }
    }

    let system = system
        .into_iter()
        .map(|content| content.trim().to_string())
        .filter(|content| !content.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");

    (
        if system.is_empty() {
            None
        } else {
            Some(system)
        },
        anthropic_messages,
    )
}

fn anthropic_text(blocks: Vec<AnthropicContentBlock>) -> Result<String> {
    let mut content = String::new();
    for block in blocks {
        if let AnthropicContentBlock::Text { text } = block {
            content.push_str(&text);
        }
    }

    if content.trim().is_empty() {
        return Err(anyhow!("model API returned no assistant content"));
    }

    Ok(content.trim().to_string())
}

#[derive(Debug, Default)]
struct StreamAccumulator {
    pending: Vec<u8>,
    content: String,
    usage: Option<TokenUsage>,
    done: bool,
}

#[derive(Debug, Default)]
struct AnthropicStreamAccumulator {
    pending: Vec<u8>,
    content: String,
    usage: Option<TokenUsage>,
    done: bool,
}

impl AnthropicStreamAccumulator {
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
            return Err(anyhow!("model API returned no streamed assistant content"));
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
            .context("decode model API stream line")?
            .trim();
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            return Ok(StreamLine::Continue);
        };
        if data.is_empty() || data == "[DONE]" {
            return Ok(StreamLine::Continue);
        }

        let event: AnthropicStreamEvent =
            serde_json::from_str(data).context("parse model API stream event")?;
        match event {
            AnthropicStreamEvent::ContentBlockDelta {
                delta: AnthropicStreamDelta::Text { text },
            } if !text.is_empty() => {
                on_delta(&text)?;
                self.content.push_str(&text);
            }
            AnthropicStreamEvent::MessageDelta { usage } => {
                self.usage = usage.map(Into::into);
            }
            AnthropicStreamEvent::MessageStop => return Ok(StreamLine::Done),
            AnthropicStreamEvent::ContentBlockDelta { .. } | AnthropicStreamEvent::Other => {}
        }

        Ok(StreamLine::Continue)
    }
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
            return Err(anyhow!("model API returned no streamed assistant content"));
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
            .context("decode model API stream line")?
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
            serde_json::from_str(data).context("parse model API stream event")?;
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
