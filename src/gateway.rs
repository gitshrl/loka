use anyhow::{Context, Result, anyhow};
use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use futures::future::BoxFuture;
use reqwest::Client;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
use tokio::net::TcpListener;

use crate::config::{AppConfig, MemoryLifecycleMode};
use crate::memory::{MemoryClient, MemoryPrefetchInput, MemoryTurnInput};
use crate::messages::{Message, Role, Transcript};
use crate::model::{ChatRequest, ModelClient};
use crate::prompt::{PromptBuilder, PromptInput, discover_context_files};
use crate::session::SessionStore;
use crate::skills::SkillStore;
use crate::tokens::{TokenScope, TokenUsage, estimate_messages_tokens};

const RECALL_LIMIT: u8 = 6;
const RECALL_DEPTH: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayRequest {
    pub gateway: String,
    pub conversation_key: String,
    pub session_key: String,
    pub text: String,
    pub recall: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayResponse {
    pub text: String,
}

pub trait GatewayAgent: Send + Sync {
    fn respond(&self, request: GatewayRequest) -> BoxFuture<'_, Result<GatewayResponse>>;
}

#[derive(Debug, Clone)]
pub struct TelegramClient {
    http: Client,
    base_url: String,
    token: String,
}

#[derive(Debug, Clone)]
pub struct TelegramGateway<A> {
    client: TelegramClient,
    agent: A,
    recall: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelegramGatewayOutcome {
    Replied { chat_id: i64 },
    Ignored,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct TelegramUpdate {
    pub update_id: i64,
    pub message: Option<TelegramMessage>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct TelegramMessage {
    pub message_id: i64,
    pub chat: TelegramChat,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct TelegramChat {
    pub id: i64,
}

#[derive(Debug, Serialize)]
struct SendMessageRequest<'a> {
    chat_id: i64,
    text: &'a str,
}

#[derive(Debug, Deserialize)]
struct TelegramApiResponse {
    ok: bool,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug)]
pub struct GatewaySessionStore {
    conn: Connection,
}

#[derive(Debug, Clone)]
pub struct LokaGatewayAgent {
    config: AppConfig,
}

#[derive(Debug)]
struct GatewayError(anyhow::Error);

impl TelegramClient {
    #[must_use]
    pub fn new(token: impl Into<String>) -> Self {
        Self::with_base_url("https://api.telegram.org", token)
    }

    #[must_use]
    pub fn with_base_url(base_url: impl AsRef<str>, token: impl Into<String>) -> Self {
        Self {
            http: Client::new(),
            base_url: base_url.as_ref().trim_end_matches('/').to_string(),
            token: token.into(),
        }
    }

    /// Sends a Telegram text message through `sendMessage`.
    ///
    /// # Errors
    ///
    /// Returns an error when Telegram rejects the request or the response cannot be decoded.
    pub async fn send_message(&self, chat_id: i64, text: &str) -> Result<()> {
        if self.token.trim().is_empty() {
            return Err(anyhow!("telegram bot token is required"));
        }
        let response = self
            .http
            .post(format!("{}/bot{}/sendMessage", self.base_url, self.token))
            .json(&SendMessageRequest { chat_id, text })
            .send()
            .await
            .context("send Telegram message")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("Telegram sendMessage failed with {status}: {body}"));
        }

        let body: TelegramApiResponse = response
            .json()
            .await
            .context("parse Telegram sendMessage response")?;
        if !body.ok {
            return Err(anyhow!(
                "Telegram sendMessage failed: {}",
                body.description
                    .unwrap_or_else(|| "unknown error".to_string())
            ));
        }

        Ok(())
    }
}

impl<A> TelegramGateway<A>
where
    A: GatewayAgent,
{
    #[must_use]
    pub const fn new(client: TelegramClient, agent: A, recall: bool) -> Self {
        Self {
            client,
            agent,
            recall,
        }
    }

    /// Handles one Telegram webhook update.
    ///
    /// # Errors
    ///
    /// Returns an error when the agent request or Telegram reply fails.
    pub async fn handle_update(&self, update: TelegramUpdate) -> Result<TelegramGatewayOutcome> {
        let Some(message) = update.message else {
            return Ok(TelegramGatewayOutcome::Ignored);
        };
        let Some(text) = message
            .text
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
        else {
            return Ok(TelegramGatewayOutcome::Ignored);
        };

        let conversation_key = message.chat.id.to_string();
        let response = self
            .agent
            .respond(GatewayRequest {
                gateway: "telegram".to_string(),
                session_key: format!("telegram:{conversation_key}"),
                conversation_key,
                text: text.to_string(),
                recall: self.recall,
            })
            .await?;
        self.client
            .send_message(message.chat.id, &response.text)
            .await?;
        Ok(TelegramGatewayOutcome::Replied {
            chat_id: message.chat.id,
        })
    }
}

impl GatewaySessionStore {
    /// Opens the gateway session mapping database inside the configured state directory.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory cannot be created, database cannot open, or migrations fail.
    pub fn open(state_dir: impl AsRef<Path>) -> Result<Self> {
        fs::create_dir_all(state_dir.as_ref()).with_context(|| {
            format!(
                "create gateway state directory {}",
                state_dir.as_ref().display()
            )
        })?;
        Self::open_database(state_dir.as_ref().join("gateways.sqlite3"))
    }

    /// Opens the gateway session mapping database at an explicit path.
    ///
    /// # Errors
    ///
    /// Returns an error when the database cannot open or migrations fail.
    pub fn open_database(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref())
            .with_context(|| format!("open gateway database {}", path.as_ref().display()))?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    /// Opens an in-memory gateway session store for tests.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` cannot open or migrations fail.
    pub fn in_memory() -> Result<Self> {
        let store = Self {
            conn: Connection::open_in_memory().context("open in-memory gateway database")?,
        };
        store.migrate()?;
        Ok(store)
    }

    /// Looks up the Loka session id for a gateway conversation.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` rejects the lookup.
    pub fn session_id(&self, gateway: &str, conversation_key: &str) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT session_id FROM gateway_sessions WHERE gateway = ?1 AND conversation_key = ?2",
                params![gateway, conversation_key],
                |row| row.get(0),
            )
            .optional()
            .context("lookup gateway session")
    }

    /// Stores or updates the session mapping for a gateway conversation.
    ///
    /// # Errors
    ///
    /// Returns an error when timestamp formatting fails or `SQLite` rejects the write.
    pub fn upsert(&self, gateway: &str, conversation_key: &str, session_id: &str) -> Result<()> {
        let now = now_rfc3339()?;
        self.conn.execute(
            "INSERT INTO gateway_sessions (
                gateway, conversation_key, session_id, created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(gateway, conversation_key) DO UPDATE SET
                session_id = excluded.session_id,
                updated_at = excluded.updated_at",
            params![gateway, conversation_key, session_id, now],
        )?;
        Ok(())
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "
                PRAGMA foreign_keys = ON;
                PRAGMA busy_timeout = 5000;
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = NORMAL;

                CREATE TABLE IF NOT EXISTS gateway_sessions (
                    gateway TEXT NOT NULL,
                    conversation_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (gateway, conversation_key)
                );
                ",
            )
            .context("migrate gateway database")
    }
}

impl LokaGatewayAgent {
    #[must_use]
    pub const fn new(config: AppConfig) -> Self {
        Self { config }
    }

    fn session_id_for(&self, request: &GatewayRequest) -> Result<String> {
        let sessions = SessionStore::open(&self.config.state_dir)?;
        let gateway_sessions = GatewaySessionStore::open(&self.config.state_dir)?;
        if let Some(session_id) =
            gateway_sessions.session_id(&request.gateway, &request.conversation_key)?
        {
            return Ok(session_id);
        }

        let session_id = sessions.create_session(&request.session_key)?;
        gateway_sessions.upsert(&request.gateway, &request.conversation_key, &session_id)?;
        Ok(session_id)
    }

    fn memory_lifecycle_strict(&self) -> bool {
        self.config.memory_lifecycle == MemoryLifecycleMode::Strict
    }
}

impl GatewayAgent for LokaGatewayAgent {
    fn respond(&self, request: GatewayRequest) -> BoxFuture<'_, Result<GatewayResponse>> {
        Box::pin(async move {
            let session_id = self.session_id_for(&request)?;
            let turns = {
                let sessions = SessionStore::open(&self.config.state_dir)?;
                sessions.session_turns(&session_id)?
            };
            let skills = {
                let skills = SkillStore::open(&self.config.state_dir)?;
                skills.enabled_for_prompt(&request.text)?
            };
            let memory_client = MemoryClient::new(&self.config.memory_base_url);
            let memory_markdown = if request.recall {
                let memory = if self.memory_lifecycle_strict() {
                    memory_client
                        .prefetch(MemoryPrefetchInput {
                            query: request.text.clone(),
                            limit: RECALL_LIMIT,
                            depth: RECALL_DEPTH,
                            session_id: Some(session_id.clone()),
                        })
                        .await?
                } else {
                    memory_client
                        .recall(&request.text, RECALL_LIMIT, RECALL_DEPTH)
                        .await?
                };
                Some(memory.markdown)
            } else {
                None
            };
            let system_prompt = PromptBuilder::new()
                .build(&PromptInput {
                    agent_id: self.config.agent_id.clone(),
                    model: self.config.model.clone(),
                    model_protocol: self.config.model_protocol,
                    session_id: Some(session_id.clone()),
                    system_message: Some(gateway_system_message(&skills)),
                    memory_markdown,
                    context_files: discover_context_files(&self.config.working_dir)?,
                    date: OffsetDateTime::now_utc().date().to_string(),
                })
                .assemble();

            let mut transcript = Transcript::new();
            transcript.push(Message::system(system_prompt));
            for turn in turns {
                transcript.push(Message {
                    role: turn.role,
                    content: turn.content,
                });
            }
            transcript.push(Message::user(request.text.clone()));
            let messages = transcript.into_messages();
            {
                let sessions = SessionStore::open(&self.config.state_dir)?;
                sessions.append_turn(&session_id, Role::User, &request.text)?;
                sessions.record_token_usage(
                    &session_id,
                    TokenScope::Prompt,
                    "gateway",
                    TokenUsage::estimated_prompt(estimate_messages_tokens(&messages)),
                )?;
            }

            let output = ModelClient::with_protocol(
                &self.config.model_base_url,
                self.config.model_api_key.clone(),
                self.config.model_protocol,
            )
            .chat(ChatRequest {
                model: self.config.model.clone(),
                messages,
            })
            .await?;

            let answer = output.content;
            {
                let sessions = SessionStore::open(&self.config.state_dir)?;
                sessions.append_turn(&session_id, Role::Assistant, &answer)?;
            }
            if self.memory_lifecycle_strict() {
                memory_client
                    .sync_turn(MemoryTurnInput {
                        session_id: Some(session_id),
                        user: request.text,
                        assistant: answer.clone(),
                        agent_id: self.config.agent_id.clone(),
                    })
                    .await?;
            }

            Ok(GatewayResponse { text: answer })
        })
    }
}

fn gateway_system_message(skills: &[crate::skills::Skill]) -> String {
    let mut parts = vec!["Message received through the Telegram gateway.".to_string()];
    if !skills.is_empty() {
        let body = skills
            .iter()
            .map(crate::skills::Skill::prompt_block)
            .collect::<Vec<_>>()
            .join("\n\n");
        parts.push(format!("# Enabled Skills\n{body}"));
    }
    parts.join("\n\n")
}

/// Runs a Telegram webhook server.
///
/// # Errors
///
/// Returns an error when binding the socket fails or the HTTP server exits with an error.
pub async fn run_telegram_gateway(
    config: AppConfig,
    token: String,
    addr: SocketAddr,
    path: String,
    recall: bool,
) -> Result<()> {
    let gateway = Arc::new(TelegramGateway::new(
        TelegramClient::new(token),
        LokaGatewayAgent::new(config),
        recall,
    ));
    let app = Router::new()
        .route(&path, post(telegram_webhook::<LokaGatewayAgent>))
        .with_state(gateway);
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind gateway listener {addr}"))?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("run Telegram gateway")
}

async fn telegram_webhook<A>(
    State(gateway): State<Arc<TelegramGateway<A>>>,
    Json(update): Json<TelegramUpdate>,
) -> Result<Json<serde_json::Value>, GatewayError>
where
    A: GatewayAgent,
{
    gateway.handle_update(update).await.map_err(GatewayError)?;
    Ok(Json(json!({ "ok": true })))
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::warn!(%error, "failed to listen for ctrl-c");
    }
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        tracing::error!(error = %self.0, "gateway request failed");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "ok": false, "error": self.0.to_string() })),
        )
            .into_response()
    }
}

fn now_rfc3339() -> Result<String> {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .context("format timestamp")
}
