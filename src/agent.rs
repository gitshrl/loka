use anyhow::{Result, anyhow, bail};

use crate::config::{AppConfig, MemoryLifecycleMode};
use crate::memory::{
    MemoryClient, MemoryNoteInput, MemoryPrefetchInput, MemorySessionEndInput, MemoryShutdownInput,
    MemoryTurnInput,
};
use crate::messages::{Message, Role, Transcript};
use crate::model::{ChatRequest, ModelClient};
use crate::prompt::{PromptBuilder, PromptInput, discover_context_files};
use crate::session::SessionStore;
use crate::session_summary::{SessionSummaryOutput, SessionSummaryRequest, summarize_session};
use crate::skills::SkillStore;
use crate::tokens::{TokenScope, TokenUsage, estimate_messages_tokens};
use time::OffsetDateTime;

const RECALL_LIMIT: u8 = 6;
const RECALL_DEPTH: u8 = 1;
pub const DEFAULT_SUMMARY_MIN_TURNS: usize = 12;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskRequest {
    pub prompt: String,
    pub recall: bool,
    pub session_id: Option<String>,
    pub system_message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskOutput {
    pub answer: String,
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatSessionRequest {
    pub messages: Vec<String>,
    pub recall: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatSessionOutput {
    pub session_id: String,
    pub answers: Vec<String>,
    pub summary_proposal_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatSession {
    session_id: String,
    transcript: Transcript,
    recall: bool,
}

impl ChatSession {
    #[must_use]
    pub fn id(&self) -> &str {
        &self.session_id
    }
}

#[derive(Debug)]
pub struct Agent {
    config: AppConfig,
    model_client: ModelClient,
    memory: MemoryClient,
    prompt_builder: PromptBuilder,
    sessions: Option<SessionStore>,
    skills: Option<SkillStore>,
}

impl Agent {
    #[must_use]
    pub fn new(config: AppConfig) -> Self {
        Self {
            model_client: ModelClient::with_protocol(
                config.model_base_url.clone(),
                config.model_api_key.clone(),
                config.model_protocol,
            ),
            memory: MemoryClient::new(config.memory_base_url.clone()),
            prompt_builder: PromptBuilder::new(),
            config,
            sessions: None,
            skills: None,
        }
    }

    #[must_use]
    pub fn with_session_store(config: AppConfig, sessions: SessionStore) -> Self {
        Self {
            model_client: ModelClient::with_protocol(
                config.model_base_url.clone(),
                config.model_api_key.clone(),
                config.model_protocol,
            ),
            memory: MemoryClient::new(config.memory_base_url.clone()),
            prompt_builder: PromptBuilder::new(),
            config,
            sessions: Some(sessions),
            skills: None,
        }
    }

    #[must_use]
    pub fn with_stores(config: AppConfig, sessions: SessionStore, skills: SkillStore) -> Self {
        Self {
            model_client: ModelClient::with_protocol(
                config.model_base_url.clone(),
                config.model_api_key.clone(),
                config.model_protocol,
            ),
            memory: MemoryClient::new(config.memory_base_url.clone()),
            prompt_builder: PromptBuilder::new(),
            config,
            sessions: Some(sessions),
            skills: Some(skills),
        }
    }

    /// Answers a prompt, optionally injecting memory context first.
    ///
    /// # Errors
    ///
    /// Returns an error when recall fails, session persistence fails, or the model request fails.
    pub async fn ask(&self, request: AskRequest) -> Result<AskOutput> {
        let session_id = self.create_session(&request.prompt)?;
        let prompt_session_id = request.session_id.clone().or_else(|| session_id.clone());
        let sync_session_id = session_id.clone().or_else(|| prompt_session_id.clone());
        let system_prompt = self
            .build_system_prompt(
                &request.prompt,
                request.recall,
                prompt_session_id,
                request.system_message,
            )
            .await?;
        let mut transcript = Transcript::new();
        transcript.push(Message::system(system_prompt));
        transcript.push(Message::user(request.prompt.clone()));
        self.append_turn(session_id.as_deref(), Role::User, &request.prompt)?;
        let messages = transcript.into_messages();
        self.record_token_usage(
            session_id.as_deref(),
            TokenScope::Prompt,
            "ask",
            TokenUsage::estimated_prompt(estimate_messages_tokens(&messages)),
        )?;

        let response = self
            .model_client
            .chat(ChatRequest {
                model: self.config.model.clone(),
                messages,
            })
            .await?;

        let answer = response.content;
        self.append_turn(session_id.as_deref(), Role::Assistant, &answer)?;
        self.sync_memory_turn(sync_session_id.as_deref(), &request.prompt, &answer)
            .await?;

        Ok(AskOutput { answer, session_id })
    }

    /// Answers a prompt with streaming deltas while persisting the final assistant text.
    ///
    /// # Errors
    ///
    /// Returns an error when recall fails, session persistence fails, the model stream fails,
    /// or the delta callback fails.
    pub async fn ask_stream<F>(&self, request: AskRequest, on_delta: F) -> Result<AskOutput>
    where
        F: FnMut(&str) -> Result<()>,
    {
        let session_id = self.create_session(&request.prompt)?;
        let prompt_session_id = request.session_id.clone().or_else(|| session_id.clone());
        let sync_session_id = session_id.clone().or_else(|| prompt_session_id.clone());
        let system_prompt = self
            .build_system_prompt(
                &request.prompt,
                request.recall,
                prompt_session_id,
                request.system_message,
            )
            .await?;
        let mut transcript = Transcript::new();
        transcript.push(Message::system(system_prompt));
        transcript.push(Message::user(request.prompt.clone()));
        self.append_turn(session_id.as_deref(), Role::User, &request.prompt)?;
        let messages = transcript.into_messages();
        self.record_token_usage(
            session_id.as_deref(),
            TokenScope::Prompt,
            "ask_stream",
            TokenUsage::estimated_prompt(estimate_messages_tokens(&messages)),
        )?;

        let response = self
            .model_client
            .chat_stream(
                ChatRequest {
                    model: self.config.model.clone(),
                    messages,
                },
                on_delta,
            )
            .await?;

        let answer = response.content;
        self.append_turn(session_id.as_deref(), Role::Assistant, &answer)?;
        self.sync_memory_turn(sync_session_id.as_deref(), &request.prompt, &answer)
            .await?;

        Ok(AskOutput { answer, session_id })
    }

    /// Answers a prompt as the next turn in an existing persisted session.
    ///
    /// # Errors
    ///
    /// Returns an error when no session store is configured, the session does not exist,
    /// recall fails, session persistence fails, or the model request fails.
    pub async fn ask_in_session(
        &self,
        session_id: &str,
        prompt: String,
        recall: bool,
        system_message: Option<String>,
    ) -> Result<AskOutput> {
        let prompt = prompt.trim().to_string();
        if prompt.is_empty() {
            bail!("session turn requires a message");
        }
        let turns = {
            let sessions = self
                .sessions
                .as_ref()
                .ok_or_else(|| anyhow!("ask_in_session requires a session store"))?;
            if !sessions.session_exists(session_id)? {
                bail!("session {session_id} not found");
            }
            sessions.session_turns(session_id)?
        };

        let system_prompt = self
            .build_system_prompt(
                &prompt,
                recall,
                Some(session_id.to_string()),
                system_message,
            )
            .await?;
        let mut transcript = Transcript::new();
        transcript.push(Message::system(system_prompt));
        for turn in turns {
            transcript.push(Message {
                role: turn.role,
                content: turn.content,
            });
        }
        transcript.push(Message::user(prompt.clone()));
        self.append_turn(Some(session_id), Role::User, &prompt)?;
        let messages = transcript.into_messages();
        self.record_token_usage(
            Some(session_id),
            TokenScope::Prompt,
            "ask_in_session",
            TokenUsage::estimated_prompt(estimate_messages_tokens(&messages)),
        )?;

        let response = self
            .model_client
            .chat(ChatRequest {
                model: self.config.model.clone(),
                messages,
            })
            .await?;

        let answer = response.content;
        self.append_turn(Some(session_id), Role::Assistant, &answer)?;
        self.sync_memory_turn(Some(session_id), &prompt, &answer)
            .await?;

        Ok(AskOutput {
            answer,
            session_id: Some(session_id.to_string()),
        })
    }

    /// Runs a multi-turn chat in one persisted session.
    ///
    /// # Errors
    ///
    /// Returns an error when no session store is configured, the request has no non-empty
    /// messages, recall fails, persistence fails, or a model request fails.
    pub async fn chat(&self, request: ChatSessionRequest) -> Result<ChatSessionOutput> {
        let prompts = normalize_chat_messages(request.messages)?;
        let mut session = self.start_chat(&prompts[0], request.recall)?;
        let mut answers = Vec::with_capacity(prompts.len());

        for prompt in prompts {
            answers.push(self.send_chat_turn(&mut session, prompt).await?);
        }
        let summary_proposal_id = self
            .summarize_session_if_long(&session.session_id, DEFAULT_SUMMARY_MIN_TURNS)
            .await?;
        self.end_memory_session(&session.session_id).await?;

        Ok(ChatSessionOutput {
            session_id: session.session_id,
            answers,
            summary_proposal_id,
        })
    }

    /// Starts a persisted chat session.
    ///
    /// # Errors
    ///
    /// Returns an error when no session store is configured or session creation fails.
    pub fn start_chat(&self, title: &str, recall: bool) -> Result<ChatSession> {
        let sessions = self
            .sessions
            .as_ref()
            .ok_or_else(|| anyhow!("chat requires a session store"))?;
        let session_id = sessions.create_session(title)?;
        Ok(ChatSession {
            session_id,
            transcript: Transcript::new(),
            recall,
        })
    }

    /// Sends one turn in an existing chat session.
    ///
    /// # Errors
    ///
    /// Returns an error when the prompt is empty, recall fails, persistence fails, or the model
    /// request fails.
    pub async fn send_chat_turn(
        &self,
        session: &mut ChatSession,
        prompt: String,
    ) -> Result<String> {
        let prompt = prompt.trim().to_string();
        if prompt.is_empty() {
            bail!("chat turn requires a message");
        }

        let sessions = self
            .sessions
            .as_ref()
            .ok_or_else(|| anyhow!("chat requires a session store"))?;
        let system_prompt = self
            .build_system_prompt(
                &prompt,
                session.recall,
                Some(session.session_id.clone()),
                None,
            )
            .await?;
        let mut call_transcript = Transcript::new();
        call_transcript.push(Message::system(system_prompt));
        for message in session.transcript.messages() {
            call_transcript.push(message.clone());
        }

        let user = Message::user(prompt.clone());
        call_transcript.push(user.clone());
        sessions.append_turn(&session.session_id, Role::User, &prompt)?;
        let messages = call_transcript.into_messages();
        self.record_token_usage(
            Some(&session.session_id),
            TokenScope::Prompt,
            "chat_turn",
            TokenUsage::estimated_prompt(estimate_messages_tokens(&messages)),
        )?;

        let response = self
            .model_client
            .chat(ChatRequest {
                model: self.config.model.clone(),
                messages,
            })
            .await?;

        let answer = response.content;
        sessions.append_turn(&session.session_id, Role::Assistant, &answer)?;
        session.transcript.push(user);
        session.transcript.push(Message::assistant(answer.clone()));
        self.sync_memory_turn(Some(&session.session_id), &prompt, &answer)
            .await?;
        Ok(answer)
    }

    /// Creates a proposal-first memory note.
    ///
    /// # Errors
    ///
    /// Returns an error when `memory API` rejects the note or does not return a proposal id.
    pub async fn remember(&self, title: String, body: String, tags: Vec<String>) -> Result<String> {
        self.memory
            .propose_note(MemoryNoteInput {
                title,
                body,
                kind: "note".to_string(),
                agent_id: self.config.agent_id.clone(),
                tags,
            })
            .await
    }

    /// Summarizes a persisted session when it has enough turns.
    ///
    /// # Errors
    ///
    /// Returns an error when no session store is configured, the session is missing, or the
    /// summary proposal cannot be created.
    pub async fn summarize_session_if_long(
        &self,
        session_id: &str,
        min_turns: usize,
    ) -> Result<Option<String>> {
        let sessions = self
            .sessions
            .as_ref()
            .ok_or_else(|| anyhow!("summarize_session_if_long requires a session store"))?;
        match summarize_session(
            &self.config,
            &self.model_client,
            &self.memory,
            sessions,
            SessionSummaryRequest {
                session_id: session_id.to_string(),
                min_turns,
            },
        )
        .await?
        {
            SessionSummaryOutput::ProposalCreated { proposal_id } => Ok(Some(proposal_id)),
            SessionSummaryOutput::TooShort { .. } => Ok(None),
        }
    }

    /// Flushes strict memory lifecycle shutdown work.
    ///
    /// # Errors
    ///
    /// Returns an error when strict memory lifecycle is enabled and the memory API rejects
    /// shutdown.
    pub async fn shutdown_memory(&self) -> Result<()> {
        if !self.memory_lifecycle_strict() {
            return Ok(());
        }

        self.memory
            .shutdown(MemoryShutdownInput {
                agent_id: self.config.agent_id.clone(),
            })
            .await
    }

    /// Runs strict memory lifecycle session-end extraction for an existing session.
    ///
    /// # Errors
    ///
    /// Returns an error when strict memory lifecycle is enabled and the memory API rejects
    /// session-end extraction.
    pub async fn finish_memory_session(&self, session_id: &str) -> Result<()> {
        self.end_memory_session(session_id).await
    }

    fn create_session(&self, prompt: &str) -> Result<Option<String>> {
        self.sessions
            .as_ref()
            .map(|store| store.create_session(prompt))
            .transpose()
    }

    fn append_turn(&self, session_id: Option<&str>, role: Role, content: &str) -> Result<()> {
        if let (Some(store), Some(session_id)) = (&self.sessions, session_id) {
            store.append_turn(session_id, role, content)?;
        }

        Ok(())
    }

    fn record_token_usage(
        &self,
        session_id: Option<&str>,
        scope: TokenScope,
        source: &str,
        usage: TokenUsage,
    ) -> Result<()> {
        if let (Some(store), Some(session_id)) = (&self.sessions, session_id) {
            store.record_token_usage(session_id, scope, source, usage)?;
        }

        Ok(())
    }

    fn enabled_skills_for_prompt(&self, prompt: &str) -> Result<Vec<crate::skills::Skill>> {
        self.skills
            .as_ref()
            .map(|store| store.enabled_for_prompt(prompt))
            .transpose()
            .map(Option::unwrap_or_default)
    }

    fn memory_lifecycle_strict(&self) -> bool {
        self.config.memory_lifecycle == MemoryLifecycleMode::Strict
    }

    async fn sync_memory_turn(
        &self,
        session_id: Option<&str>,
        user: &str,
        assistant: &str,
    ) -> Result<()> {
        if !self.memory_lifecycle_strict() {
            return Ok(());
        }

        self.memory
            .sync_turn(MemoryTurnInput {
                session_id: session_id.map(ToString::to_string),
                user: user.to_string(),
                assistant: assistant.to_string(),
                agent_id: self.config.agent_id.clone(),
            })
            .await
    }

    async fn end_memory_session(&self, session_id: &str) -> Result<()> {
        if !self.memory_lifecycle_strict() {
            return Ok(());
        }

        self.memory
            .end_session(MemorySessionEndInput {
                session_id: session_id.to_string(),
                agent_id: self.config.agent_id.clone(),
            })
            .await
            .map(|_| ())
    }

    async fn build_system_prompt(
        &self,
        prompt: &str,
        recall: bool,
        session_id: Option<String>,
        system_message: Option<String>,
    ) -> Result<String> {
        let memory_markdown = if recall {
            let memory = if self.memory_lifecycle_strict() {
                self.memory
                    .prefetch(MemoryPrefetchInput {
                        query: prompt.to_string(),
                        limit: RECALL_LIMIT,
                        depth: RECALL_DEPTH,
                        session_id: session_id.clone(),
                    })
                    .await?
            } else {
                self.memory
                    .recall(prompt, RECALL_LIMIT, RECALL_DEPTH)
                    .await?
            };
            Some(memory.markdown)
        } else {
            None
        };
        let context_files = discover_context_files(&self.config.working_dir)?;
        let system_message = self.context_system_message(prompt, system_message)?;

        let prompt_input = PromptInput {
            agent_id: self.config.agent_id.clone(),
            model: self.config.model.clone(),
            model_protocol: self.config.model_protocol,
            session_id,
            system_message,
            memory_markdown,
            context_files,
            date: conversation_date(),
        };

        Ok(self.prompt_builder.build(&prompt_input).assemble())
    }

    fn context_system_message(
        &self,
        prompt: &str,
        system_message: Option<String>,
    ) -> Result<Option<String>> {
        let mut parts = Vec::new();
        if let Some(system_message) = system_message
            && !system_message.trim().is_empty()
        {
            parts.push(system_message.trim().to_string());
        }

        for skill in self.enabled_skills_for_prompt(prompt)? {
            parts.push(format!(
                "Enabled skill available for this request:\n\n{}",
                skill.prompt_block()
            ));
        }

        Ok(if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        })
    }
}

fn normalize_chat_messages(messages: Vec<String>) -> Result<Vec<String>> {
    let prompts = messages
        .into_iter()
        .map(|message| message.trim().to_string())
        .filter(|message| !message.is_empty())
        .collect::<Vec<_>>();

    if prompts.is_empty() {
        bail!("chat requires at least one message");
    }

    Ok(prompts)
}

fn conversation_date() -> String {
    OffsetDateTime::now_local()
        .unwrap_or_else(|_| OffsetDateTime::now_utc())
        .date()
        .to_string()
}
