use anyhow::{Result, anyhow, bail};

use crate::config::AppConfig;
use crate::llm::{ChatRequest, LlmClient};
use crate::messages::{Message, Role, Transcript};
use crate::session::SessionStore;
use crate::skills::SkillStore;
use crate::wiki::{NoteInput, WikiClient};

const RECALL_LIMIT: u8 = 6;
const RECALL_DEPTH: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskRequest {
    pub prompt: String,
    pub recall: bool,
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
    llm: LlmClient,
    wiki: WikiClient,
    sessions: Option<SessionStore>,
    skills: Option<SkillStore>,
}

impl Agent {
    #[must_use]
    pub fn new(config: AppConfig) -> Self {
        Self {
            llm: LlmClient::new(
                config.pengepul_base_url.clone(),
                config.pengepul_api_key.clone(),
            ),
            wiki: WikiClient::new(config.wiki_base_url.clone()),
            config,
            sessions: None,
            skills: None,
        }
    }

    #[must_use]
    pub fn with_session_store(config: AppConfig, sessions: SessionStore) -> Self {
        Self {
            llm: LlmClient::new(
                config.pengepul_base_url.clone(),
                config.pengepul_api_key.clone(),
            ),
            wiki: WikiClient::new(config.wiki_base_url.clone()),
            config,
            sessions: Some(sessions),
            skills: None,
        }
    }

    #[must_use]
    pub fn with_stores(config: AppConfig, sessions: SessionStore, skills: SkillStore) -> Self {
        Self {
            llm: LlmClient::new(
                config.pengepul_base_url.clone(),
                config.pengepul_api_key.clone(),
            ),
            wiki: WikiClient::new(config.wiki_base_url.clone()),
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
        let mut transcript = Transcript::new();
        transcript.push(Message::system(system_prompt()));

        for skill in self.enabled_skills_for_prompt(&request.prompt)? {
            transcript.push(Message::system(format!(
                "Enabled skill available for this request:\n\n{}",
                skill.prompt_block()
            )));
        }

        if request.recall {
            let memory = self
                .wiki
                .rag(&request.prompt, RECALL_LIMIT, RECALL_DEPTH)
                .await?;
            if !memory.markdown.trim().is_empty() {
                transcript.push(Message::system(format!(
                    "Relevant memory from personal-wiki:\n\n{}",
                    memory.markdown.trim()
                )));
            }
        }

        transcript.push(Message::user(request.prompt.clone()));
        self.append_turn(session_id.as_deref(), Role::User, &request.prompt)?;

        let response = self
            .llm
            .chat(ChatRequest {
                model: self.config.model.clone(),
                messages: transcript.into_messages(),
            })
            .await?;

        self.append_turn(session_id.as_deref(), Role::Assistant, &response.content)?;

        Ok(AskOutput {
            answer: response.content,
            session_id,
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

        Ok(ChatSessionOutput {
            session_id: session.session_id,
            answers,
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
        let mut transcript = Transcript::new();
        transcript.push(Message::system(system_prompt()));
        Ok(ChatSession {
            session_id,
            transcript,
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
        let mut call_transcript = session.transcript.clone();
        for skill in self.enabled_skills_for_prompt(&prompt)? {
            call_transcript.push(Message::system(format!(
                "Enabled skill available for this request:\n\n{}",
                skill.prompt_block()
            )));
        }

        if session.recall {
            let memory = self.wiki.rag(&prompt, RECALL_LIMIT, RECALL_DEPTH).await?;
            if !memory.markdown.trim().is_empty() {
                call_transcript.push(Message::system(format!(
                    "Relevant memory from personal-wiki:\n\n{}",
                    memory.markdown.trim()
                )));
            }
        }

        let user = Message::user(prompt.clone());
        call_transcript.push(user.clone());
        sessions.append_turn(&session.session_id, Role::User, &prompt)?;

        let response = self
            .llm
            .chat(ChatRequest {
                model: self.config.model.clone(),
                messages: call_transcript.into_messages(),
            })
            .await?;

        sessions.append_turn(&session.session_id, Role::Assistant, &response.content)?;
        session.transcript.push(user);
        session
            .transcript
            .push(Message::assistant(response.content.clone()));
        Ok(response.content)
    }

    /// Creates a proposal-first memory note.
    ///
    /// # Errors
    ///
    /// Returns an error when `personal-wiki` rejects the note or does not return a proposal id.
    pub async fn remember(&self, title: String, body: String, tags: Vec<String>) -> Result<String> {
        self.wiki
            .add_note(NoteInput {
                title,
                body,
                kind: "note".to_string(),
                agent_id: self.config.agent_id.clone(),
                tags,
            })
            .await
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

    fn enabled_skills_for_prompt(&self, prompt: &str) -> Result<Vec<crate::skills::Skill>> {
        self.skills
            .as_ref()
            .map(|store| store.enabled_for_prompt(prompt))
            .transpose()
            .map(Option::unwrap_or_default)
    }
}

fn system_prompt() -> &'static str {
    "You are Loka, a personal agent platform. Use provided memory only when it is relevant. Be direct, concrete, and practical."
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
