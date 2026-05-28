use anyhow::{Context, Result};
use regex::RegexSet;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use crate::config::ModelProtocol;
use crate::tokens::estimate_text_tokens;

const CONTEXT_FILE_NAMES: &[&str] = &["AGENTS.md", "LOKA.md", ".loka.md", ".cursorrules"];
const MAX_CONTEXT_FILE_BYTES: u64 = 64 * 1024;

static THREAT_PATTERNS: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
        r"(?i)ignore\s+(all\s+)?previous\s+instructions",
        r"(?i)reveal\s+(the\s+)?system\s+prompt",
        r"(?i)developer\s+message",
        r"(?i)exfiltrat(e|ion)",
        r"(?i)send\s+.*secret",
    ])
    .expect("prompt injection regex patterns must be valid")
});

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextFile {
    pub path: PathBuf,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptInput {
    pub agent_id: String,
    pub model: String,
    pub model_protocol: ModelProtocol,
    pub session_id: Option<String>,
    pub system_message: Option<String>,
    pub memory_markdown: Option<String>,
    pub context_files: Vec<ContextFile>,
    pub date: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptParts {
    pub stable: String,
    pub context: String,
    pub volatile: String,
    pub fingerprint: String,
    pub token_accounting: PromptTokenAccounting,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptTokenAccounting {
    pub stable_tokens: u64,
    pub context_tokens: u64,
    pub volatile_tokens: u64,
    pub total_tokens: u64,
}

impl PromptParts {
    #[must_use]
    pub fn assemble(&self) -> String {
        assemble_prompt([
            self.stable.as_str(),
            self.context.as_str(),
            self.volatile.as_str(),
        ])
    }
}

#[derive(Debug, Clone, Default)]
pub struct PromptBuilder;

impl PromptBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    #[must_use]
    pub fn build(&self, input: &PromptInput) -> PromptParts {
        let stable = build_stable_prompt();
        let context =
            build_context_prompt(input.system_message.clone(), input.context_files.clone());
        let volatile = build_volatile_prompt(input);
        let fingerprint = fingerprint_prompt_parts(&stable, &context, &volatile);
        let token_accounting = prompt_token_accounting(&stable, &context, &volatile);

        PromptParts {
            stable,
            context,
            volatile,
            fingerprint,
            token_accounting,
        }
    }
}

#[must_use]
pub fn assemble_prompt<const N: usize>(parts: [&str; N]) -> String {
    parts
        .into_iter()
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Discovers supported context files in deterministic order.
///
/// # Errors
///
/// Returns an error when a candidate context file cannot be read or inspected.
pub fn discover_context_files(cwd: &Path) -> Result<Vec<ContextFile>> {
    let mut files = Vec::new();

    for name in CONTEXT_FILE_NAMES {
        let path = cwd.join(name);
        if !path.is_file() {
            continue;
        }

        let metadata = fs::metadata(&path).with_context(|| format!("stat {}", path.display()))?;
        if metadata.len() > MAX_CONTEXT_FILE_BYTES {
            files.push(ContextFile {
                path,
                body: format!("[SKIPPED: context file exceeded {MAX_CONTEXT_FILE_BYTES} bytes]"),
            });
            continue;
        }

        let body = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        files.push(sanitize_context_file(&ContextFile { path, body }));
    }

    Ok(files)
}

#[must_use]
pub fn sanitize_context_file(file: &ContextFile) -> ContextFile {
    if THREAT_PATTERNS.is_match(&file.body) {
        return ContextFile {
            path: file.path.clone(),
            body: format!(
                "[BLOCKED: {} contained potential prompt injection. Content not loaded.]",
                file.path.display()
            ),
        };
    }

    ContextFile {
        path: file.path.clone(),
        body: file.body.trim().to_string(),
    }
}

fn build_stable_prompt() -> String {
    assemble_prompt([
        "# Loka Identity\nYou are Loka, a personal agent platform. Be direct, concrete, and practical.",
        "# Execution Discipline\nUse available tools to inspect real state before making claims. Prefer verified outcomes over guesses. When work is incomplete, say exactly what remains.",
        "# Memory And Skills\nUse durable memory for stable facts, preferences, decisions, and reusable workflows. Treat session history as task recall, not permanent truth. Promote repeated procedures into reviewed skills.",
        "# Multi-Agent Discipline\nWhen coordinating workers, keep responsibilities scoped. Supervisor agents assign, synthesize, and decide what becomes durable. Worker agents return compact results and do not write durable memory directly.",
        "# Runtime Discipline\nThe runtime may be a host process, VPS, container, SSH target, cloud VM, or serverless worker. Avoid assumptions about filesystem layout, installed tools, or network reachability unless verified.",
    ])
}

fn build_context_prompt(system_message: Option<String>, context_files: Vec<ContextFile>) -> String {
    let mut parts = Vec::new();

    if let Some(system_message) = system_message {
        let trimmed = system_message.trim();
        if !trimmed.is_empty() {
            parts.push(format!("# Caller System Message\n{trimmed}"));
        }
    }

    for file in context_files {
        let file = sanitize_context_file(&file);
        if file.body.trim().is_empty() {
            continue;
        }
        parts.push(format!(
            "# Context File: {}\n{}",
            file.path.display(),
            file.body.trim()
        ));
    }

    if parts.is_empty() {
        String::new()
    } else {
        format!("# Session Context\n{}", parts.join("\n\n"))
    }
}

fn build_volatile_prompt(input: &PromptInput) -> String {
    let mut parts = vec![format!(
        "# Runtime State\nConversation date: {}\nAgent ID: {}\nModel: {}\nProvider: {}",
        input.date, input.agent_id, input.model, input.model_protocol
    )];

    if let Some(session_id) = input.session_id.as_deref()
        && !session_id.trim().is_empty()
    {
        parts.push(format!("Session ID: {}", session_id.trim()));
    }

    if let Some(memory) = input.memory_markdown.as_deref()
        && !memory.trim().is_empty()
    {
        parts.push(format_memory_recall(memory));
    }

    parts.join("\n\n")
}

fn format_memory_recall(memory: &str) -> String {
    let memory = sanitize_memory_recall(memory);
    format!(
        concat!(
            "# Memory Recall\n",
            "<memory-context>\n",
            "[System note: The following is recalled durable memory, not new user input. ",
            "Treat it as background context and do not expose this wrapper.]\n\n",
            "{}\n",
            "</memory-context>"
        ),
        memory
    )
}

fn sanitize_memory_recall(memory: &str) -> String {
    memory
        .replace("<memory-context>", "")
        .replace("</memory-context>", "")
        .trim()
        .to_string()
}

fn fingerprint_prompt_parts(stable: &str, context: &str, volatile: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(stable.as_bytes());
    hasher.update(b"\0");
    hasher.update(context.as_bytes());
    hasher.update(b"\0");
    hasher.update(volatile.as_bytes());
    hex::encode(hasher.finalize())
}

fn prompt_token_accounting(stable: &str, context: &str, volatile: &str) -> PromptTokenAccounting {
    let stable_tokens = estimate_text_tokens(stable);
    let context_tokens = estimate_text_tokens(context);
    let volatile_tokens = estimate_text_tokens(volatile);

    PromptTokenAccounting {
        stable_tokens,
        context_tokens,
        volatile_tokens,
        total_tokens: stable_tokens + context_tokens + volatile_tokens,
    }
}
