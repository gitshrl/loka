use anyhow::{Context, Result, anyhow};
use ignore::{WalkBuilder, overrides::OverrideBuilder};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

use crate::session::SessionStore;
use crate::wiki::{NoteInput, WikiClient};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub input: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ToolResult {
    pub output: Value,
}

#[derive(Debug)]
pub struct ToolRuntime {
    sessions: SessionStore,
    wiki: Option<WikiClient>,
    agent_id: String,
    host: Option<HostRuntime>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostRuntime {
    workspace_root: PathBuf,
    max_file_bytes: u64,
    max_search_results: usize,
    max_output_bytes: usize,
    shell_timeout: Duration,
}

#[derive(Debug, Deserialize)]
struct SessionListInput {
    limit: Option<u16>,
}

#[derive(Debug, Deserialize)]
struct SessionSearchInput {
    query: String,
    limit: Option<u16>,
}

#[derive(Debug, Deserialize)]
struct WikiRagInput {
    query: String,
    limit: Option<u8>,
    depth: Option<u8>,
}

#[derive(Debug, Deserialize)]
struct WikiAddNoteInput {
    title: String,
    body: String,
    tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct ReadFileInput {
    path: String,
}

#[derive(Debug, Deserialize)]
struct SearchFilesInput {
    query: String,
    glob: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GitStatusInput {
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ShellInput {
    command: String,
    working_directory: Option<String>,
}

#[derive(Debug, Serialize)]
struct FileSearchHit {
    path: String,
    line: usize,
    text: String,
}

impl ToolRuntime {
    #[must_use]
    pub fn new(sessions: SessionStore) -> Self {
        Self {
            sessions,
            wiki: None,
            agent_id: "loka-agent".to_string(),
            host: None,
        }
    }

    #[must_use]
    pub fn with_wiki(mut self, wiki: WikiClient, agent_id: impl Into<String>) -> Self {
        self.wiki = Some(wiki);
        self.agent_id = agent_id.into();
        self
    }

    /// Adds host workspace execution support.
    ///
    /// # Errors
    ///
    /// Returns an error when the workspace cannot be canonicalized.
    pub fn with_host_workspace(mut self, workspace_root: impl AsRef<Path>) -> Result<Self> {
        self.host = Some(HostRuntime::new(workspace_root)?);
        Ok(self)
    }

    /// Executes a supported tool call.
    ///
    /// # Errors
    ///
    /// Returns an error when tool input is invalid, required services are not configured,
    /// the backing service rejects the request, or the registered tool has no runtime executor yet.
    pub async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        match call.name.as_str() {
            "session_list" => self.execute_session_list(call.input),
            "session_search" => self.execute_session_search(call.input),
            "wiki_rag" => self.execute_wiki_rag(call.input).await,
            "wiki_add_note" => self.execute_wiki_add_note(call.input).await,
            "read_file" => self.execute_read_file(call.input),
            "search_files" => self.execute_search_files(call.input),
            "git_status" => self.execute_git_status(call.input).await,
            "shell" => self.execute_shell(call.input).await,
            name => Err(anyhow!("tool {name} has no runtime executor")),
        }
    }

    /// Executes a supported tool call and persists its transcript in the session store.
    ///
    /// # Errors
    ///
    /// Returns an error when the tool call cannot be recorded, the tool call itself fails,
    /// or the final tool result cannot be persisted.
    pub async fn execute_in_session(&self, session_id: &str, call: ToolCall) -> Result<ToolResult> {
        let call_id =
            self.sessions
                .record_tool_call_started(session_id, &call.name, &call.input)?;

        match self.execute(call).await {
            Ok(result) => {
                self.sessions
                    .record_tool_call_completed(&call_id, &result.output)?;
                Ok(result)
            }
            Err(error) => {
                let error_text = error.to_string();
                self.sessions
                    .record_tool_call_failed(&call_id, &error_text)
                    .with_context(|| format!("record failed tool call {call_id}"))?;
                Err(error)
            }
        }
    }

    fn execute_session_list(&self, input: Value) -> Result<ToolResult> {
        let input: SessionListInput = serde_json::from_value(input)?;
        let sessions = self.sessions.list_sessions(input.limit.unwrap_or(20))?;
        Ok(ToolResult {
            output: json!({ "sessions": sessions }),
        })
    }

    fn execute_session_search(&self, input: Value) -> Result<ToolResult> {
        let input: SessionSearchInput = serde_json::from_value(input)?;
        let hits = self
            .sessions
            .search(&input.query, input.limit.unwrap_or(20))?;
        Ok(ToolResult {
            output: json!({ "hits": hits }),
        })
    }

    async fn execute_wiki_rag(&self, input: Value) -> Result<ToolResult> {
        let input: WikiRagInput = serde_json::from_value(input)?;
        let wiki = self
            .wiki
            .as_ref()
            .ok_or_else(|| anyhow!("wiki_rag requires personal-wiki configuration"))?;
        let context = wiki
            .rag(
                &input.query,
                input.limit.unwrap_or(6),
                input.depth.unwrap_or(1),
            )
            .await?;
        Ok(ToolResult {
            output: json!({ "context": context }),
        })
    }

    async fn execute_wiki_add_note(&self, input: Value) -> Result<ToolResult> {
        let input: WikiAddNoteInput = serde_json::from_value(input)?;
        let wiki = self
            .wiki
            .as_ref()
            .ok_or_else(|| anyhow!("wiki_add_note requires personal-wiki configuration"))?;
        let proposal_id = wiki
            .add_note(NoteInput {
                title: input.title,
                body: input.body,
                kind: "note".to_string(),
                agent_id: self.agent_id.clone(),
                tags: input.tags.unwrap_or_default(),
            })
            .await?;
        Ok(ToolResult {
            output: json!({ "proposal_id": proposal_id }),
        })
    }

    fn execute_read_file(&self, input: Value) -> Result<ToolResult> {
        let input: ReadFileInput = serde_json::from_value(input)?;
        let host = self.host()?;
        let path = host.resolve_existing_file(&input.path)?;
        let metadata = path.metadata()?;
        if metadata.len() > host.max_file_bytes {
            return Err(anyhow!(
                "file {} exceeds max readable size of {} bytes",
                path.display(),
                host.max_file_bytes
            ));
        }

        let content = fs::read_to_string(&path)
            .with_context(|| format!("read UTF-8 file {}", path.display()))?;
        Ok(ToolResult {
            output: json!({
                "path": host.display_path(&path),
                "bytes": content.len(),
                "content": content,
            }),
        })
    }

    fn execute_search_files(&self, input: Value) -> Result<ToolResult> {
        let input: SearchFilesInput = serde_json::from_value(input)?;
        let host = self.host()?;
        let mut hits = Vec::with_capacity(host.max_search_results);
        let needle = input.query;
        if needle.is_empty() {
            return Err(anyhow!("search query is required"));
        }

        let mut builder = WalkBuilder::new(&host.workspace_root);
        builder
            .hidden(false)
            .git_ignore(true)
            .git_exclude(true)
            .require_git(false);
        if let Some(glob) = input.glob.as_deref().filter(|glob| !glob.trim().is_empty()) {
            let mut overrides = OverrideBuilder::new(&host.workspace_root);
            overrides.add(glob)?;
            builder.overrides(overrides.build()?);
        }

        for entry in builder.build().filter_map(std::result::Result::ok) {
            if hits.len() >= host.max_search_results {
                break;
            }

            let path = entry.path();
            if !path.is_file()
                || path
                    .metadata()
                    .map_or(true, |metadata| metadata.len() > host.max_file_bytes)
            {
                continue;
            }

            let Ok(file) = fs::File::open(path) else {
                continue;
            };
            let mut reader = BufReader::new(file);
            let mut line = String::new();
            let mut line_number = 0;

            loop {
                line.clear();
                let bytes = reader.read_line(&mut line);
                let Ok(bytes) = bytes else {
                    break;
                };
                if bytes == 0 {
                    break;
                }

                line_number += 1;
                if line.contains(&needle) {
                    hits.push(FileSearchHit {
                        path: host.display_path(path),
                        line: line_number,
                        text: truncate(line.trim(), 512),
                    });
                    if hits.len() >= host.max_search_results {
                        break;
                    }
                }
            }
        }

        Ok(ToolResult {
            output: json!({ "hits": hits }),
        })
    }

    async fn execute_git_status(&self, input: Value) -> Result<ToolResult> {
        let input: GitStatusInput = serde_json::from_value(input)?;
        let host = self.host()?;
        let working_dir = host.resolve_directory(input.path.as_deref().unwrap_or("."))?;
        let output = host
            .run_command("git", &["status", "--short"], &working_dir)
            .await?;
        Ok(ToolResult {
            output: json!({
                "status": output.status,
                "stdout": output.stdout,
                "stderr": output.stderr,
            }),
        })
    }

    async fn execute_shell(&self, input: Value) -> Result<ToolResult> {
        let input: ShellInput = serde_json::from_value(input)?;
        let host = self.host()?;
        let working_dir =
            host.resolve_directory(input.working_directory.as_deref().unwrap_or("."))?;
        let output = host.run_shell(&input.command, &working_dir).await?;
        Ok(ToolResult {
            output: json!({
                "status": output.status,
                "stdout": output.stdout,
                "stderr": output.stderr,
            }),
        })
    }

    fn host(&self) -> Result<&HostRuntime> {
        self.host
            .as_ref()
            .ok_or_else(|| anyhow!("host tool requires host workspace configuration"))
    }
}

impl HostRuntime {
    /// Creates host runtime limits for a workspace.
    ///
    /// # Errors
    ///
    /// Returns an error when the workspace cannot be canonicalized.
    pub fn new(workspace_root: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            workspace_root: workspace_root.as_ref().canonicalize().with_context(|| {
                format!(
                    "canonicalize workspace {}",
                    workspace_root.as_ref().display()
                )
            })?,
            max_file_bytes: 1_048_576,
            max_search_results: 100,
            max_output_bytes: 64_000,
            shell_timeout: Duration::from_secs(30),
        })
    }

    fn resolve_existing_file(&self, requested: &str) -> Result<PathBuf> {
        let path = self.resolve_path(requested)?;
        if !path.is_file() {
            return Err(anyhow!("{} is not a file", path.display()));
        }
        Ok(path)
    }

    fn resolve_directory(&self, requested: &str) -> Result<PathBuf> {
        let path = self.resolve_path(requested)?;
        if !path.is_dir() {
            return Err(anyhow!("{} is not a directory", path.display()));
        }
        Ok(path)
    }

    fn resolve_path(&self, requested: &str) -> Result<PathBuf> {
        let requested = requested.trim();
        if requested.is_empty() {
            return Err(anyhow!("path is required"));
        }

        let path = self.workspace_root.join(requested).canonicalize()?;
        if !path.starts_with(&self.workspace_root) {
            return Err(anyhow!(
                "path {} escapes workspace {}",
                path.display(),
                self.workspace_root.display()
            ));
        }
        Ok(path)
    }

    fn display_path(&self, path: &Path) -> String {
        path.strip_prefix(&self.workspace_root)
            .unwrap_or(path)
            .display()
            .to_string()
    }

    async fn run_command(
        &self,
        program: &str,
        args: &[&str],
        working_dir: &Path,
    ) -> Result<CommandOutput> {
        let child = Command::new(program)
            .args(args)
            .current_dir(working_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("spawn {program}"))?;

        let output = timeout(self.shell_timeout, child.wait_with_output())
            .await
            .map_err(|_| anyhow!("{program} timed out after {:?}", self.shell_timeout))??;

        Ok(CommandOutput {
            status: output.status.code().unwrap_or(-1),
            stdout: truncate_bytes(&output.stdout, self.max_output_bytes),
            stderr: truncate_bytes(&output.stderr, self.max_output_bytes),
        })
    }

    async fn run_shell(&self, command: &str, working_dir: &Path) -> Result<CommandOutput> {
        self.run_command("sh", &["-lc", command], working_dir).await
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CommandOutput {
    status: i32,
    stdout: String,
    stderr: String,
}

fn truncate(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn truncate_bytes(bytes: &[u8], max_bytes: usize) -> String {
    String::from_utf8_lossy(&bytes[..bytes.len().min(max_bytes)]).to_string()
}
