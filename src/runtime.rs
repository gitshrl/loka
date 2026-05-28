use anyhow::{Context, Result, anyhow};
use futures::future::BoxFuture;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeCommand {
    pub program: String,
    pub args: Vec<String>,
    #[serde(rename = "workingDirectory")]
    pub working_dir: Option<String>,
    pub env: Vec<(String, String)>,
    pub stdin: Option<String>,
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeOutput {
    pub status: i32,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    pub duration: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessInvocation {
    pub program: String,
    pub args: Vec<String>,
    pub working_dir: Option<PathBuf>,
    pub env: Vec<(String, String)>,
    pub stdin: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeLimits {
    pub timeout: Duration,
    pub max_output_bytes: usize,
}

pub trait RuntimeExecutor {
    fn name(&self) -> &'static str;
    fn run(&self, command: RuntimeCommand) -> BoxFuture<'_, Result<RuntimeOutput>>;
}

pub trait ProcessRunner: Send + Sync {
    fn run(
        &self,
        invocation: ProcessInvocation,
        limits: RuntimeLimits,
    ) -> BoxFuture<'_, Result<RuntimeOutput>>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TokioProcessRunner;

pub struct HostExecutor {
    runner: Arc<dyn ProcessRunner>,
}

pub struct DockerExecutor {
    image: String,
    workspace_root: Option<PathBuf>,
    runner: Arc<dyn ProcessRunner>,
}

pub struct SshExecutor {
    target: String,
    remote_dir: String,
    runner: Arc<dyn ProcessRunner>,
}

pub struct CloudVmExecutor {
    ssh: SshExecutor,
    bootstrap: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ServerlessExecutor {
    endpoint: String,
    http: Client,
}

#[derive(Debug, Deserialize)]
struct ServerlessResponse {
    status: i32,
    stdout: String,
    stderr: String,
}

impl RuntimeCommand {
    #[must_use]
    pub fn limits(&self) -> RuntimeLimits {
        RuntimeLimits {
            timeout: Duration::from_secs(self.timeout_seconds.unwrap_or(30).max(1)),
            max_output_bytes: 64_000,
        }
    }
}

impl RuntimeExecutor for HostExecutor {
    fn name(&self) -> &'static str {
        "host"
    }

    fn run(&self, command: RuntimeCommand) -> BoxFuture<'_, Result<RuntimeOutput>> {
        Box::pin(async move {
            validate_runtime_command(&command)?;
            let limits = command.limits();
            let invocation = ProcessInvocation {
                program: command.program,
                args: command.args,
                working_dir: command.working_dir.map(PathBuf::from),
                env: command.env,
                stdin: command.stdin,
            };
            self.runner.run(invocation, limits).await
        })
    }
}

impl HostExecutor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            runner: Arc::new(TokioProcessRunner),
        }
    }

    #[must_use]
    pub fn with_runner(runner: impl ProcessRunner + 'static) -> Self {
        Self {
            runner: Arc::new(runner),
        }
    }
}

impl Default for HostExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeExecutor for DockerExecutor {
    fn name(&self) -> &'static str {
        "docker"
    }

    fn run(&self, command: RuntimeCommand) -> BoxFuture<'_, Result<RuntimeOutput>> {
        Box::pin(async move {
            validate_runtime_command(&command)?;
            let limits = command.limits();
            let invocation = self.invocation(command)?;
            self.runner.run(invocation, limits).await
        })
    }
}

impl DockerExecutor {
    /// Creates a Docker runtime executor.
    ///
    /// # Errors
    ///
    /// Returns an error when the image is empty or the workspace cannot be canonicalized.
    pub fn new(image: impl Into<String>, workspace_root: Option<impl AsRef<Path>>) -> Result<Self> {
        Self::with_runner(image, workspace_root, TokioProcessRunner)
    }

    /// Creates a Docker runtime executor with an injected process runner.
    ///
    /// # Errors
    ///
    /// Returns an error when the image is empty or the workspace cannot be canonicalized.
    pub fn with_runner<P>(
        image: impl Into<String>,
        workspace_root: Option<P>,
        runner: impl ProcessRunner + 'static,
    ) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let image = image.into();
        if image.trim().is_empty() {
            return Err(anyhow!("docker image is required"));
        }
        let workspace_root = workspace_root
            .map(|path| {
                path.as_ref()
                    .canonicalize()
                    .with_context(|| format!("canonicalize workspace {}", path.as_ref().display()))
            })
            .transpose()?;

        Ok(Self {
            image,
            workspace_root,
            runner: Arc::new(runner),
        })
    }

    fn invocation(&self, command: RuntimeCommand) -> Result<ProcessInvocation> {
        let mut args = vec!["run".to_string(), "--rm".to_string(), "-i".to_string()];

        for (key, value) in &command.env {
            validate_env_key(key)?;
            args.push("-e".to_string());
            args.push(format!("{key}={value}"));
        }

        if let Some(workspace_root) = &self.workspace_root {
            args.push("-v".to_string());
            args.push(format!("{}:/workspace", workspace_root.display()));
            args.push("--workdir".to_string());
            args.push(
                command
                    .working_dir
                    .as_deref()
                    .filter(|dir| !dir.trim().is_empty())
                    .unwrap_or("/workspace")
                    .to_string(),
            );
        } else if let Some(working_dir) = command.working_dir.as_deref() {
            args.push("--workdir".to_string());
            args.push(working_dir.to_string());
        }

        args.push(self.image.clone());
        args.push(command.program);
        args.extend(command.args);

        Ok(ProcessInvocation {
            program: "docker".to_string(),
            args,
            working_dir: None,
            env: Vec::new(),
            stdin: command.stdin,
        })
    }
}

impl RuntimeExecutor for SshExecutor {
    fn name(&self) -> &'static str {
        "ssh"
    }

    fn run(&self, command: RuntimeCommand) -> BoxFuture<'_, Result<RuntimeOutput>> {
        Box::pin(async move { self.run_with_prefix(command, None).await })
    }
}

impl SshExecutor {
    #[must_use]
    pub fn new(target: impl Into<String>, remote_dir: impl Into<String>) -> Self {
        Self::with_runner(target, remote_dir, TokioProcessRunner)
    }

    #[must_use]
    pub fn with_runner(
        target: impl Into<String>,
        remote_dir: impl Into<String>,
        runner: impl ProcessRunner + 'static,
    ) -> Self {
        Self {
            target: target.into(),
            remote_dir: remote_dir.into(),
            runner: Arc::new(runner),
        }
    }

    async fn run_with_prefix(
        &self,
        command: RuntimeCommand,
        prefix: Option<&str>,
    ) -> Result<RuntimeOutput> {
        validate_runtime_command(&command)?;
        if self.target.trim().is_empty() {
            return Err(anyhow!("ssh target is required"));
        }

        let limits = command.limits();
        let invocation = self.invocation(command, prefix)?;
        self.runner.run(invocation, limits).await
    }

    fn invocation(
        &self,
        command: RuntimeCommand,
        prefix: Option<&str>,
    ) -> Result<ProcessInvocation> {
        let remote_script = self.remote_script(&command, prefix)?;
        Ok(ProcessInvocation {
            program: "ssh".to_string(),
            args: vec![
                "-o".to_string(),
                "BatchMode=yes".to_string(),
                self.target.clone(),
                "sh".to_string(),
                "-lc".to_string(),
                remote_script,
            ],
            working_dir: None,
            env: Vec::new(),
            stdin: command.stdin,
        })
    }

    fn remote_script(&self, command: &RuntimeCommand, prefix: Option<&str>) -> Result<String> {
        let working_dir = command
            .working_dir
            .as_deref()
            .filter(|dir| !dir.trim().is_empty())
            .unwrap_or(&self.remote_dir);
        let command_script = shell_command(command)?;
        let script = format!("cd {} && {command_script}", shell_quote(working_dir));
        Ok(
            match prefix.map(str::trim).filter(|value| !value.is_empty()) {
                Some(prefix) => format!("{prefix} && {script}"),
                None => script,
            },
        )
    }
}

impl RuntimeExecutor for CloudVmExecutor {
    fn name(&self) -> &'static str {
        "cloud-vm"
    }

    fn run(&self, command: RuntimeCommand) -> BoxFuture<'_, Result<RuntimeOutput>> {
        Box::pin(async move {
            self.ssh
                .run_with_prefix(command, self.bootstrap.as_deref())
                .await
        })
    }
}

impl CloudVmExecutor {
    #[must_use]
    pub fn new(
        target: impl Into<String>,
        remote_dir: impl Into<String>,
        bootstrap: Option<String>,
    ) -> Self {
        Self::with_runner(target, remote_dir, bootstrap, TokioProcessRunner)
    }

    #[must_use]
    pub fn with_runner(
        target: impl Into<String>,
        remote_dir: impl Into<String>,
        bootstrap: Option<String>,
        runner: impl ProcessRunner + 'static,
    ) -> Self {
        Self {
            ssh: SshExecutor::with_runner(target, remote_dir, runner),
            bootstrap,
        }
    }
}

impl RuntimeExecutor for ServerlessExecutor {
    fn name(&self) -> &'static str {
        "serverless"
    }

    fn run(&self, command: RuntimeCommand) -> BoxFuture<'_, Result<RuntimeOutput>> {
        Box::pin(async move {
            validate_runtime_command(&command)?;
            let started = Instant::now();
            let response = self
                .http
                .post(&self.endpoint)
                .json(&command)
                .send()
                .await
                .context("send serverless runtime command")?;

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("serverless runtime failed with {status}: {body}"));
            }

            let body: ServerlessResponse = response
                .json()
                .await
                .context("parse serverless runtime response")?;
            Ok(RuntimeOutput {
                status: body.status,
                stdout: body.stdout,
                stderr: body.stderr,
                timed_out: false,
                duration: started.elapsed(),
            })
        })
    }
}

impl ServerlessExecutor {
    /// Creates a serverless runtime executor.
    ///
    /// # Errors
    ///
    /// Returns an error when the endpoint is not an HTTP or HTTPS URL.
    pub fn new(endpoint: impl Into<String>) -> Result<Self> {
        let endpoint = endpoint.into();
        let parsed = Url::parse(&endpoint).context("serverless endpoint must be a valid URL")?;
        match parsed.scheme() {
            "http" | "https" => Ok(Self {
                endpoint,
                http: Client::new(),
            }),
            scheme => Err(anyhow!(
                "serverless endpoint must use http or https, got {scheme}"
            )),
        }
    }
}

impl ProcessRunner for TokioProcessRunner {
    fn run(
        &self,
        invocation: ProcessInvocation,
        limits: RuntimeLimits,
    ) -> BoxFuture<'_, Result<RuntimeOutput>> {
        Box::pin(async move {
            let started = Instant::now();
            let mut command = Command::new(&invocation.program);
            command
                .args(&invocation.args)
                .envs(invocation.env.iter().map(|(key, value)| (key, value)));
            if let Some(working_dir) = invocation.working_dir.as_deref() {
                command.current_dir(working_dir);
            }
            let mut child = command
                .stdin(if invocation.stdin.is_some() {
                    Stdio::piped()
                } else {
                    Stdio::null()
                })
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true)
                .spawn()
                .with_context(|| format!("spawn {}", invocation.program))?;

            if let Some(stdin) = invocation.stdin {
                let mut child_stdin = child
                    .stdin
                    .take()
                    .ok_or_else(|| anyhow!("stdin pipe unavailable"))?;
                child_stdin
                    .write_all(stdin.as_bytes())
                    .await
                    .context("write runtime stdin")?;
            }

            let output = match timeout(limits.timeout, child.wait_with_output()).await {
                Ok(output) => output?,
                Err(_) => {
                    return Ok(RuntimeOutput {
                        status: -1,
                        stdout: String::new(),
                        stderr: format!("timed out after {:?}", limits.timeout),
                        timed_out: true,
                        duration: started.elapsed(),
                    });
                }
            };

            Ok(RuntimeOutput {
                status: output.status.code().unwrap_or(-1),
                stdout: truncate_bytes(&output.stdout, limits.max_output_bytes),
                stderr: truncate_bytes(&output.stderr, limits.max_output_bytes),
                timed_out: false,
                duration: started.elapsed(),
            })
        })
    }
}

fn validate_runtime_command(command: &RuntimeCommand) -> Result<()> {
    if command.program.trim().is_empty() {
        return Err(anyhow!("runtime command program is required"));
    }
    for (key, _) in &command.env {
        validate_env_key(key)?;
    }
    Ok(())
}

fn validate_env_key(key: &str) -> Result<()> {
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        return Err(anyhow!("environment variable key is required"));
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return Err(anyhow!("invalid environment variable key {key}"));
    }
    if chars.any(|ch| !(ch == '_' || ch.is_ascii_alphanumeric())) {
        return Err(anyhow!("invalid environment variable key {key}"));
    }
    Ok(())
}

fn shell_command(command: &RuntimeCommand) -> Result<String> {
    let mut parts = Vec::with_capacity(command.env.len() + command.args.len() + 1);
    for (key, value) in &command.env {
        validate_env_key(key)?;
        parts.push(format!("{key}={}", shell_quote(value)));
    }
    parts.push(shell_quote(&command.program));
    parts.extend(command.args.iter().map(|arg| shell_quote(arg)));
    Ok(parts.join(" "))
}

fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    if value
        .bytes()
        .all(|byte| matches!(byte, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'@' | b'%' | b'+' | b'=' | b':' | b',' | b'.' | b'/' | b'-'))
    {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn truncate_bytes(bytes: &[u8], max_bytes: usize) -> String {
    String::from_utf8_lossy(&bytes[..bytes.len().min(max_bytes)]).to_string()
}
