use futures::future::BoxFuture;
use httpmock::prelude::*;
use loka::runtime::{
    CloudVmExecutor, DockerExecutor, HostExecutor, ProcessInvocation, ProcessRunner,
    RuntimeCommand, RuntimeExecutor, RuntimeLimits, RuntimeOutput, ServerlessExecutor, SshExecutor,
};
use serde_json::json;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[tokio::test]
async fn host_executor_runs_command_in_working_directory() {
    let workspace = tempfile::tempdir().expect("workspace");
    let output = HostExecutor::new()
        .run(RuntimeCommand {
            program: "sh".to_string(),
            args: vec!["-lc".to_string(), "printf \"$PWD\"".to_string()],
            working_dir: Some(workspace.path().display().to_string()),
            env: vec![],
            stdin: None,
            timeout_seconds: Some(5),
        })
        .await
        .expect("host command");

    assert_eq!(output.status, 0);
    assert_eq!(output.stdout, workspace.path().display().to_string());
}

#[tokio::test]
async fn docker_executor_builds_container_command_without_requiring_docker() {
    let workspace = tempfile::tempdir().expect("workspace");
    let runner = CaptureRunner::new(runtime_success("ok"));
    let output = DockerExecutor::with_runner("rust:1.95", Some(workspace.path()), runner.clone())
        .expect("docker executor")
        .run(RuntimeCommand {
            program: "cargo".to_string(),
            args: vec!["test".to_string()],
            working_dir: Some("/workspace".to_string()),
            env: vec![("RUST_LOG".to_string(), "debug".to_string())],
            stdin: Some("input".to_string()),
            timeout_seconds: Some(20),
        })
        .await
        .expect("docker command");

    assert_eq!(output.stdout, "ok");
    let calls = runner.calls();
    assert_eq!(calls.len(), 1);
    let call = &calls[0];
    assert_eq!(call.program, "docker");
    assert_eq!(call.stdin.as_deref(), Some("input"));
    assert!(call.args.contains(&"run".to_string()));
    assert!(call.args.contains(&"--rm".to_string()));
    assert!(call.args.contains(&"rust:1.95".to_string()));
    assert!(call.args.contains(&"cargo".to_string()));
    assert!(call.args.contains(&"test".to_string()));
    assert!(call.args.contains(&"RUST_LOG=debug".to_string()));
    assert!(call.args.iter().any(|arg| arg.ends_with(":/workspace")));
}

#[tokio::test]
async fn ssh_executor_builds_remote_shell_command() {
    let runner = CaptureRunner::new(runtime_success("remote"));
    let output = SshExecutor::with_runner("dev@example.com", "/srv/loka", runner.clone())
        .run(RuntimeCommand {
            program: "printf".to_string(),
            args: vec!["hello world".to_string()],
            working_dir: None,
            env: vec![("LOKA_ENV".to_string(), "prod".to_string())],
            stdin: None,
            timeout_seconds: Some(10),
        })
        .await
        .expect("ssh command");

    assert_eq!(output.stdout, "remote");
    let calls = runner.calls();
    let call = &calls[0];
    assert_eq!(call.program, "ssh");
    assert!(call.args.contains(&"dev@example.com".to_string()));
    let script = call.args.last().expect("remote script");
    assert!(script.contains("cd /srv/loka"));
    assert!(script.contains("LOKA_ENV=prod"));
    assert!(script.contains("printf 'hello world'"));
}

#[tokio::test]
async fn cloud_vm_executor_bootstraps_before_running_remote_command() {
    let runner = CaptureRunner::new(runtime_success("vm"));
    let output = CloudVmExecutor::with_runner(
        "ubuntu@vm.example.com",
        "/srv/loka",
        Some("test -d /srv/loka || mkdir -p /srv/loka".to_string()),
        runner.clone(),
    )
    .run(RuntimeCommand {
        program: "loka-worker".to_string(),
        args: vec!["run".to_string()],
        working_dir: None,
        env: vec![],
        stdin: None,
        timeout_seconds: Some(30),
    })
    .await
    .expect("cloud vm command");

    assert_eq!(output.stdout, "vm");
    let script = runner.calls()[0]
        .args
        .last()
        .expect("remote script")
        .clone();
    assert!(script.contains("test -d /srv/loka"));
    assert!(script.contains("&& loka-worker run"));
}

#[tokio::test]
async fn serverless_executor_posts_command_protocol() {
    let server = MockServer::start();
    let run = server.mock(|when, then| {
        when.method(POST).path("/run").json_body(json!({
            "program": "loka-worker",
            "args": ["run"],
            "workingDirectory": "/tmp",
            "env": [["LOKA_ENV", "serverless"]],
            "stdin": null,
            "timeoutSeconds": 15
        }));

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "status": 0,
                "stdout": "done",
                "stderr": ""
            }));
    });

    let output = ServerlessExecutor::new(format!("{}/run", server.base_url()))
        .expect("serverless executor")
        .run(RuntimeCommand {
            program: "loka-worker".to_string(),
            args: vec!["run".to_string()],
            working_dir: Some("/tmp".to_string()),
            env: vec![("LOKA_ENV".to_string(), "serverless".to_string())],
            stdin: None,
            timeout_seconds: Some(15),
        })
        .await
        .expect("serverless command");

    run.assert();
    assert_eq!(output.status, 0);
    assert_eq!(output.stdout, "done");
}

#[derive(Clone)]
struct CaptureRunner {
    calls: Arc<Mutex<Vec<ProcessInvocation>>>,
    output: RuntimeOutput,
}

impl CaptureRunner {
    fn new(output: RuntimeOutput) -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            output,
        }
    }

    fn calls(&self) -> Vec<ProcessInvocation> {
        self.calls.lock().expect("calls").clone()
    }
}

impl ProcessRunner for CaptureRunner {
    fn run(
        &self,
        invocation: ProcessInvocation,
        _limits: RuntimeLimits,
    ) -> BoxFuture<'_, anyhow::Result<RuntimeOutput>> {
        Box::pin(async move {
            self.calls.lock().expect("calls").push(invocation);
            Ok(self.output.clone())
        })
    }
}

fn runtime_success(stdout: &str) -> RuntimeOutput {
    RuntimeOutput {
        status: 0,
        stdout: stdout.to_string(),
        stderr: String::new(),
        timed_out: false,
        duration: Duration::ZERO,
    }
}
