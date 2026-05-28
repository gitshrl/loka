# Loka

Loka is a personal agent platform for CLI/TUI workflows, session search, learning loops, reusable skills, multi-agent runs, messaging gateways, and portable runtimes.

## Status

This repository is under active product development. The current implementation includes:

- `ask`, `chat`, and proposal-first `remember`
- persisted sessions with SQLite FTS search
- built-in learning proposals from sessions
- skill proposal, enablement, and direct skill runs
- layered prompt assembly with workspace context discovery
- tool registry, approval policy checks, and typed tool execution
- token accounting across prompts, tools, sessions, and workers
- MCP adapter layer for external tool providers
- supervisor/worker multi-agent runs
- host, Docker, SSH, cloud VM, and serverless runtime executors
- Ratatui terminal operator interface
- Telegram webhook gateway

## Configuration

Loka reads config from `~/.loka/config.toml`, with environment variables taking precedence.

```toml
model_base_url = "http://127.0.0.1:8317"
model_api_key = "sk-development-example"
memory_base_url = "http://127.0.0.1:4321"
model = "gpt-5.5"
agent_id = "loka-agent"
model_protocol = "openai-compatible"
memory_lifecycle = "off"
state_dir = "/home/dev/.loka"
working_dir = "/home/dev/.loka/workspace"
telegram_bot_token = "telegram-token"
```

## Commands

Install the CLI once:

```bash
cargo install --path .
```

```bash
loka ask --recall --stream "what should i work on next?"
loka chat --recall
loka sessions search "runtime"
loka sessions summarize <session-id>
loka learn session <session-id>
loka learn review
loka skills propose --name "Rust review" --trigger "rust review" --instruction "Review code strictly."
loka skills propose-from-session <session-id>
loka run --agents "ship the next product slice"
loka runtime run --backend host -- printf ok
loka tui --search runtime
loka gateway telegram --addr 127.0.0.1:8787 --path /telegram/webhook
```

## Quality Gates

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all --check
```
