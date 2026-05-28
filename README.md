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
- supervisor/worker multi-agent runs
- host, Docker, SSH, cloud VM, and serverless runtime executors
- Ratatui terminal operator interface
- Telegram webhook gateway

## Configuration

Loka reads config from `~/.loka/config.toml`, with environment variables taking precedence.

```toml
pengepul_base_url = "http://127.0.0.1:8317"
pengepul_api_key = "sk-development-example"
wiki_base_url = "http://127.0.0.1:4321"
model = "gpt-5.5"
agent_id = "loka-agent"
provider_id = "pengepul"
state_dir = "/home/dev/.loka"
working_dir = "/home/dev/code/lab/loka"
telegram_bot_token = "telegram-token"
```

## Commands

```bash
cargo run -- ask --recall --stream "what should i work on next?"
cargo run -- chat --recall
cargo run -- sessions search "runtime"
cargo run -- learn session <session-id>
cargo run -- learn review
cargo run -- skills propose --name "Rust review" --trigger "rust review" --instruction "Review code strictly."
cargo run -- skills propose-from-session <session-id>
cargo run -- run --agents "ship the next product slice"
cargo run -- runtime run --backend host -- printf ok
cargo run -- tui --search runtime
cargo run -- gateway telegram --addr 127.0.0.1:8787 --path /telegram/webhook
```

## Quality Gates

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all --check
```
