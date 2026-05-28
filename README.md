# Loka

Loka is a personal agent platform for CLI/TUI workflows, session search,
built-in learning loops, reusable skills, multi-agent runs, messaging gateways,
and portable runtimes.

It is built around a small control plane: typed model adapters, durable session
state, proposal-first memory writes, bounded worker agents, and explicit runtime
execution contracts.

## Status

Current implementation:

- `ask`, `chat`, and proposal-first `remember`
- persisted sessions with SQLite FTS search, summaries, and token usage records
- built-in learning proposals from completed sessions
- skill proposal, enablement, creation from sessions, and direct skill runs
- layered prompt assembly with workspace context discovery
- tool registry, approval policy checks, typed tool execution, and tool-call transcripts
- token accounting across prompts, tools, sessions, and workers
- typed eval fixtures for ask, chat, learning, skill creation, and multi-agent runs
- MCP adapter layer for external tool providers
- supervisor/worker multi-agent runs
- host, Docker, SSH, cloud VM, and serverless runtime executors
- Ratatui terminal operator interface
- Telegram webhook gateway

## Install

Install from source:

```bash
cargo install --path .
loka health
```

## Configuration

Loka reads config from `~/.loka/config.toml`, with environment variables taking precedence.

```toml
model_base_url = "http://127.0.0.1:8317"
model_api_key = "sk-development-example"
memory_base_url = "http://127.0.0.1:4321"
model = "gpt-5.5"
agent_id = "loka"
model_protocol = "openai-compatible"
memory_lifecycle = "off"
state_dir = "/home/dev/.loka"
working_dir = "/home/dev/.loka/workspace"
telegram_bot_token = "telegram-token"
```

Supported model protocols:

- `openai-compatible`
- `anthropic-compatible`

`memory_lifecycle = "strict"` enables memory prefetch, post-turn sync,
session-end extraction, and shutdown hooks. `memory_lifecycle = "off"` keeps
memory calls explicit.

State defaults to `~/.loka`. The default workspace is `~/.loka/workspace`.

## Commands

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
loka eval validate
loka tui --search runtime
loka gateway telegram --addr 127.0.0.1:8787 --path /telegram/webhook
```

Eval fixtures live in `evals/fixtures`:

```bash
loka eval list
loka eval validate
```

## Quality Gates

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all --check
```
