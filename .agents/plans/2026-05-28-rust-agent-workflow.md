# Rust Agent Workflow Implementation Plan

Updated: 2026-05-29

## Goal

Build Loka as a real Rust agent platform for CLI/TUI workflows, session search,
built-in learning, reusable skill creation, multi-agent runs, messaging
gateways, and portable runtimes.

This is product work, not an MVP. The core must be safe, fast, testable, and
operable from a laptop, VPS, container, SSH target, cloud VM, or serverless
worker.

## Architecture

Loka owns the agent control plane:

- Agent loop and layered prompt assembly.
- Session persistence, search, summaries, and tool-call transcript records.
- Learning loop that extracts durable knowledge from completed sessions.
- Proposal-first memory writes.
- Skill proposal, review, enablement, and runtime loading.
- Supervisor/worker multi-agent execution.
- CLI, TUI, and messaging gateway surfaces.
- Runtime executors for host, Docker, SSH, cloud VM, and serverless backends.

Loka does not hard-code a concrete model service or durable memory product.
External integrations sit behind adapters:

- Model API adapter: `openai-compatible` or `anthropic-compatible`.
- Memory API adapter: recall, proposal-first write, and pending proposal list.
- MCP adapter: external tools are loaded through JSON-RPC tool discovery and
  execution, then exposed through the same approval policy as built-in tools.

Concrete third-party product names are forbidden in `src/`, `tests/`, and
`README.md`. Use generic model, memory, runtime, skill, and gateway language.

## Runtime Contracts

Config lives in `~/.loka/config.toml`. Environment variables override file
config.

```toml
model_base_url = "http://127.0.0.1:8317"
model_api_key = "sk-development-example"
model_protocol = "openai-compatible" # or "anthropic-compatible"
memory_base_url = "http://127.0.0.1:4321"
memory_lifecycle = "off" # or "strict"
model = "gpt-5.5"
agent_id = "loka-agent"
state_dir = "/home/dev/.loka"
working_dir = "/home/dev/.loka/workspace"
telegram_bot_token = "telegram-token"
```

Environment variables:

```bash
LOKA_MODEL_BASE_URL
LOKA_MODEL_API_KEY
LOKA_MODEL_PROTOCOL
LOKA_MEMORY_BASE_URL
LOKA_MEMORY_LIFECYCLE
LOKA_MODEL
LOKA_AGENT_ID
LOKA_STATE_DIR
LOKA_WORKING_DIR
LOKA_TELEGRAM_BOT_TOKEN
```

## Harness Principles

- Typed transcript and provider adapters. Internal message shape is stable;
  adapter code absorbs OpenAI-compatible and Anthropic-compatible wire formats.
- Tool definitions are contracts: semantic names, JSON schemas, access class,
  and permission policy.
- Memory recall is fenced as durable context, not treated as new user input.
- Learning is proposal-first. The model can suggest durable knowledge; the
  system records a reviewable proposal instead of silently mutating memory.
- Session state is durable. Turns, tool calls, summaries, and learning outputs
  must survive process restarts.
- Multi-agent execution is bounded. Supervisor assigns and synthesizes; workers
  return compact results and do not directly write durable memory.
- Runtime tools are defensive at I/O edges. Workspace paths, command execution,
  and remote backends must be validated before use.

## Current Product Surface

Commands:

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

Built-in tools:

- `session_list`
- `session_search`
- `memory_search`
- `memory_propose`
- `read_file`
- `search_files`
- `git_status`
- `shell`
- `learn_session`

Eval fixtures:

- `evals/fixtures/ask.json`
- `evals/fixtures/chat.json`
- `evals/fixtures/learning.json`
- `evals/fixtures/skill-creation.json`
- `evals/fixtures/multi-agent.json`

## Implementation Tasks

- [x] Bootstrap Rust 2024 crate with Rust `1.95.0`.
- [x] Add CLI command tree with direct `loka ...` binary support.
- [x] Load config from `~/.loka/config.toml` plus environment overrides.
- [x] Add `ModelProtocol` with only `openai-compatible` and
  `anthropic-compatible`.
- [x] Implement OpenAI-compatible chat and streaming.
- [x] Implement Anthropic-compatible chat and streaming.
- [x] Add generic memory API client.
- [x] Add layered prompt builder and workspace context discovery.
- [x] Add SQLite session store and FTS search.
- [x] Persist tool-call transcript records.
- [x] Add session summaries and learning proposals.
- [x] Add skill proposals, enablement, and direct skill execution.
- [x] Add tool registry and approval policy.
- [x] Add typed runtime execution for host, Docker, SSH, cloud VM, and serverless.
- [x] Add supervisor/worker multi-agent runtime.
- [x] Add TUI and Telegram gateway.
- [x] Add explicit MCP adapter layer for external tool/skill providers.
- [x] Add memory provider lifecycle hooks for prefetch, post-turn sync,
  session-end extraction, and shutdown.
- [x] Add token budget accounting for prompt, tool, worker, and session scopes.
- [x] Add eval fixtures for ask, chat, learning, skill creation, and multi-agent
  runs.

## Verification

Final gates run after implementation:

```bash
rtk cargo test
rtk cargo clippy --all-targets --all-features -- -D warnings
rtk cargo fmt --all --check
rtk proxy git diff --check
rtk /home/dev/.cargo/bin/loka health
rtk /home/dev/.cargo/bin/loka eval validate
rtk rg -n -i "<retired integration name pattern>" src tests README.md Cargo.toml rust-toolchain.toml evals .agents/plans/2026-05-28-rust-agent-workflow.md .agents/plans/2026-05-28-layered-prompt-method.md
```

Results:

- `cargo test`: 100 passed.
- `cargo clippy --all-targets --all-features -- -D warnings`: clean.
- `cargo fmt --all --check`: clean.
- `loka health`: `ok`.
- `loka eval validate`: validated 5 eval fixtures.
- Forbidden-name search: no matches.
