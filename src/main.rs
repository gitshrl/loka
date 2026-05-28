use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use loka_agent::agent::{Agent, AskRequest, ChatSessionRequest};
use loka_agent::config::AppConfig;
use loka_agent::learning::{
    LearnSessionOutput, LearnSessionRequest, LearningEngine, pending_learning_proposals,
};
use loka_agent::llm::{ChatRequest, LlmClient};
use loka_agent::messages::Message;
use loka_agent::multi_agent::{
    AgentProfile, MultiAgentRunRequest, MultiAgentRuntime, TaskGraphStore, WorkerSpec,
};
use loka_agent::permissions::{ApprovalPolicy, PermissionMode};
use loka_agent::session::SessionStore;
use loka_agent::skills::{SkillDraft, SkillStatus, SkillStore};
use loka_agent::tools::ToolRegistry;
use loka_agent::wiki::WikiClient;
use std::fmt;
use std::io::{self, Write};

#[derive(Debug, Parser)]
#[command(name = "loka")]
#[command(about = "personal agent platform")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(about = "ask the agent one question")]
    Ask {
        #[arg(help = "Question or instruction to send to the agent")]
        prompt: String,
        #[arg(long, help = "Inject relevant personal-wiki context before answering")]
        recall: bool,
        #[arg(long, help = "Session id to expose in the prompt runtime state")]
        session_id: Option<String>,
        #[arg(long, help = "Additional caller system message for this request")]
        system_message: Option<String>,
    },
    #[command(about = "chat with the agent in one persisted session")]
    Chat {
        #[arg(long, help = "Inject relevant personal-wiki context before each turn")]
        recall: bool,
        #[arg(long = "message", help = "Message to send; repeat for scripted chats")]
        messages: Vec<String>,
    },
    #[command(about = "create a proposal-first memory note")]
    Remember {
        #[arg(long, help = "Memory note title")]
        title: String,
        #[arg(long, help = "Memory note body")]
        body: String,
        #[arg(long = "tag", help = "Tag to attach to the note proposal")]
        tags: Vec<String>,
    },
    #[command(about = "inspect persisted agent sessions")]
    Sessions {
        #[command(subcommand)]
        command: SessionsCommand,
    },
    #[command(about = "extract durable knowledge from sessions")]
    Learn {
        #[command(subcommand)]
        command: LearnCommand,
    },
    #[command(about = "inspect tool registry and approval decisions")]
    Tools {
        #[command(subcommand)]
        command: ToolsCommand,
    },
    #[command(about = "manage and run reusable skills")]
    Skills {
        #[command(subcommand)]
        command: SkillsCommand,
    },
    #[command(about = "run a supervisor with bounded worker agents")]
    Run {
        #[arg(long, help = "Use supervisor/worker multi-agent execution")]
        agents: bool,
        #[arg(help = "Objective for the multi-agent run")]
        objective: String,
        #[arg(long = "worker", help = "Worker profile to include")]
        workers: Vec<WorkerProfileArg>,
        #[arg(
            long,
            help = "Inject shared personal-wiki memory into the supervisor and workers"
        )]
        recall: bool,
        #[arg(
            long,
            default_value_t = 4,
            help = "Maximum workers allowed for this run"
        )]
        max_workers: usize,
        #[arg(long, default_value_t = 90, help = "Per-worker timeout in seconds")]
        timeout_seconds: u64,
    },
    #[command(about = "check whether the CLI can start")]
    Health,
}

#[derive(Debug, Subcommand)]
enum SessionsCommand {
    #[command(about = "list recent sessions")]
    List {
        #[arg(long, default_value_t = 20, help = "Maximum sessions to print")]
        limit: u16,
    },
    #[command(about = "search prior session turns")]
    Search {
        #[arg(help = "Search query")]
        query: String,
        #[arg(long, default_value_t = 20, help = "Maximum search hits to print")]
        limit: u16,
    },
}

#[derive(Debug, Subcommand)]
enum LearnCommand {
    #[command(about = "extract durable knowledge from one persisted session")]
    Session {
        #[arg(help = "Session id to learn from")]
        session_id: String,
    },
    #[command(about = "list pending learning proposals")]
    Review {
        #[arg(long, default_value_t = 20, help = "Maximum proposals to print")]
        limit: u16,
    },
}

#[derive(Debug, Subcommand)]
enum ToolsCommand {
    #[command(about = "list registered tools")]
    List,
    #[command(about = "evaluate whether a tool call would run")]
    Check {
        #[arg(help = "Registered tool name")]
        name: String,
        #[arg(long, default_value_t = PermissionModeArg::AutoRead, help = "Permission mode to evaluate with")]
        mode: PermissionModeArg,
        #[arg(long = "allow", help = "Tool name to auto-approve")]
        allow: Vec<String>,
        #[arg(long = "deny", help = "Tool name to block")]
        deny: Vec<String>,
    },
}

#[derive(Debug, Subcommand)]
enum SkillsCommand {
    #[command(about = "list persisted skills")]
    List {
        #[arg(long, help = "Filter by skill status")]
        status: Option<SkillStatusArg>,
    },
    #[command(about = "propose a skill for later enablement")]
    Propose {
        #[arg(long, help = "Skill name")]
        name: String,
        #[arg(long, help = "Text trigger that activates the skill")]
        trigger: String,
        #[arg(long, help = "Skill instructions")]
        instruction: String,
        #[arg(long = "tool", help = "Required tool name")]
        tools: Vec<String>,
        #[arg(long = "safety-note", help = "Safety note")]
        safety_notes: Vec<String>,
        #[arg(long = "example", help = "Usage example")]
        examples: Vec<String>,
    },
    #[command(about = "enable a proposed or disabled skill")]
    Enable {
        #[arg(help = "Skill id")]
        id: String,
    },
    #[command(about = "run one enabled skill directly")]
    Run {
        #[arg(help = "Skill id")]
        id: String,
        #[arg(help = "Input for the skill")]
        input: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SkillStatusArg {
    Proposed,
    Enabled,
    Disabled,
}

impl From<SkillStatusArg> for SkillStatus {
    fn from(value: SkillStatusArg) -> Self {
        match value {
            SkillStatusArg::Proposed => Self::Proposed,
            SkillStatusArg::Enabled => Self::Enabled,
            SkillStatusArg::Disabled => Self::Disabled,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PermissionModeArg {
    Ask,
    AutoRead,
    Plan,
    Bypass,
}

impl fmt::Display for PermissionModeArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Ask => "ask",
            Self::AutoRead => "auto-read",
            Self::Plan => "plan",
            Self::Bypass => "bypass",
        })
    }
}

impl From<PermissionModeArg> for PermissionMode {
    fn from(value: PermissionModeArg) -> Self {
        match value {
            PermissionModeArg::Ask => Self::Ask,
            PermissionModeArg::AutoRead => Self::AutoRead,
            PermissionModeArg::Plan => Self::Plan,
            PermissionModeArg::Bypass => Self::Bypass,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum WorkerProfileArg {
    Planner,
    Researcher,
    Coder,
    Reviewer,
}

impl From<WorkerProfileArg> for AgentProfile {
    fn from(value: WorkerProfileArg) -> Self {
        match value {
            WorkerProfileArg::Planner => Self::Planner,
            WorkerProfileArg::Researcher => Self::Researcher,
            WorkerProfileArg::Coder => Self::Coder,
            WorkerProfileArg::Reviewer => Self::Reviewer,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Ask {
            prompt,
            recall,
            session_id,
            system_message,
        } => handle_ask(prompt, recall, session_id, system_message).await?,
        Command::Chat { recall, messages } => handle_chat(recall, messages).await?,
        Command::Remember { title, body, tags } => handle_remember(title, body, tags).await?,
        Command::Sessions { command } => handle_sessions(command)?,
        Command::Learn { command } => handle_learn(command).await?,
        Command::Tools { command } => handle_tools(command),
        Command::Skills { command } => handle_skills(command).await?,
        Command::Run {
            agents,
            objective,
            workers,
            recall,
            max_workers,
            timeout_seconds,
        } => {
            handle_run(
                agents,
                objective,
                workers,
                recall,
                max_workers,
                timeout_seconds,
            )
            .await?;
        }
        Command::Health => {
            println!("ok");
        }
    }

    Ok(())
}

async fn handle_ask(
    prompt: String,
    recall: bool,
    session_id: Option<String>,
    system_message: Option<String>,
) -> Result<()> {
    let config = AppConfig::from_env()?;
    let sessions = SessionStore::open(&config.state_dir)?;
    let skills = SkillStore::open(&config.state_dir)?;
    let agent = Agent::with_stores(config, sessions, skills);
    let output = agent
        .ask(AskRequest {
            prompt,
            recall,
            session_id,
            system_message,
        })
        .await?;
    println!("{}", output.answer);
    Ok(())
}

async fn handle_chat(recall: bool, messages: Vec<String>) -> Result<()> {
    let config = AppConfig::from_env()?;
    let sessions = SessionStore::open(&config.state_dir)?;
    let skills = SkillStore::open(&config.state_dir)?;
    let agent = Agent::with_stores(config, sessions, skills);

    if messages.is_empty() {
        run_interactive_chat(&agent, recall).await
    } else {
        let output = agent.chat(ChatSessionRequest { messages, recall }).await?;
        println!("session\t{}", output.session_id);
        for answer in output.answers {
            println!("{answer}");
        }
        Ok(())
    }
}

async fn run_interactive_chat(agent: &Agent, recall: bool) -> Result<()> {
    let mut chat = None;
    let mut input = String::new();

    loop {
        print!("you> ");
        io::stdout().flush()?;
        input.clear();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }

        let message = input.trim();
        if message.eq_ignore_ascii_case("/exit") || message.eq_ignore_ascii_case("/quit") {
            break;
        }
        if !message.is_empty() {
            if chat.is_none() {
                let started = agent.start_chat(message, recall)?;
                println!("session\t{}", started.id());
                chat = Some(started);
            }
            let Some(chat) = &mut chat else {
                continue;
            };
            let answer = agent.send_chat_turn(chat, message.to_string()).await?;
            println!("assistant> {answer}");
        }
    }

    Ok(())
}

async fn handle_remember(title: String, body: String, tags: Vec<String>) -> Result<()> {
    let config = AppConfig::from_env()?;
    let agent = Agent::new(config);
    let proposal_id = agent.remember(title, body, tags).await?;
    println!("created proposal {proposal_id}");
    Ok(())
}

fn handle_sessions(command: SessionsCommand) -> Result<()> {
    let state_dir = AppConfig::state_dir_from_env()?;
    let sessions = SessionStore::open(&state_dir)?;

    match command {
        SessionsCommand::List { limit } => {
            for session in sessions.list_sessions(limit)? {
                println!(
                    "{}\t{}\t{}\t{} turns",
                    session.updated_at, session.id, session.title, session.turn_count
                );
            }
        }
        SessionsCommand::Search { query, limit } => {
            for hit in sessions.search(&query, limit)? {
                println!(
                    "{}\t{}\t{}\t{}",
                    hit.session_id,
                    hit.role,
                    hit.title,
                    hit.content.replace('\n', " ")
                );
            }
        }
    }

    Ok(())
}

async fn handle_learn(command: LearnCommand) -> Result<()> {
    match command {
        LearnCommand::Session { session_id } => {
            let config = AppConfig::from_env()?;
            let sessions = SessionStore::open(&config.state_dir)?;
            let learning = LearningEngine::new(config, sessions);
            match learning
                .learn_session(LearnSessionRequest { session_id })
                .await?
            {
                LearnSessionOutput::ProposalCreated { proposal_id } => {
                    println!("created learning proposal {proposal_id}");
                }
                LearnSessionOutput::NoDurableKnowledge => {
                    println!("no durable knowledge");
                }
            }
        }
        LearnCommand::Review { limit } => {
            let wiki = WikiClient::new(AppConfig::wiki_base_url_from_env()?);
            for proposal in pending_learning_proposals(&wiki, limit).await? {
                println!(
                    "{}\t{}\t{}",
                    proposal.id,
                    proposal.title,
                    proposal.tags.join(",")
                );
            }
        }
    }

    Ok(())
}

fn handle_tools(command: ToolsCommand) {
    let registry = ToolRegistry::built_in();
    match command {
        ToolsCommand::List => {
            let policy = ApprovalPolicy::default();
            for tool in registry.list() {
                let decision = policy.evaluate(&registry, tool.name);
                println!(
                    "{}\t{}\t{}\t{}",
                    tool.name,
                    tool.access,
                    decision.as_str(),
                    tool.description
                );
            }
        }
        ToolsCommand::Check {
            name,
            mode,
            allow,
            deny,
        } => {
            let policy = ApprovalPolicy::new(mode.into())
                .with_allowed(allow)
                .with_denied(deny);
            println!("{}", policy.evaluate(&registry, &name));
        }
    }
}

async fn handle_skills(command: SkillsCommand) -> Result<()> {
    match command {
        SkillsCommand::List { status } => handle_skills_list(status),
        SkillsCommand::Propose {
            name,
            trigger,
            instruction,
            tools,
            safety_notes,
            examples,
        } => handle_skills_propose(name, trigger, instruction, tools, safety_notes, examples),
        SkillsCommand::Enable { id } => handle_skills_enable(&id),
        SkillsCommand::Run { id, input } => handle_skills_run(&id, input).await,
    }
}

fn handle_skills_list(status: Option<SkillStatusArg>) -> Result<()> {
    let state_dir = AppConfig::state_dir_from_env()?;
    let skills = SkillStore::open(&state_dir)?;
    for skill in skills.list(status.map(Into::into))? {
        println!(
            "{}\t{}\t{}\t{}",
            skill.updated_at, skill.id, skill.status, skill.name
        );
    }
    Ok(())
}

fn handle_skills_propose(
    name: String,
    trigger: String,
    instruction: String,
    tools: Vec<String>,
    safety_notes: Vec<String>,
    examples: Vec<String>,
) -> Result<()> {
    let state_dir = AppConfig::state_dir_from_env()?;
    let skills = SkillStore::open(&state_dir)?;
    let skill = skills.propose(&SkillDraft {
        name,
        trigger,
        instructions: instruction,
        required_tools: tools,
        safety_notes,
        examples,
    })?;
    println!("proposed skill {}", skill.id);
    Ok(())
}

fn handle_skills_enable(id: &str) -> Result<()> {
    let state_dir = AppConfig::state_dir_from_env()?;
    let skills = SkillStore::open(&state_dir)?;
    let skill = skills.enable(id)?;
    println!("enabled skill {}", skill.id);
    Ok(())
}

async fn handle_skills_run(id: &str, input: String) -> Result<()> {
    let config = AppConfig::from_env()?;
    let skills = SkillStore::open(&config.state_dir)?;
    let skill = skills
        .get(id)?
        .ok_or_else(|| anyhow::anyhow!("skill {id} not found"))?;
    if skill.status != SkillStatus::Enabled {
        anyhow::bail!("skill {id} is not enabled");
    }

    let llm = LlmClient::new(&config.pengepul_base_url, config.pengepul_api_key);
    let output = llm
        .chat(ChatRequest {
            model: config.model,
            messages: vec![
                Message::system(format!(
                    "Run this enabled Loka skill:\n\n{}",
                    skill.prompt_block()
                )),
                Message::user(input),
            ],
        })
        .await?;
    println!("{}", output.content);
    Ok(())
}

async fn handle_run(
    agents: bool,
    objective: String,
    workers: Vec<WorkerProfileArg>,
    recall: bool,
    max_workers: usize,
    timeout_seconds: u64,
) -> Result<()> {
    if !agents {
        anyhow::bail!("run currently requires --agents");
    }

    let config = AppConfig::from_env()?;
    let sessions = SessionStore::open(&config.state_dir)?;
    let tasks = TaskGraphStore::open(&config.state_dir)?;
    let runtime = MultiAgentRuntime::new(config, sessions, tasks);
    let worker_specs = default_worker_specs(&objective, workers, timeout_seconds);
    let output = runtime
        .run(MultiAgentRunRequest {
            objective,
            recall,
            max_workers,
            workers: worker_specs,
        })
        .await?;

    println!("run\t{}", output.run_id);
    println!("supervisor_session\t{}", output.supervisor_session_id);
    println!("tokens\t{}", output.total_tokens);
    for worker in output.workers {
        let summary = match worker.error {
            Some(error) => format!("error: {}", error.replace('\n', " ")),
            None => worker.summary.replace('\n', " "),
        };
        println!(
            "worker\t{}\t{}\t{}\t{} tokens\t{}",
            worker.profile, worker.status, worker.session_id, worker.tokens_used, summary
        );
    }
    println!("\nsynthesis\n{}", output.synthesis);

    Ok(())
}

fn default_worker_specs(
    objective: &str,
    requested: Vec<WorkerProfileArg>,
    timeout_seconds: u64,
) -> Vec<WorkerSpec> {
    let profiles = if requested.is_empty() {
        vec![
            AgentProfile::Planner,
            AgentProfile::Researcher,
            AgentProfile::Coder,
            AgentProfile::Reviewer,
        ]
    } else {
        requested.into_iter().map(Into::into).collect()
    };

    profiles
        .into_iter()
        .map(|profile| default_worker_spec(profile, objective, timeout_seconds))
        .collect()
}

fn default_worker_spec(profile: AgentProfile, objective: &str, timeout_seconds: u64) -> WorkerSpec {
    let objective = objective.trim();
    match profile {
        AgentProfile::Supervisor => WorkerSpec {
            profile,
            objective: objective.to_string(),
            output_format: "supervisor synthesis notes".to_string(),
            tools_allowed: vec!["session_search".to_string(), "wiki_rag".to_string()],
            max_iterations: 3,
            max_tokens: 2_000,
            timeout_seconds,
            justification: "supervisor profile is reserved for synthesis".to_string(),
        },
        AgentProfile::Planner => WorkerSpec {
            profile,
            objective: format!("Create an execution plan for: {objective}"),
            output_format: "concise plan with sequencing, dependencies, and risks".to_string(),
            tools_allowed: vec!["session_search".to_string(), "wiki_rag".to_string()],
            max_iterations: 4,
            max_tokens: 3_000,
            timeout_seconds,
            justification: "planner decomposes the objective before implementation".to_string(),
        },
        AgentProfile::Researcher => WorkerSpec {
            profile,
            objective: format!(
                "Find relevant context, unknowns, and external constraints for: {objective}"
            ),
            output_format: "research findings with sources or explicit uncertainty".to_string(),
            tools_allowed: vec![
                "wiki_rag".to_string(),
                "session_search".to_string(),
                "read_file".to_string(),
                "search_files".to_string(),
            ],
            max_iterations: 4,
            max_tokens: 3_000,
            timeout_seconds,
            justification: "researcher reduces uncertainty before code changes".to_string(),
        },
        AgentProfile::Coder => WorkerSpec {
            profile,
            objective: format!("Identify implementation changes for: {objective}"),
            output_format: "code-level plan with files, tests, and failure modes".to_string(),
            tools_allowed: vec![
                "read_file".to_string(),
                "search_files".to_string(),
                "git_status".to_string(),
            ],
            max_iterations: 5,
            max_tokens: 3_500,
            timeout_seconds,
            justification: "coder maps the objective into concrete code changes".to_string(),
        },
        AgentProfile::Reviewer => WorkerSpec {
            profile,
            objective: format!(
                "Review the approach for correctness, safety, performance, and tests: {objective}"
            ),
            output_format: "review findings ordered by severity".to_string(),
            tools_allowed: vec![
                "read_file".to_string(),
                "search_files".to_string(),
                "git_status".to_string(),
                "session_search".to_string(),
            ],
            max_iterations: 4,
            max_tokens: 3_000,
            timeout_seconds,
            justification: "reviewer catches correctness, safety, and performance gaps".to_string(),
        },
    }
}
