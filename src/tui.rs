use anyhow::Result;
use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use std::io;
use std::time::Duration;

use crate::session::{SearchHit, SessionStore, SessionSummary, ToolCallRecord};

const PANES: [TuiPane; 5] = [
    TuiPane::Conversation,
    TuiPane::ToolCalls,
    TuiPane::MemoryContext,
    TuiPane::SessionSearch,
    TuiPane::ApprovalQueue,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiPane {
    Conversation,
    ToolCalls,
    MemoryContext,
    SessionSearch,
    ApprovalQueue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiApp {
    selected_pane: TuiPane,
    sessions: Vec<SessionSummary>,
    search_hits: Vec<SearchHit>,
    memory_context: String,
    tool_calls: Vec<String>,
    approvals: Vec<String>,
}

impl TuiApp {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            selected_pane: TuiPane::Conversation,
            sessions: Vec::new(),
            search_hits: Vec::new(),
            memory_context: String::new(),
            tool_calls: Vec::new(),
            approvals: Vec::new(),
        }
    }

    /// Builds TUI state from the local session store without requiring provider credentials.
    ///
    /// # Errors
    ///
    /// Returns an error when the session store rejects the list or search query.
    pub fn from_sessions(sessions: &SessionStore, search: &str, limit: u16) -> Result<Self> {
        let search_hits = if search.trim().is_empty() {
            Vec::new()
        } else {
            sessions.search(search, limit)?
        };
        let recent_sessions = sessions.list_sessions(20)?;
        let tool_calls = match recent_sessions.first() {
            Some(session) => sessions
                .session_tool_calls(&session.id)?
                .into_iter()
                .map(format_tool_call)
                .collect(),
            None => Vec::new(),
        };

        Ok(Self {
            sessions: recent_sessions,
            search_hits,
            tool_calls,
            ..Self::empty()
        })
    }

    #[must_use]
    pub const fn panes(&self) -> &'static [TuiPane] {
        &PANES
    }

    #[must_use]
    pub fn sessions(&self) -> &[SessionSummary] {
        &self.sessions
    }

    #[must_use]
    pub fn search_hits(&self) -> &[SearchHit] {
        &self.search_hits
    }

    #[must_use]
    pub fn tool_calls(&self) -> &[String] {
        &self.tool_calls
    }

    fn select_next_pane(&mut self) {
        let current = PANES
            .iter()
            .position(|pane| *pane == self.selected_pane)
            .unwrap_or(0);
        self.selected_pane = PANES[(current + 1) % PANES.len()];
    }
}

fn format_tool_call(call: ToolCallRecord) -> String {
    let finished_at = call
        .completed_at
        .as_deref()
        .unwrap_or(call.created_at.as_str());
    match call.error {
        Some(error) => format!(
            "{}  {}  {}  {}",
            finished_at,
            call.status.as_str(),
            call.name,
            error.replace('\n', " ")
        ),
        None => format!("{}  {}  {}", finished_at, call.status.as_str(), call.name),
    }
}

/// Runs the terminal UI until the user exits with `q` or `Esc`.
///
/// # Errors
///
/// Returns an error when terminal setup, rendering, or event polling fails.
pub fn run_tui(app: &mut TuiApp) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_tui_loop(&mut terminal, app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

pub fn render_tui_frame(frame: &mut Frame<'_>, app: &TuiApp) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(42),
            Constraint::Percentage(30),
            Constraint::Percentage(28),
        ])
        .split(frame.area());
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(64), Constraint::Percentage(36)])
        .split(rows[0]);
    let middle = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[1]);

    render_conversation(frame, top[0], app);
    render_tool_calls(frame, top[1], app);
    render_memory_context(frame, middle[0], app);
    render_session_search(frame, middle[1], app);
    render_approval_queue(frame, rows[2], app);
}

fn run_tui_loop<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut TuiApp,
) -> Result<()>
where
    B::Error: std::error::Error + Send + Sync + 'static,
{
    loop {
        terminal.draw(|frame| render_tui_frame(frame, app))?;
        if event::poll(Duration::from_millis(200))? {
            match event::read()? {
                Event::Key(key) if matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) => break,
                Event::Key(key) if key.code == KeyCode::Tab => app.select_next_pane(),
                _ => {}
            }
        }
    }
    Ok(())
}

fn render_conversation(frame: &mut Frame<'_>, area: Rect, app: &TuiApp) {
    let items = app
        .sessions
        .iter()
        .map(|session| {
            ListItem::new(format!(
                "{}  {}  {} turns",
                session.updated_at, session.title, session.turn_count
            ))
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        List::new(items).block(block("Conversation", app, TuiPane::Conversation)),
        area,
    );
}

fn render_tool_calls(frame: &mut Frame<'_>, area: Rect, app: &TuiApp) {
    let text = if app.tool_calls.is_empty() {
        "no tool calls".to_string()
    } else {
        app.tool_calls.join("\n")
    };
    frame.render_widget(
        Paragraph::new(text)
            .block(block("Tool Calls", app, TuiPane::ToolCalls))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn render_memory_context(frame: &mut Frame<'_>, area: Rect, app: &TuiApp) {
    let text = if app.memory_context.trim().is_empty() {
        "no memory context loaded"
    } else {
        app.memory_context.as_str()
    };
    frame.render_widget(
        Paragraph::new(text)
            .block(block("Memory Context", app, TuiPane::MemoryContext))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn render_session_search(frame: &mut Frame<'_>, area: Rect, app: &TuiApp) {
    let items = app
        .search_hits
        .iter()
        .map(|hit| {
            ListItem::new(format!(
                "{}  {}  {}",
                hit.title,
                hit.role,
                hit.content.replace('\n', " ")
            ))
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        List::new(items).block(block("Session Search", app, TuiPane::SessionSearch)),
        area,
    );
}

fn render_approval_queue(frame: &mut Frame<'_>, area: Rect, app: &TuiApp) {
    let lines = if app.approvals.is_empty() {
        vec![Line::from("no pending approvals")]
    } else {
        app.approvals
            .iter()
            .map(|item| Line::from(item.as_str()))
            .collect()
    };
    frame.render_widget(
        Paragraph::new(lines)
            .block(block("Approval Queue", app, TuiPane::ApprovalQueue))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn block<'a>(title: &'a str, app: &TuiApp, pane: TuiPane) -> Block<'a> {
    let style = if app.selected_pane == pane {
        Style::default().add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };
    Block::default()
        .title(title)
        .borders(Borders::ALL)
        .style(style)
}
