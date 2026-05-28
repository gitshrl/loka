use loka_agent::messages::Role;
use loka_agent::session::{SessionStore, ToolCallStatus};
use loka_agent::tui::{TuiApp, TuiPane, render_tui_frame};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use serde_json::json;

#[test]
fn tui_app_exposes_required_product_panes() {
    let app = TuiApp::empty();
    assert_eq!(
        app.panes(),
        &[
            TuiPane::Conversation,
            TuiPane::ToolCalls,
            TuiPane::MemoryContext,
            TuiPane::SessionSearch,
            TuiPane::ApprovalQueue,
        ]
    );
}

#[test]
fn tui_app_loads_session_search_without_provider_credentials() {
    let sessions = SessionStore::in_memory().expect("sessions");
    let session_id = sessions
        .create_session("runtime planning")
        .expect("session");
    sessions
        .append_turn(&session_id, Role::User, "compare docker and ssh runtime")
        .expect("turn");

    let app = TuiApp::from_sessions(&sessions, "docker", 10).expect("tui app");

    assert_eq!(app.search_hits().len(), 1);
    assert_eq!(app.search_hits()[0].session_id, session_id);
    assert_eq!(app.sessions().len(), 1);
}

#[test]
fn tui_app_loads_latest_session_tool_calls() {
    let sessions = SessionStore::in_memory().expect("sessions");
    let session_id = sessions.create_session("tool calls").expect("session");
    let call_id = sessions
        .record_tool_call_started(
            &session_id,
            "session_search",
            &json!({ "query": "runtime" }),
        )
        .expect("tool call");
    sessions
        .record_tool_call_completed(&call_id, &json!({ "hits": [] }))
        .expect("complete");

    let app = TuiApp::from_sessions(&sessions, "", 10).expect("tui app");

    assert_eq!(app.tool_calls().len(), 1);
    assert!(app.tool_calls()[0].contains(ToolCallStatus::Completed.as_str()));
    assert!(app.tool_calls()[0].contains("session_search"));
}

#[test]
fn tui_frame_renders_required_pane_titles() {
    let app = TuiApp::empty();
    let backend = TestBackend::new(120, 40);
    let mut terminal = Terminal::new(backend).expect("terminal");

    terminal
        .draw(|frame| render_tui_frame(frame, &app))
        .expect("draw");

    let buffer = terminal.backend().buffer();
    let rendered = buffer
        .content()
        .iter()
        .map(ratatui::buffer::Cell::symbol)
        .collect::<String>();

    assert!(rendered.contains("Conversation"));
    assert!(rendered.contains("Tool Calls"));
    assert!(rendered.contains("Memory Context"));
    assert!(rendered.contains("Session Search"));
    assert!(rendered.contains("Approval Queue"));
}
