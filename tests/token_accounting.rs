use loka::messages::{Message, Role};
use loka::prompt::{PromptBuilder, PromptInput};
use loka::session::SessionStore;
use loka::tokens::{TokenScope, estimate_messages_tokens, estimate_text_tokens};

#[test]
fn token_estimates_are_deterministic_and_nonzero_for_content() {
    assert_eq!(estimate_text_tokens(""), 0);
    assert_eq!(estimate_text_tokens("1234"), 1);
    assert_eq!(estimate_text_tokens("12345"), 2);

    let messages = vec![
        Message::system("stable guidance"),
        Message::user("what next"),
    ];
    assert_eq!(
        estimate_messages_tokens(&messages),
        estimate_messages_tokens(&messages)
    );
    assert!(estimate_messages_tokens(&messages) > estimate_text_tokens("what next"));
}

#[test]
fn prompt_builder_accounts_for_prompt_layers() {
    let input = PromptInput {
        agent_id: "loka".to_string(),
        model: "gpt-5.5".to_string(),
        model_protocol: loka::config::ModelProtocol::OpenAiCompatible,
        session_id: Some("session-1".to_string()),
        system_message: Some("caller guidance".to_string()),
        memory_markdown: Some("remember this".to_string()),
        context_files: vec![],
        date: "2026-05-29".to_string(),
    };

    let prompt = PromptBuilder::new().build(&input);

    assert!(prompt.token_accounting.stable_tokens > 0);
    assert!(prompt.token_accounting.context_tokens > 0);
    assert!(prompt.token_accounting.volatile_tokens > 0);
    assert_eq!(
        prompt.token_accounting.total_tokens,
        prompt.token_accounting.stable_tokens
            + prompt.token_accounting.context_tokens
            + prompt.token_accounting.volatile_tokens
    );
}

#[test]
fn session_store_records_session_scope_turn_tokens() {
    let store = SessionStore::in_memory().expect("store");
    let session_id = store.create_session("tokens").expect("session");

    store
        .append_turn(&session_id, Role::User, "count this turn")
        .expect("turn");

    let records = store
        .session_token_usage_records(&session_id)
        .expect("token records");
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].scope, TokenScope::Session);
    assert_eq!(records[0].source, "turn:user");
    assert!(records[0].total_tokens > 0);

    let summary = store.session_token_usage(&session_id).expect("summary");
    assert_eq!(summary.total_tokens, records[0].total_tokens);
}
