use loka::messages::Role;
use loka::session::SessionStore;

#[test]
fn session_store_persists_and_searches_turns() {
    let store = SessionStore::in_memory().expect("store");
    let session_id = store
        .create_session("Investigate runtime backends")
        .expect("session");

    store
        .append_turn(
            &session_id,
            Role::User,
            "compare Docker and SSH runtime execution",
        )
        .expect("user turn");
    store
        .append_turn(&session_id, Role::Assistant, "Use Docker first, SSH next.")
        .expect("assistant turn");

    let sessions = store.list_sessions(10).expect("sessions");
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].turn_count, 2);

    let turns = store.session_turns(&session_id).expect("turns");
    assert_eq!(turns.len(), 2);
    assert_eq!(turns[0].role, Role::User);
    assert_eq!(turns[1].role, Role::Assistant);

    let hits = store.search("Docker", 10).expect("search");
    assert_eq!(hits.len(), 2);
    assert!(hits.iter().any(|hit| hit.role == Role::User));
    assert!(hits.iter().any(|hit| hit.role == Role::Assistant));
}

#[test]
fn session_store_returns_no_hits_for_blank_search() {
    let store = SessionStore::in_memory().expect("store");
    let session_id = store.create_session("Blank query").expect("session");
    store
        .append_turn(&session_id, Role::User, "content")
        .expect("turn");

    let hits = store.search("   ", 10).expect("search");
    assert!(hits.is_empty());
}
