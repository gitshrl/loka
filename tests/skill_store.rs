use loka::skills::{SkillDraft, SkillStatus, SkillStore};

#[test]
fn skill_store_proposes_lists_enables_and_matches_skills() {
    let store = SkillStore::in_memory().expect("store");
    let skill = store
        .propose(&SkillDraft {
            name: "Rust review".to_string(),
            trigger: "rust review".to_string(),
            instructions: "Review Rust code with strict clippy expectations.".to_string(),
            required_tools: vec!["read_file".to_string(), "search_files".to_string()],
            safety_notes: vec!["Do not execute shell commands without approval.".to_string()],
            examples: vec!["rust review src/main.rs".to_string()],
        })
        .expect("skill proposal");

    assert_eq!(skill.status, SkillStatus::Proposed);
    assert_eq!(store.list(None).expect("list").len(), 1);

    let enabled = store.enable(&skill.id).expect("enable");
    assert_eq!(enabled.status, SkillStatus::Enabled);

    let matches = store
        .enabled_for_prompt("please do a Rust review of this crate")
        .expect("matches");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].id, skill.id);
    assert!(matches[0].prompt_block().contains("strict clippy"));
}

#[test]
fn skill_store_rejects_blank_required_fields() {
    let store = SkillStore::in_memory().expect("store");
    let error = store
        .propose(&SkillDraft {
            name: " ".to_string(),
            trigger: "rust".to_string(),
            instructions: "Do the work.".to_string(),
            required_tools: vec![],
            safety_notes: vec![],
            examples: vec![],
        })
        .expect_err("blank name should fail");

    assert!(error.to_string().contains("skill name"));
}
