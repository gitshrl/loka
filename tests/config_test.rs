use loka_agent::config::AppConfig;
use std::fs;
use std::path::PathBuf;

#[test]
fn config_uses_development_service_defaults() {
    let config = AppConfig::from_env_map(|key| match key {
        "LOKA_PENGEPUL_API_KEY" => Some("sk-test".to_string()),
        _ => None,
    })
    .expect("config should load");

    assert_eq!(config.pengepul_base_url, "http://127.0.0.1:8317");
    assert_eq!(config.wiki_base_url, "http://127.0.0.1:4321");
    assert_eq!(config.model, "gpt-5");
    assert_eq!(config.agent_id, "loka-agent");
    assert_eq!(config.pengepul_api_key, "sk-test");
    assert_eq!(config.state_dir, PathBuf::from(".loka"));
}

#[test]
fn config_requires_pengepul_api_key() {
    let error = AppConfig::from_env_map(|_| None).expect_err("missing key should fail");
    assert!(error.to_string().contains("LOKA_PENGEPUL_API_KEY"));
}

#[test]
fn config_loads_from_home_loka_config_file() {
    let home = tempfile::tempdir().expect("home");
    let config_dir = home.path().join(".loka");
    fs::create_dir(&config_dir).expect("config dir");
    fs::write(
        config_dir.join("config.toml"),
        r#"
pengepul_base_url = "http://127.0.0.1:9001"
pengepul_api_key = "sk-file"
wiki_base_url = "http://127.0.0.1:9002"
model = "gpt-file"
agent_id = "agent-file"
state_dir = "/srv/loka-state"
"#,
    )
    .expect("config file");

    let config = AppConfig::from_env_map(|key| match key {
        "HOME" => Some(home.path().display().to_string()),
        _ => None,
    })
    .expect("config should load");

    assert_eq!(config.pengepul_base_url, "http://127.0.0.1:9001");
    assert_eq!(config.pengepul_api_key, "sk-file");
    assert_eq!(config.wiki_base_url, "http://127.0.0.1:9002");
    assert_eq!(config.model, "gpt-file");
    assert_eq!(config.agent_id, "agent-file");
    assert_eq!(config.state_dir, PathBuf::from("/srv/loka-state"));
}

#[test]
fn env_overrides_home_loka_config_file() {
    let home = tempfile::tempdir().expect("home");
    let config_dir = home.path().join(".loka");
    fs::create_dir(&config_dir).expect("config dir");
    fs::write(
        config_dir.join("config.toml"),
        r#"
pengepul_api_key = "sk-file"
model = "gpt-file"
"#,
    )
    .expect("config file");

    let config = AppConfig::from_env_map(|key| match key {
        "HOME" => Some(home.path().display().to_string()),
        "LOKA_PENGEPUL_API_KEY" => Some("sk-env".to_string()),
        "LOKA_MODEL" => Some("gpt-env".to_string()),
        _ => None,
    })
    .expect("config should load");

    assert_eq!(config.pengepul_api_key, "sk-env");
    assert_eq!(config.model, "gpt-env");
}

#[test]
fn config_rejects_invalid_service_url() {
    let error = AppConfig::from_env_map(|key| match key {
        "LOKA_PENGEPUL_API_KEY" => Some("sk-test".to_string()),
        "LOKA_WIKI_BASE_URL" => Some("file:///tmp/wiki".to_string()),
        _ => None,
    })
    .expect_err("file URL should fail");

    assert!(error.to_string().contains("LOKA_WIKI_BASE_URL"));
}

#[test]
fn config_uses_explicit_state_dir() {
    let config = AppConfig::from_env_map(|key| match key {
        "LOKA_PENGEPUL_API_KEY" => Some("sk-test".to_string()),
        "LOKA_STATE_DIR" => Some("/var/lib/loka".to_string()),
        _ => None,
    })
    .expect("config should load");

    assert_eq!(config.state_dir, PathBuf::from("/var/lib/loka"));
}

#[test]
fn state_dir_can_load_without_provider_credentials() {
    let state_dir = AppConfig::state_dir_from_env_map(|key| match key {
        "HOME" => Some("/home/dev".to_string()),
        _ => None,
    })
    .expect("state dir");

    assert_eq!(state_dir, PathBuf::from("/home/dev/.loka"));
}

#[test]
fn state_dir_ignores_xdg_state_home_by_default() {
    let state_dir = AppConfig::state_dir_from_env_map(|key| match key {
        "HOME" => Some("/home/dev".to_string()),
        "XDG_STATE_HOME" => Some("/srv/state".to_string()),
        _ => None,
    })
    .expect("state dir");

    assert_eq!(state_dir, PathBuf::from("/home/dev/.loka"));
}

#[test]
fn state_dir_loads_from_home_loka_config_without_provider_credentials() {
    let home = tempfile::tempdir().expect("home");
    let config_dir = home.path().join(".loka");
    fs::create_dir(&config_dir).expect("config dir");
    fs::write(
        config_dir.join("config.toml"),
        r#"
state_dir = "/srv/loka-state"
"#,
    )
    .expect("config file");

    let state_dir = AppConfig::state_dir_from_env_map(|key| match key {
        "HOME" => Some(home.path().display().to_string()),
        _ => None,
    })
    .expect("state dir");

    assert_eq!(state_dir, PathBuf::from("/srv/loka-state"));
}
