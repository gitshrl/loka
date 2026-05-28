use anyhow::{Context, Result, anyhow};
use reqwest::Url;
use serde::Deserialize;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppConfig {
    pub pengepul_base_url: String,
    pub pengepul_api_key: String,
    pub wiki_base_url: String,
    pub model: String,
    pub agent_id: String,
    pub state_dir: PathBuf,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileConfig {
    pengepul_base_url: Option<String>,
    pengepul_api_key: Option<String>,
    wiki_base_url: Option<String>,
    model: Option<String>,
    agent_id: Option<String>,
    state_dir: Option<String>,
}

impl AppConfig {
    /// Loads the state directory from environment and `~/.loka/config.toml` without requiring
    /// provider credentials.
    ///
    /// # Errors
    ///
    /// Returns an error when the config file exists but cannot be read or parsed.
    pub fn state_dir_from_env() -> Result<PathBuf> {
        Self::state_dir_from_env_map(|key| std::env::var(key).ok())
    }

    /// Loads the state directory from a caller-provided key/value source and optional config file.
    ///
    /// # Errors
    ///
    /// Returns an error when the config file exists but cannot be read or parsed.
    pub fn state_dir_from_env_map<F>(get: F) -> Result<PathBuf>
    where
        F: Fn(&str) -> Option<String>,
    {
        let file = load_file_config(&config_file_path(&get))?;
        Ok(get_state_dir(&get, &file))
    }

    /// Loads application configuration from process environment variables.
    ///
    /// # Errors
    ///
    /// Returns an error when required values are missing or service URLs are invalid.
    pub fn from_env() -> Result<Self> {
        Self::from_env_map(|key| std::env::var(key).ok())
    }

    /// Loads application configuration from a caller-provided key/value source.
    ///
    /// # Errors
    ///
    /// Returns an error when required values are missing or service URLs are invalid.
    pub fn from_env_map<F>(get: F) -> Result<Self>
    where
        F: Fn(&str) -> Option<String>,
    {
        let file = load_file_config(&config_file_path(&get))?;
        let pengepul_api_key = get_required(
            &get,
            "LOKA_PENGEPUL_API_KEY",
            file.pengepul_api_key.as_deref(),
        )?;

        Ok(Self {
            pengepul_base_url: get_optional_url(
                &get,
                "LOKA_PENGEPUL_BASE_URL",
                file.pengepul_base_url.as_deref(),
                "http://127.0.0.1:8317",
            )?,
            pengepul_api_key,
            wiki_base_url: get_optional_url(
                &get,
                "LOKA_WIKI_BASE_URL",
                file.wiki_base_url.as_deref(),
                "http://127.0.0.1:4321",
            )?,
            model: get_optional(&get, "LOKA_MODEL", file.model.as_deref(), "gpt-5"),
            agent_id: get_optional(
                &get,
                "LOKA_AGENT_ID",
                file.agent_id.as_deref(),
                "loka-agent",
            ),
            state_dir: get_state_dir(&get, &file),
        })
    }
}

fn get_required<F>(get: &F, key: &str, file_value: Option<&str>) -> Result<String>
where
    F: Fn(&str) -> Option<String>,
{
    get_value(get, key)
        .or_else(|| normalize_optional(file_value))
        .ok_or_else(|| anyhow!("{key} is required"))
}

fn get_optional<F>(get: &F, key: &str, file_value: Option<&str>, default: &str) -> String
where
    F: Fn(&str) -> Option<String>,
{
    get_value(get, key)
        .or_else(|| normalize_optional(file_value))
        .unwrap_or_else(|| default.to_string())
}

fn get_optional_url<F>(
    get: &F,
    key: &str,
    file_value: Option<&str>,
    default: &str,
) -> Result<String>
where
    F: Fn(&str) -> Option<String>,
{
    let value = get_optional(get, key, file_value, default);
    normalize_url(key, &value)
}

fn normalize_url(key: &str, value: &str) -> Result<String> {
    let parsed = Url::parse(value).with_context(|| format!("{key} must be a valid URL"))?;
    match parsed.scheme() {
        "http" | "https" => Ok(value.trim_end_matches('/').to_string()),
        scheme => Err(anyhow!("{key} must use http or https, got {scheme}")),
    }
}

fn get_state_dir<F>(get: &F, file: &FileConfig) -> PathBuf
where
    F: Fn(&str) -> Option<String>,
{
    if let Some(value) = get_value(get, "LOKA_STATE_DIR") {
        return PathBuf::from(value);
    }

    if let Some(value) = normalize_optional(file.state_dir.as_deref()) {
        return PathBuf::from(value);
    }

    loka_dir(get)
}

fn config_file_path<F>(get: &F) -> PathBuf
where
    F: Fn(&str) -> Option<String>,
{
    loka_dir(get).join("config.toml")
}

fn loka_dir<F>(get: &F) -> PathBuf
where
    F: Fn(&str) -> Option<String>,
{
    if let Some(value) = get("HOME")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        return PathBuf::from(value).join(".loka");
    }

    PathBuf::from(".loka")
}

fn load_file_config(path: &Path) -> Result<FileConfig> {
    match fs::read_to_string(path) {
        Ok(content) => toml::from_str(&content)
            .with_context(|| format!("parse config file {}", path.display())),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(FileConfig::default()),
        Err(error) => Err(error).with_context(|| format!("read config file {}", path.display())),
    }
}

fn get_value<F>(get: &F, key: &str) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    get(key).and_then(|value| normalize_optional(Some(&value)))
}

fn normalize_optional(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}
