use anyhow::{Context, Result, anyhow};
use reqwest::Url;
use serde::Deserialize;
use std::fmt;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelProtocol {
    OpenAiCompatible,
    AnthropicCompatible,
}

impl ModelProtocol {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAiCompatible => "openai-compatible",
            Self::AnthropicCompatible => "anthropic-compatible",
        }
    }
}

impl fmt::Display for ModelProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ModelProtocol {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim() {
            "openai-compatible" => Ok(Self::OpenAiCompatible),
            "anthropic-compatible" => Ok(Self::AnthropicCompatible),
            other => Err(anyhow!(
                "LOKA_MODEL_PROTOCOL must be openai-compatible or anthropic-compatible, got {other}"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLifecycleMode {
    Off,
    Strict,
}

impl MemoryLifecycleMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Strict => "strict",
        }
    }
}

impl fmt::Display for MemoryLifecycleMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for MemoryLifecycleMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim() {
            "off" => Ok(Self::Off),
            "strict" => Ok(Self::Strict),
            other => Err(anyhow!(
                "LOKA_MEMORY_LIFECYCLE must be off or strict, got {other}"
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppConfig {
    pub model_base_url: String,
    pub model_api_key: String,
    pub memory_base_url: String,
    pub model: String,
    pub agent_id: String,
    pub model_protocol: ModelProtocol,
    pub memory_lifecycle: MemoryLifecycleMode,
    pub working_dir: PathBuf,
    pub state_dir: PathBuf,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileConfig {
    model_base_url: Option<String>,
    model_api_key: Option<String>,
    memory_base_url: Option<String>,
    model: Option<String>,
    agent_id: Option<String>,
    model_protocol: Option<String>,
    memory_lifecycle: Option<String>,
    working_dir: Option<String>,
    state_dir: Option<String>,
    telegram_bot_token: Option<String>,
}

impl AppConfig {
    /// Loads the Telegram bot token from environment and `~/.loka/config.toml` without requiring
    /// provider credentials.
    ///
    /// # Errors
    ///
    /// Returns an error when the token is missing or the config file cannot be read or parsed.
    pub fn telegram_bot_token_from_env() -> Result<String> {
        Self::telegram_bot_token_from_env_map(|key| std::env::var(key).ok())
    }

    /// Loads the Telegram bot token from a caller-provided key/value source and optional config file.
    ///
    /// # Errors
    ///
    /// Returns an error when the token is missing or the config file cannot be read or parsed.
    pub fn telegram_bot_token_from_env_map<F>(get: F) -> Result<String>
    where
        F: Fn(&str) -> Option<String>,
    {
        let file = load_file_config(&config_file_path(&get))?;
        get_required(
            &get,
            "LOKA_TELEGRAM_BOT_TOKEN",
            file.telegram_bot_token.as_deref(),
        )
    }

    /// Loads the memory base URL from environment and `~/.loka/config.toml` without requiring
    /// provider credentials.
    ///
    /// # Errors
    ///
    /// Returns an error when the config file exists but cannot be read or parsed, or when the
    /// memory URL is invalid.
    pub fn memory_base_url_from_env() -> Result<String> {
        Self::memory_base_url_from_env_map(|key| std::env::var(key).ok())
    }

    /// Loads the memory base URL from a caller-provided key/value source and optional config file.
    ///
    /// # Errors
    ///
    /// Returns an error when the config file exists but cannot be read or parsed, or when the
    /// memory URL is invalid.
    pub fn memory_base_url_from_env_map<F>(get: F) -> Result<String>
    where
        F: Fn(&str) -> Option<String>,
    {
        let file = load_file_config(&config_file_path(&get))?;
        get_optional_url(
            &get,
            "LOKA_MEMORY_BASE_URL",
            file.memory_base_url.as_deref(),
            "http://127.0.0.1:4321",
        )
    }

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
        let model_api_key =
            get_required(&get, "LOKA_MODEL_API_KEY", file.model_api_key.as_deref())?;
        let working_dir = get_working_dir(&get, file.working_dir.as_deref())?;

        Ok(Self {
            model_base_url: get_optional_url(
                &get,
                "LOKA_MODEL_BASE_URL",
                file.model_base_url.as_deref(),
                "http://127.0.0.1:8317",
            )?,
            model_api_key,
            memory_base_url: get_optional_url(
                &get,
                "LOKA_MEMORY_BASE_URL",
                file.memory_base_url.as_deref(),
                "http://127.0.0.1:4321",
            )?,
            model: get_optional(&get, "LOKA_MODEL", file.model.as_deref(), "gpt-5.5"),
            agent_id: get_optional(&get, "LOKA_AGENT_ID", file.agent_id.as_deref(), "loka"),
            model_protocol: get_model_protocol(&get, file.model_protocol.as_deref())?,
            memory_lifecycle: get_memory_lifecycle(&get, file.memory_lifecycle.as_deref())?,
            working_dir,
            state_dir: get_state_dir(&get, &file),
        })
    }
}

fn get_model_protocol<F>(get: &F, file_value: Option<&str>) -> Result<ModelProtocol>
where
    F: Fn(&str) -> Option<String>,
{
    get_optional(
        get,
        "LOKA_MODEL_PROTOCOL",
        file_value,
        ModelProtocol::OpenAiCompatible.as_str(),
    )
    .parse()
}

fn get_memory_lifecycle<F>(get: &F, file_value: Option<&str>) -> Result<MemoryLifecycleMode>
where
    F: Fn(&str) -> Option<String>,
{
    get_optional(
        get,
        "LOKA_MEMORY_LIFECYCLE",
        file_value,
        MemoryLifecycleMode::Off.as_str(),
    )
    .parse()
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

fn get_working_dir<F>(get: &F, file_value: Option<&str>) -> Result<PathBuf>
where
    F: Fn(&str) -> Option<String>,
{
    let dir = get_value(get, "LOKA_WORKING_DIR")
        .or_else(|| normalize_optional(file_value))
        .map(PathBuf::from)
        .map_or_else(|| default_working_dir(get), Ok)
        .context("read current working directory")?;

    if !dir.is_absolute() {
        return Err(anyhow!("LOKA_WORKING_DIR must be an absolute path"));
    }

    Ok(dir)
}

fn default_working_dir<F>(get: &F) -> Result<PathBuf>
where
    F: Fn(&str) -> Option<String>,
{
    if let Some(home) = get("HOME")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        return Ok(PathBuf::from(home).join(".loka").join("workspace"));
    }

    Ok(std::env::current_dir()?.join(".loka").join("workspace"))
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
