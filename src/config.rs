use anyhow::{Context, Result, anyhow};
use reqwest::Url;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppConfig {
    pub pengepul_base_url: String,
    pub pengepul_api_key: String,
    pub wiki_base_url: String,
    pub model: String,
    pub agent_id: String,
    pub state_dir: PathBuf,
}

impl AppConfig {
    #[must_use]
    pub fn state_dir_from_env() -> PathBuf {
        Self::state_dir_from_env_map(|key| std::env::var(key).ok())
    }

    #[must_use]
    pub fn state_dir_from_env_map<F>(get: F) -> PathBuf
    where
        F: Fn(&str) -> Option<String>,
    {
        get_state_dir(&get)
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
        let pengepul_api_key = get_required(&get, "LOKA_PENGEPUL_API_KEY")?;

        Ok(Self {
            pengepul_base_url: get_optional_url(
                &get,
                "LOKA_PENGEPUL_BASE_URL",
                "http://127.0.0.1:8317",
            )?,
            pengepul_api_key,
            wiki_base_url: get_optional_url(&get, "LOKA_WIKI_BASE_URL", "http://127.0.0.1:4321")?,
            model: get_optional(&get, "LOKA_MODEL", "gpt-5"),
            agent_id: get_optional(&get, "LOKA_AGENT_ID", "loka-agent"),
            state_dir: get_state_dir(&get),
        })
    }
}

fn get_required<F>(get: &F, key: &str) -> Result<String>
where
    F: Fn(&str) -> Option<String>,
{
    get(key)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("{key} is required"))
}

fn get_optional<F>(get: &F, key: &str, default: &str) -> String
where
    F: Fn(&str) -> Option<String>,
{
    get(key)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn get_optional_url<F>(get: &F, key: &str, default: &str) -> Result<String>
where
    F: Fn(&str) -> Option<String>,
{
    let value = get_optional(get, key, default);
    normalize_url(key, &value)
}

fn normalize_url(key: &str, value: &str) -> Result<String> {
    let parsed = Url::parse(value).with_context(|| format!("{key} must be a valid URL"))?;
    match parsed.scheme() {
        "http" | "https" => Ok(value.trim_end_matches('/').to_string()),
        scheme => Err(anyhow!("{key} must use http or https, got {scheme}")),
    }
}

fn get_state_dir<F>(get: &F) -> PathBuf
where
    F: Fn(&str) -> Option<String>,
{
    if let Some(value) = get("LOKA_STATE_DIR")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        return PathBuf::from(value);
    }

    if let Some(value) = get("HOME")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        return PathBuf::from(value).join(".loka");
    }

    PathBuf::from(".loka")
}
