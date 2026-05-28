use serde::{Deserialize, Serialize};
use std::fmt;

use crate::messages::Message;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct TokenUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

impl TokenUsage {
    pub const ZERO: Self = Self {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };

    #[must_use]
    pub const fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }

    #[must_use]
    pub const fn total(total_tokens: u64) -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens,
        }
    }

    #[must_use]
    pub const fn estimated_prompt(prompt_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            completion_tokens: 0,
            total_tokens: prompt_tokens,
        }
    }

    #[must_use]
    pub const fn normalized(self) -> Self {
        if self.total_tokens == 0 {
            Self::new(self.prompt_tokens, self.completion_tokens)
        } else {
            self
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TokenScope {
    Prompt,
    Tool,
    Worker,
    Session,
}

impl TokenScope {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prompt => "prompt",
            Self::Tool => "tool",
            Self::Worker => "worker",
            Self::Session => "session",
        }
    }

    #[must_use]
    pub fn from_db(value: &str) -> Option<Self> {
        match value {
            "prompt" => Some(Self::Prompt),
            "tool" => Some(Self::Tool),
            "worker" => Some(Self::Worker),
            "session" => Some(Self::Session),
            _ => None,
        }
    }
}

impl fmt::Display for TokenScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str((*self).as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenBudget {
    max_tokens: u64,
}

impl TokenBudget {
    #[must_use]
    pub const fn new(max_tokens: u64) -> Self {
        Self { max_tokens }
    }

    #[must_use]
    pub const fn check(self, used_tokens: u64) -> TokenBudgetCheck {
        if used_tokens > self.max_tokens {
            TokenBudgetCheck::Exceeded {
                used_tokens,
                max_tokens: self.max_tokens,
            }
        } else {
            TokenBudgetCheck::Within {
                used_tokens,
                max_tokens: self.max_tokens,
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenBudgetCheck {
    Within { used_tokens: u64, max_tokens: u64 },
    Exceeded { used_tokens: u64, max_tokens: u64 },
}

impl TokenBudgetCheck {
    #[must_use]
    pub const fn is_exceeded(self) -> bool {
        matches!(self, Self::Exceeded { .. })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct TokenUsageSummary {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

impl TokenUsageSummary {
    pub fn add(&mut self, usage: TokenUsage) {
        let usage = usage.normalized();
        self.prompt_tokens = self.prompt_tokens.saturating_add(usage.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(usage.completion_tokens);
        self.total_tokens = self.total_tokens.saturating_add(usage.total_tokens);
    }
}

#[must_use]
pub fn estimate_text_tokens(text: &str) -> u64 {
    let bytes = text.trim().len() as u64;
    if bytes == 0 { 0 } else { bytes.div_ceil(4) }
}

#[must_use]
pub fn estimate_json_tokens(value: &serde_json::Value) -> u64 {
    estimate_text_tokens(&value.to_string())
}

#[must_use]
pub fn estimate_messages_tokens(messages: &[Message]) -> u64 {
    messages
        .iter()
        .map(|message| {
            estimate_text_tokens(message.role.as_str()) + estimate_text_tokens(&message.content) + 4
        })
        .sum()
}
