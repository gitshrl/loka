use std::collections::BTreeSet;
use std::fmt;

use crate::tools::ToolRegistry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    Ask,
    AutoRead,
    Plan,
    Bypass,
}

impl PermissionMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::AutoRead => "auto-read",
            Self::Plan => "plan",
            Self::Bypass => "bypass",
        }
    }
}

impl fmt::Display for PermissionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalDecision {
    Allow,
    Ask,
    Deny { reason: String },
}

impl ApprovalDecision {
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Ask => "ask",
            Self::Deny { .. } => "deny",
        }
    }
}

impl fmt::Display for ApprovalDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allow => f.write_str("allow"),
            Self::Ask => f.write_str("ask"),
            Self::Deny { reason } => write!(f, "deny: {reason}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalPolicy {
    mode: PermissionMode,
    allowed_tools: BTreeSet<String>,
    denied_tools: BTreeSet<String>,
}

impl Default for ApprovalPolicy {
    fn default() -> Self {
        Self::new(PermissionMode::AutoRead)
    }
}

impl ApprovalPolicy {
    #[must_use]
    pub fn new(mode: PermissionMode) -> Self {
        Self {
            mode,
            allowed_tools: BTreeSet::new(),
            denied_tools: BTreeSet::new(),
        }
    }

    #[must_use]
    pub fn with_allowed(mut self, tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.allowed_tools.extend(tools.into_iter().map(Into::into));
        self
    }

    #[must_use]
    pub fn with_denied(mut self, tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.denied_tools.extend(tools.into_iter().map(Into::into));
        self
    }

    #[must_use]
    pub const fn mode(&self) -> PermissionMode {
        self.mode
    }

    #[must_use]
    pub fn evaluate(&self, registry: &ToolRegistry, tool_name: &str) -> ApprovalDecision {
        let Some(tool) = registry.get(tool_name) else {
            return ApprovalDecision::Deny {
                reason: format!("unknown tool {tool_name}"),
            };
        };

        if self.denied_tools.contains(tool_name) {
            return ApprovalDecision::Deny {
                reason: format!("{tool_name} is explicitly denied"),
            };
        }

        if self.mode == PermissionMode::Plan {
            return ApprovalDecision::Deny {
                reason: "plan mode blocks tool execution".to_string(),
            };
        }

        if self.allowed_tools.contains(tool_name) {
            return ApprovalDecision::Allow;
        }

        match self.mode {
            PermissionMode::Ask => ApprovalDecision::Ask,
            PermissionMode::AutoRead => {
                if tool.access.is_read_only() {
                    ApprovalDecision::Allow
                } else {
                    ApprovalDecision::Ask
                }
            }
            PermissionMode::Plan => unreachable!("plan mode returned before mode evaluation"),
            PermissionMode::Bypass => ApprovalDecision::Allow,
        }
    }
}
