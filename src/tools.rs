use anyhow::{Result, anyhow};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fmt;

use crate::mcp::McpTool;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolAccess {
    Read,
    Write,
    Execute,
    Orchestrate,
}

impl ToolAccess {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Execute => "execute",
            Self::Orchestrate => "orchestrate",
        }
    }

    #[must_use]
    pub const fn is_read_only(self) -> bool {
        matches!(self, Self::Read)
    }
}

impl fmt::Display for ToolAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub access: ToolAccess,
    pub input_schema: Value,
}

impl ToolDefinition {
    #[must_use]
    pub fn read_only_hint(&self) -> bool {
        self.access.is_read_only()
    }
}

#[derive(Debug, Clone)]
pub struct ToolRegistry {
    definitions: BTreeMap<String, ToolDefinition>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::built_in()
    }
}

impl ToolRegistry {
    #[must_use]
    pub fn built_in() -> Self {
        let mut definitions = BTreeMap::new();
        for definition in built_in_definitions() {
            definitions.insert(definition.name.clone(), definition);
        }
        Self { definitions }
    }

    /// Adds externally discovered MCP tools to this registry.
    ///
    /// # Errors
    ///
    /// Returns an error when a discovered tool collides with an existing tool name.
    pub fn with_mcp_tools(mut self, tools: impl IntoIterator<Item = McpTool>) -> Result<Self> {
        for tool in tools {
            self.insert(tool.into_tool_definition())?;
        }
        Ok(self)
    }

    /// Adds a tool definition to this registry.
    ///
    /// # Errors
    ///
    /// Returns an error when the name is empty or already registered.
    pub fn insert(&mut self, definition: ToolDefinition) -> Result<()> {
        if definition.name.trim().is_empty() {
            return Err(anyhow!("tool name is required"));
        }
        if self.definitions.contains_key(&definition.name) {
            return Err(anyhow!("tool {} is already registered", definition.name));
        }
        self.definitions.insert(definition.name.clone(), definition);
        Ok(())
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ToolDefinition> {
        self.definitions.get(name)
    }

    #[must_use]
    pub fn list(&self) -> Vec<&ToolDefinition> {
        self.definitions.values().collect()
    }

    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.definitions.contains_key(name)
    }
}

fn built_in_definitions() -> Vec<ToolDefinition> {
    vec![
        built_in_tool(
            "session_list",
            "List persisted agent sessions.",
            ToolAccess::Read,
            object_schema(vec![number_property("limit", 1, 200)], &[]),
        ),
        built_in_tool(
            "session_search",
            "Search persisted session turns using the indexed session store.",
            ToolAccess::Read,
            object_schema(
                vec![
                    string_property("query", 1),
                    number_property("limit", 1, 200),
                ],
                &["query"],
            ),
        ),
        built_in_tool(
            "memory_search",
            "Fetch relevant memory context from memory API.",
            ToolAccess::Read,
            object_schema(
                vec![
                    string_property("query", 0),
                    number_property("limit", 1, 20),
                    number_property("depth", 0, 3),
                ],
                &["query"],
            ),
        ),
        built_in_tool(
            "memory_propose",
            "Create a proposal-first memory note in memory API.",
            ToolAccess::Write,
            object_schema(
                vec![
                    string_property("title", 1),
                    string_property("body", 1),
                    array_string_property("tags"),
                ],
                &["title", "body"],
            ),
        ),
        built_in_tool(
            "read_file",
            "Read a UTF-8 file from an approved runtime workspace.",
            ToolAccess::Read,
            object_schema(vec![string_property("path", 1)], &["path"]),
        ),
        built_in_tool(
            "search_files",
            "Search files in an approved runtime workspace.",
            ToolAccess::Read,
            object_schema(
                vec![string_property("query", 1), string_property("glob", 0)],
                &["query"],
            ),
        ),
        built_in_tool(
            "git_status",
            "Inspect git status in an approved runtime workspace.",
            ToolAccess::Read,
            object_schema(vec![string_property("path", 0)], &[]),
        ),
        built_in_tool(
            "shell",
            "Run a shell command in an approved runtime workspace.",
            ToolAccess::Execute,
            object_schema(
                vec![
                    string_property("command", 1),
                    string_property("working_directory", 0),
                ],
                &["command"],
            ),
        ),
        built_in_tool(
            "learn_session",
            "Extract durable knowledge from a persisted session and write a proposal.",
            ToolAccess::Orchestrate,
            object_schema(vec![string_property("session_id", 1)], &["session_id"]),
        ),
    ]
}

fn built_in_tool(
    name: &'static str,
    description: &'static str,
    access: ToolAccess,
    input_schema: Value,
) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: description.to_string(),
        access,
        input_schema,
    }
}

fn object_schema(properties: Vec<(&'static str, Value)>, required: &[&'static str]) -> Value {
    let properties = properties
        .into_iter()
        .map(|(name, schema)| (name.to_string(), schema))
        .collect::<serde_json::Map<_, _>>();

    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": properties,
        "required": required,
    })
}

fn string_property(description: &'static str, min_length: u64) -> (&'static str, Value) {
    (
        description,
        json!({
            "type": "string",
            "minLength": min_length,
        }),
    )
}

fn number_property(name: &'static str, minimum: u64, maximum: u64) -> (&'static str, Value) {
    (
        name,
        json!({
            "type": "integer",
            "minimum": minimum,
            "maximum": maximum,
        }),
    )
}

fn array_string_property(name: &'static str) -> (&'static str, Value) {
    (
        name,
        json!({
            "type": "array",
            "items": { "type": "string", "minLength": 1 },
        }),
    )
}
