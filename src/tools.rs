use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fmt;

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
    pub name: &'static str,
    pub description: &'static str,
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
    definitions: BTreeMap<&'static str, ToolDefinition>,
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
            definitions.insert(definition.name, definition);
        }
        Self { definitions }
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
        ToolDefinition {
            name: "session_list",
            description: "List persisted agent sessions.",
            access: ToolAccess::Read,
            input_schema: object_schema(vec![number_property("limit", 1, 200)], &[]),
        },
        ToolDefinition {
            name: "session_search",
            description: "Search persisted session turns using the indexed session store.",
            access: ToolAccess::Read,
            input_schema: object_schema(
                vec![
                    string_property("query", 1),
                    number_property("limit", 1, 200),
                ],
                &["query"],
            ),
        },
        ToolDefinition {
            name: "wiki_rag",
            description: "Fetch relevant memory context from personal-wiki.",
            access: ToolAccess::Read,
            input_schema: object_schema(
                vec![
                    string_property("query", 0),
                    number_property("limit", 1, 20),
                    number_property("depth", 0, 3),
                ],
                &["query"],
            ),
        },
        ToolDefinition {
            name: "wiki_add_note",
            description: "Create a proposal-first memory note in personal-wiki.",
            access: ToolAccess::Write,
            input_schema: object_schema(
                vec![
                    string_property("title", 1),
                    string_property("body", 1),
                    array_string_property("tags"),
                ],
                &["title", "body"],
            ),
        },
        ToolDefinition {
            name: "read_file",
            description: "Read a UTF-8 file from an approved runtime workspace.",
            access: ToolAccess::Read,
            input_schema: object_schema(vec![string_property("path", 1)], &["path"]),
        },
        ToolDefinition {
            name: "search_files",
            description: "Search files in an approved runtime workspace.",
            access: ToolAccess::Read,
            input_schema: object_schema(
                vec![string_property("query", 1), string_property("glob", 0)],
                &["query"],
            ),
        },
        ToolDefinition {
            name: "git_status",
            description: "Inspect git status in an approved runtime workspace.",
            access: ToolAccess::Read,
            input_schema: object_schema(vec![string_property("path", 0)], &[]),
        },
        ToolDefinition {
            name: "shell",
            description: "Run a shell command in an approved runtime workspace.",
            access: ToolAccess::Execute,
            input_schema: object_schema(
                vec![
                    string_property("command", 1),
                    string_property("working_directory", 0),
                ],
                &["command"],
            ),
        },
        ToolDefinition {
            name: "learn_session",
            description: "Extract durable knowledge from a persisted session and write a proposal.",
            access: ToolAccess::Orchestrate,
            input_schema: object_schema(vec![string_property("session_id", 1)], &["session_id"]),
        },
    ]
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
