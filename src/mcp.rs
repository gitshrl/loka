use anyhow::{Context, Result, anyhow, bail};
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::tools::{ToolAccess, ToolDefinition};

const JSON_RPC_VERSION: &str = "2.0";
const MCP_TOOL_PREFIX: &str = "mcp";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerConfig {
    pub name: String,
    pub endpoint: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct McpTool {
    pub server: String,
    pub name: String,
    pub raw_name: String,
    pub description: String,
    pub input_schema: Value,
    pub access: ToolAccess,
}

impl McpTool {
    #[must_use]
    pub fn into_tool_definition(self) -> ToolDefinition {
        ToolDefinition {
            name: self.name,
            description: self.description,
            access: self.access,
            input_schema: self.input_schema,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct McpToolOutput {
    pub content: String,
    pub raw: Value,
}

#[derive(Debug, Clone)]
pub struct McpClient {
    http: Client,
    server: String,
    endpoint: String,
}

#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    id: u64,
    method: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

#[derive(Debug, Deserialize)]
struct ListToolsResult {
    #[serde(default)]
    tools: Vec<WireTool>,
}

#[derive(Debug, Deserialize)]
struct WireTool {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default, rename = "inputSchema")]
    input_schema: Option<Value>,
    #[serde(default)]
    annotations: Option<ToolAnnotations>,
}

#[derive(Debug, Deserialize)]
struct ToolAnnotations {
    #[serde(default, rename = "readOnlyHint")]
    read_only_hint: bool,
    #[serde(default, rename = "destructiveHint")]
    destructive_hint: bool,
}

impl McpClient {
    /// Creates an HTTP JSON-RPC MCP client for one configured server endpoint.
    ///
    /// # Errors
    ///
    /// Returns an error when the server name or endpoint is invalid.
    pub fn new(config: McpServerConfig) -> Result<Self> {
        validate_mcp_component("MCP server name", &config.name)?;
        let endpoint = normalize_http_endpoint(&config.endpoint)?;

        Ok(Self {
            http: Client::new(),
            server: config.name,
            endpoint,
        })
    }

    #[must_use]
    pub fn server(&self) -> &str {
        &self.server
    }

    /// Lists tools exposed by this MCP server.
    ///
    /// # Errors
    ///
    /// Returns an error when the server rejects the JSON-RPC request or returns invalid tools.
    pub async fn list_tools(&self) -> Result<Vec<McpTool>> {
        let result = self.request(1, "tools/list", None).await?;
        let listing: ListToolsResult =
            serde_json::from_value(result).context("parse MCP tools/list response")?;

        listing
            .tools
            .into_iter()
            .map(|tool| self.decode_tool(tool))
            .collect()
    }

    /// Calls one qualified MCP tool.
    ///
    /// # Errors
    ///
    /// Returns an error when the tool name is not owned by this server, the server rejects the
    /// request, or the response cannot be decoded.
    pub async fn call_tool(&self, qualified_name: &str, arguments: Value) -> Result<McpToolOutput> {
        let (server, raw_name) = parse_mcp_tool_name(qualified_name)
            .ok_or_else(|| anyhow!("invalid MCP tool name {qualified_name}"))?;
        if server != self.server {
            bail!(
                "MCP tool {qualified_name} belongs to server {server}, not {}",
                self.server
            );
        }

        let result = self
            .request(
                2,
                "tools/call",
                Some(json!({
                    "name": raw_name,
                    "arguments": arguments,
                })),
            )
            .await?;
        let content = stringify_mcp_result(&result);

        Ok(McpToolOutput {
            content,
            raw: result,
        })
    }

    async fn request(&self, id: u64, method: &'static str, params: Option<Value>) -> Result<Value> {
        let response = self
            .http
            .post(&self.endpoint)
            .json(&JsonRpcRequest {
                jsonrpc: JSON_RPC_VERSION,
                id,
                method,
                params,
            })
            .send()
            .await
            .with_context(|| format!("send MCP {method} request"))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("MCP {method} failed with {status}: {body}"));
        }

        let body: JsonRpcResponse = response
            .json()
            .await
            .with_context(|| format!("parse MCP {method} response"))?;
        if let Some(error) = body.error {
            return Err(anyhow!(
                "MCP {method} returned JSON-RPC error {}: {}",
                error.code,
                error.message
            ));
        }

        body.result
            .ok_or_else(|| anyhow!("MCP {method} response did not include result"))
    }

    fn decode_tool(&self, tool: WireTool) -> Result<McpTool> {
        validate_mcp_component("MCP tool name", &tool.name)?;
        let name = qualify_mcp_tool_name(&self.server, &tool.name)?;
        let access = match tool.annotations {
            Some(ToolAnnotations {
                destructive_hint: true,
                ..
            }) => ToolAccess::Write,
            Some(ToolAnnotations {
                read_only_hint: true,
                ..
            }) => ToolAccess::Read,
            _ => ToolAccess::Write,
        };

        Ok(McpTool {
            server: self.server.clone(),
            name,
            raw_name: tool.name,
            description: tool.description,
            input_schema: tool.input_schema.unwrap_or_else(empty_object_schema),
            access,
        })
    }
}

/// Builds the collision-safe name used for MCP-backed tools.
///
/// # Errors
///
/// Returns an error when either component is empty or contains unsupported characters.
pub fn qualify_mcp_tool_name(server: &str, raw_name: &str) -> Result<String> {
    validate_mcp_component("MCP server name", server)?;
    validate_mcp_component("MCP tool name", raw_name)?;
    Ok(format!("{MCP_TOOL_PREFIX}__{server}__{raw_name}"))
}

#[must_use]
pub fn is_mcp_tool_name(name: &str) -> bool {
    parse_mcp_tool_name(name).is_some()
}

/// Wraps MCP text output as untrusted content before it can be fed back to a model.
#[must_use]
pub fn wrap_untrusted_mcp_content(source: &str, content: &str) -> String {
    let content = content
        .replace("<untrusted_content", "&lt;untrusted_content")
        .replace("</untrusted_content>", "&lt;/untrusted_content&gt;");
    format!("<untrusted_content source=\"{source}\">\n{content}\n</untrusted_content>")
}

fn parse_mcp_tool_name(name: &str) -> Option<(String, String)> {
    let mut parts = name.split("__");
    let prefix = parts.next()?;
    let server = parts.next()?;
    let raw_name = parts.next()?;
    if parts.next().is_some() || prefix != MCP_TOOL_PREFIX {
        return None;
    }
    validate_mcp_component("MCP server name", server).ok()?;
    validate_mcp_component("MCP tool name", raw_name).ok()?;
    Some((server.to_string(), raw_name.to_string()))
}

fn validate_mcp_component(kind: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        bail!("{kind} is required");
    }
    if value.contains("__") {
        bail!("{kind} cannot contain double underscores");
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        bail!("{kind} must contain only ASCII letters, digits, underscores, or hyphens");
    }
    Ok(())
}

fn normalize_http_endpoint(endpoint: &str) -> Result<String> {
    let parsed = Url::parse(endpoint).context("MCP endpoint must be a valid URL")?;
    match parsed.scheme() {
        "http" | "https" => Ok(endpoint.trim_end_matches('/').to_string()),
        scheme => Err(anyhow!("MCP endpoint must use http or https, got {scheme}")),
    }
}

fn empty_object_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {},
    })
}

fn stringify_mcp_result(result: &Value) -> String {
    let Some(content) = result.get("content").and_then(Value::as_array) else {
        return stringify_value(result);
    };

    content
        .iter()
        .map(|block| {
            if block.get("type").and_then(Value::as_str) == Some("text")
                && let Some(text) = block.get("text").and_then(Value::as_str)
            {
                return text.to_string();
            }
            stringify_value(block)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn stringify_value(value: &Value) -> String {
    value
        .as_str()
        .map_or_else(|| value.to_string(), ToString::to_string)
}
