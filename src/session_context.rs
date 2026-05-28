use crate::session::{SessionTurn, ToolCallRecord};

const MAX_TOOL_JSON_CHARS: usize = 4_096;

pub(crate) fn format_session_context(
    session_id: &str,
    turns: &[SessionTurn],
    tool_calls: &[ToolCallRecord],
) -> String {
    let mut output = String::with_capacity(
        256 + turns.iter().map(|turn| turn.content.len()).sum::<usize>() + tool_calls.len() * 256,
    );
    output.push_str("Session id: ");
    output.push_str(session_id);
    output.push_str("\n\n");

    for turn in turns {
        output.push_str(turn.role.as_str());
        output.push_str(": ");
        output.push_str(turn.content.trim());
        output.push_str("\n\n");
    }

    if !tool_calls.is_empty() {
        output.push_str("# Tool Calls\n\n");
        for call in tool_calls {
            output.push_str("- ");
            output.push_str(call.status.as_str());
            output.push(' ');
            output.push_str(&call.name);
            output.push_str(" at ");
            output.push_str(
                call.completed_at
                    .as_deref()
                    .unwrap_or(call.created_at.as_str()),
            );
            output.push('\n');
            output.push_str("  input: ");
            output.push_str(&truncate_chars(
                &call.input.to_string(),
                MAX_TOOL_JSON_CHARS,
            ));
            output.push('\n');
            if let Some(output_json) = &call.output {
                output.push_str("  output: ");
                output.push_str(&truncate_chars(
                    &output_json.to_string(),
                    MAX_TOOL_JSON_CHARS,
                ));
                output.push('\n');
            }
            if let Some(error) = &call.error {
                output.push_str("  error: ");
                output.push_str(&truncate_chars(error, MAX_TOOL_JSON_CHARS));
                output.push('\n');
            }
            output.push('\n');
        }
    }

    output
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}
