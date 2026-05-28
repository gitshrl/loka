use httpmock::prelude::*;
use loka_agent::config::ModelProtocol;
use loka_agent::messages::Message;
use loka_agent::model::{ChatRequest, ModelClient};
use serde_json::json;

#[tokio::test]
async fn chat_completion_sends_openai_compatible_request() {
    let server = MockServer::start();
    let completion = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .header("authorization", "Bearer sk-test")
            .json_body(json!({
                "model": "gpt-5.5",
                "messages": [
                    { "role": "user", "content": "ping" }
                ]
            }));

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "role": "assistant", "content": "pong" } }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7
                }
            }));
    });

    let client = ModelClient::new(server.base_url(), "sk-test".to_string());
    let output = client
        .chat(ChatRequest {
            model: "gpt-5.5".to_string(),
            messages: vec![Message::user("ping")],
        })
        .await
        .expect("chat should succeed");

    completion.assert();
    assert_eq!(output.content, "pong");
    assert_eq!(output.usage.expect("usage").total_tokens, 7);
}

#[tokio::test]
async fn chat_completion_sends_anthropic_compatible_request() {
    let server = MockServer::start();
    let completion = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/messages")
            .header("x-api-key", "sk-test")
            .header("anthropic-version", "2023-06-01")
            .json_body(json!({
                "model": "gpt-5.5",
                "max_tokens": 4096,
                "system": "system guidance",
                "messages": [
                    { "role": "user", "content": "ping" }
                ]
            }));

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "content": [
                    { "type": "text", "text": "pong" }
                ],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 2
                }
            }));
    });

    let client = ModelClient::with_protocol(
        server.base_url(),
        "sk-test".to_string(),
        ModelProtocol::AnthropicCompatible,
    );
    let output = client
        .chat(ChatRequest {
            model: "gpt-5.5".to_string(),
            messages: vec![Message::system("system guidance"), Message::user("ping")],
        })
        .await
        .expect("chat should succeed");

    completion.assert();
    assert_eq!(output.content, "pong");
    assert_eq!(output.usage.expect("usage").total_tokens, 7);
}

#[tokio::test]
async fn chat_completion_reports_upstream_error_body() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/chat/completions");
        then.status(503).body("no provider accounts available");
    });

    let client = ModelClient::new(server.base_url(), "sk-test".to_string());
    let error = client
        .chat(ChatRequest {
            model: "gpt-5.5".to_string(),
            messages: vec![Message::user("ping")],
        })
        .await
        .expect_err("upstream 503 should fail");

    let message = error.to_string();
    assert!(message.contains("503"));
    assert!(message.contains("no provider accounts available"));
}

#[tokio::test]
async fn chat_completion_rejects_empty_assistant_content() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "role": "assistant", "content": "   " } }
                ]
            }));
    });

    let client = ModelClient::new(server.base_url(), "sk-test".to_string());
    let error = client
        .chat(ChatRequest {
            model: "gpt-5.5".to_string(),
            messages: vec![Message::user("ping")],
        })
        .await
        .expect_err("empty content should fail");

    assert!(error.to_string().contains("no assistant content"));
}

#[tokio::test]
async fn chat_stream_sends_stream_request_and_yields_deltas() {
    let server = MockServer::start();
    let stream = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .header("authorization", "Bearer sk-test")
            .json_body(json!({
                "model": "gpt-5.5",
                "messages": [
                    { "role": "user", "content": "ping" }
                ],
                "stream": true
            }));

        then.status(200)
            .header("content-type", "text/event-stream")
            .body(concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"po\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"ng\"}}]}\n\n",
                "data: [DONE]\n\n"
            ));
    });

    let client = ModelClient::new(server.base_url(), "sk-test".to_string());
    let mut deltas = Vec::new();
    let output = client
        .chat_stream(
            ChatRequest {
                model: "gpt-5.5".to_string(),
                messages: vec![Message::user("ping")],
            },
            |delta| {
                deltas.push(delta.to_string());
                Ok(())
            },
        )
        .await
        .expect("stream should succeed");

    stream.assert();
    assert_eq!(deltas, vec!["po", "ng"]);
    assert_eq!(output.content, "pong");
}

#[tokio::test]
async fn chat_stream_handles_anthropic_compatible_events() {
    let server = MockServer::start();
    let stream = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/messages")
            .header("x-api-key", "sk-test")
            .header("anthropic-version", "2023-06-01")
            .json_body(json!({
                "model": "gpt-5.5",
                "max_tokens": 4096,
                "stream": true,
                "messages": [
                    { "role": "user", "content": "ping" }
                ]
            }));

        then.status(200)
            .header("content-type", "text/event-stream")
            .body(concat!(
                "event: content_block_delta\n",
                "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"po\"}}\n\n",
                "event: content_block_delta\n",
                "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"ng\"}}\n\n",
                "event: message_delta\n",
                "data: {\"type\":\"message_delta\",\"usage\":{\"input_tokens\":5,\"output_tokens\":2}}\n\n",
                "event: message_stop\n",
                "data: {\"type\":\"message_stop\"}\n\n"
            ));
    });

    let client = ModelClient::with_protocol(
        server.base_url(),
        "sk-test".to_string(),
        ModelProtocol::AnthropicCompatible,
    );
    let mut deltas = Vec::new();
    let output = client
        .chat_stream(
            ChatRequest {
                model: "gpt-5.5".to_string(),
                messages: vec![Message::user("ping")],
            },
            |delta| {
                deltas.push(delta.to_string());
                Ok(())
            },
        )
        .await
        .expect("stream should succeed");

    stream.assert();
    assert_eq!(deltas, vec!["po", "ng"]);
    assert_eq!(output.content, "pong");
    assert_eq!(output.usage.expect("usage").total_tokens, 7);
}
