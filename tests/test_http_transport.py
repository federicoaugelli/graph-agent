from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import SystemMessage

from conftest import ScriptedLLMBackend
from graph_agent.config import AppConfig
from graph_agent.core.service import AgentService
from graph_agent.core.sessions import SessionManager
from graph_agent.events import ApprovalRequestEvent
from graph_agent.models.llm import StreamChunk, ToolCallRequest
from graph_agent.tools.base import ToolContext, ToolSpec
from graph_agent.transports.http import create_app


class ConfirmTool:
    def __init__(self) -> None:
        self.executed: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="confirm",
            description="Do something dangerous.",
            parameters={"type": "object", "properties": {}, "required": []},
            requires_approval=True,
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        self.executed.append(args)
        return "done"


def make_client(
    service: AgentService, config: AppConfig, sessions: SessionManager
) -> httpx.AsyncClient:
    app = create_app(service, config, sessions)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent.test")


def text_chunks(text: str) -> list[StreamChunk]:
    return [StreamChunk(delta_text=text, finish_reason="stop")]


async def test_health_and_models(client: httpx.AsyncClient) -> None:
    assert (await client.get("/health")).status_code == 200
    models = (await client.get("/v1/models")).json()
    assert models["object"] == "list"
    assert models["data"][0]["id"] == "graph-agent"


async def test_completion_non_streaming(app_config: AppConfig) -> None:
    backend = ScriptedLLMBackend(
        [
            [
                StreamChunk(delta_text="Hello ", finish_reason=None),
                StreamChunk(delta_text="world", finish_reason="stop"),
            ]
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    client = make_client(service, app_config, SessionManager(service))

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "parla"}]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "graph-agent"
    assert body["id"].startswith("chatcmpl-")
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hello world"
    assert choice["finish_reason"] == "stop"

    await client.aclose()
    await service.shutdown()


async def test_completion_empty_messages_is_422(app_config: AppConfig) -> None:
    service = AgentService(app_config)
    await service.setup(ScriptedLLMBackend([]))
    client = make_client(service, app_config, SessionManager(service))

    resp = await client.post("/v1/chat/completions", json={"messages": []})

    assert resp.status_code == 422

    await client.aclose()
    await service.shutdown()


async def test_completion_streams_sse(app_config: AppConfig) -> None:
    backend = ScriptedLLMBackend(
        [
            [
                StreamChunk(delta_text="abc", finish_reason=None),
                StreamChunk(delta_text="def", finish_reason="stop"),
            ]
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    client = make_client(service, app_config, SessionManager(service))

    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "go"}], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = [line async for line in resp.aiter_lines()]

    data = [line.removeprefix("data:").strip() for line in lines if line.startswith("data:")]
    assert data[-1] == "[DONE]"
    chunks = [json.loads(item) for item in data[:-1]]

    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    streamed = "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks)
    assert streamed == "abcdef"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    await client.aclose()
    await service.shutdown()


async def test_user_field_maps_to_session(app_config: AppConfig) -> None:
    backend = ScriptedLLMBackend(
        [
            text_chunks("answer-1"),
            text_chunks("answer-2"),
            text_chunks("answer-3"),
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    client = make_client(service, app_config, SessionManager(service))

    async def ask(user: str, content: str) -> None:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": content}], "user": user},
        )
        assert resp.status_code == 200

    await ask("alice", "uno")
    await ask("bob", "due")
    await ask("alice", "tre")

    alice_last = backend.calls[-1]
    contents = [message.content for message in alice_last]
    assert "uno" in contents
    assert "answer-1" in contents
    assert "due" not in contents

    await client.aclose()
    await service.shutdown()


async def test_missing_user_uses_default_session(app_config: AppConfig) -> None:
    backend = ScriptedLLMBackend([text_chunks("r1"), text_chunks("r2")])
    service = AgentService(app_config)
    await service.setup(backend)
    client = make_client(service, app_config, SessionManager(service))

    payload = {"messages": [{"role": "user", "content": "ping"}]}
    await client.post("/v1/chat/completions", json=payload)
    await client.post("/v1/chat/completions", json=payload)

    contents = [message.content for message in backend.calls[-1]]
    assert contents.count("ping") == 2

    await client.aclose()
    await service.shutdown()


async def test_http_is_always_auto_with_system_prompt(app_config: AppConfig) -> None:
    backend = ScriptedLLMBackend(
        [
            [
                StreamChunk(
                    tool_calls=[ToolCallRequest(id="call_http", name="confirm", args={})],
                    finish_reason="tool_calls",
                )
            ],
            text_chunks("fatto"),
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    tool = ConfirmTool()
    service.registry.register(tool)
    client = make_client(service, app_config, SessionManager(service))

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "burn"}], "user": "carol"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert "approval" not in body
    assert body["choices"][0]["message"]["content"] == "fatto"
    assert tool.executed == [{}]

    first_call = backend.calls[0]
    assert isinstance(first_call[0], SystemMessage)
    assert "confirm" in first_call[0].content.lower()

    await client.aclose()
    await service.shutdown()


async def test_approvals_endpoint_resumes_manual_session(app_config: AppConfig) -> None:
    backend = ScriptedLLMBackend(
        [
            [
                StreamChunk(
                    tool_calls=[ToolCallRequest(id="call_http", name="confirm", args={})],
                    finish_reason="tool_calls",
                )
            ],
            text_chunks("fatto"),
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    tool = ConfirmTool()
    service.registry.register(tool)
    sessions = SessionManager(service)
    client = make_client(service, app_config, sessions)

    thread_id = await sessions.thread_id("http", "dave")
    events = [event async for event in service.run(thread_id, "burn")]
    assert isinstance(events[0], ApprovalRequestEvent)
    assert tool.executed == []

    resume = await client.post("/v1/sessions/dave/approvals/call_http", json={"approved": True})

    assert resume.status_code == 200
    resume_body = resume.json()
    assert resume_body["object"] == "chat.completion"
    assert resume_body["choices"][0]["message"]["content"] == "fatto"
    assert tool.executed == [{}]

    await client.aclose()
    await service.shutdown()


async def test_new_command_resets_session(app_config: AppConfig) -> None:
    backend = ScriptedLLMBackend([text_chunks("uno-risposta"), text_chunks("due-risposta")])
    service = AgentService(app_config)
    await service.setup(backend)
    client = make_client(service, app_config, SessionManager(service))

    async def ask(content: str) -> str:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": content}], "user": "eve"},
        )
        assert resp.status_code == 200
        return str(resp.json()["choices"][0]["message"]["content"])

    await ask("uno")
    reset_reply = await ask("/new")
    await ask("due")

    assert "new session" in reset_reply
    last_turn = [message.content for message in backend.calls[-1]]
    assert "due" in last_turn
    assert "uno" not in last_turn
    assert "uno-risposta" not in last_turn

    await client.aclose()
    await service.shutdown()


async def test_file_event_renders_as_text_note(app_config: AppConfig) -> None:
    workspace = app_config.agent.workspace
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "x.txt").write_text("data")
    backend = ScriptedLLMBackend(
        [
            [
                StreamChunk(
                    tool_calls=[
                        ToolCallRequest(
                            id="call_f", name="send_file", args={"path": "x.txt", "caption": "ecco"}
                        )
                    ],
                    finish_reason="tool_calls",
                )
            ],
            text_chunks("fatto"),
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    client = make_client(service, app_config, SessionManager(service))

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "manda"}], "user": "frank"},
    )

    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert "[file:" in content
    assert "x.txt" in content

    await client.aclose()
    await service.shutdown()


async def test_resume_unknown_session_is_404(app_config: AppConfig) -> None:
    service = AgentService(app_config)
    await service.setup(ScriptedLLMBackend([text_chunks("x")]))
    client = make_client(service, app_config, SessionManager(service))

    resp = await client.post("/v1/sessions/ghost/approvals/nope", json={"approved": False})

    assert resp.status_code == 404

    await client.aclose()
    await service.shutdown()


TOKEN_ENV = "GRAPH_AGENT_TEST_HTTP_TOKEN"
TOKEN = "s3cret-token"


def secure(config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    config.channels.http.api_key_env = TOKEN_ENV
    return config


async def test_v1_requires_bearer_token(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    secure(app_config, monkeypatch)
    service = AgentService(app_config)
    await service.setup(ScriptedLLMBackend([text_chunks("ok")]))
    client = make_client(service, app_config, SessionManager(service))

    assert (await client.get("/health")).status_code == 200

    missing = await client.get("/v1/models")
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert missing.json()["error"]["code"] == "invalid_api_key"

    wrong = await client.get("/v1/models", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401

    no_scheme = await client.get("/v1/models", headers={"Authorization": TOKEN})
    assert no_scheme.status_code == 401

    ok = await client.get("/v1/models", headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.status_code == 200

    await client.aclose()
    await service.shutdown()


async def test_completion_is_protected(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    secure(app_config, monkeypatch)
    service = AgentService(app_config)
    await service.setup(ScriptedLLMBackend([text_chunks("ok")]))
    client = make_client(service, app_config, SessionManager(service))

    payload = {"messages": [{"role": "user", "content": "ping"}]}
    assert (await client.post("/v1/chat/completions", json=payload)).status_code == 401
    assert (
        await client.post("/v1/chat/completions", json={**payload, "stream": True})
    ).status_code == 401

    authorized = await client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert authorized.status_code == 200

    await client.aclose()
    await service.shutdown()


async def test_missing_token_env_fails_closed(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    app_config.channels.http.api_key_env = TOKEN_ENV
    service = AgentService(app_config)
    await service.setup(ScriptedLLMBackend([text_chunks("ok")]))
    client = make_client(service, app_config, SessionManager(service))

    resp = await client.get("/v1/models", headers={"Authorization": "Bearer anything"})

    assert resp.status_code == 401

    await client.aclose()
    await service.shutdown()


async def test_no_api_key_env_keeps_v1_open(app_config: AppConfig) -> None:
    assert app_config.channels.http.api_key_env is None
    service = AgentService(app_config)
    await service.setup(ScriptedLLMBackend([text_chunks("ok")]))
    client = make_client(service, app_config, SessionManager(service))

    assert (await client.get("/v1/models")).status_code == 200

    await client.aclose()
    await service.shutdown()


@pytest.fixture
async def client(app_config: AppConfig) -> Any:
    service = AgentService(app_config)
    await service.setup(ScriptedLLMBackend([text_chunks("ok")]))
    http_client = make_client(service, app_config, SessionManager(service))
    yield http_client
    await http_client.aclose()
    await service.shutdown()
