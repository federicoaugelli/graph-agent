from __future__ import annotations

import hmac
import json
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from graph_agent.config import AppConfig
from graph_agent.core.service import AgentService
from graph_agent.core.sessions import SessionManager
from graph_agent.events import (
    ApprovalRequestEvent,
    ErrorEvent,
    Event,
    FileEvent,
    TokenEvent,
)
from graph_agent.transports.realtime.bridge import RealtimeBridge, StarletteWebSocketConnection

DEFAULT_SESSION_ID = "default"
CHANNEL = "http"
NEW_COMMAND = "/new"

HTTP_SYSTEM_PROMPT = """\
You are graph-agent, a personal assistant reached through an OpenAI-compatible \
HTTP API. The human can only reply in chat: there are no approval buttons or \
interrupts on this channel, and tool approval is handled linguistically.

Before any sensitive action (writing, overwriting or deleting files, running \
shell commands with side effects, network calls, or anything irreversible), \
describe exactly what you intend to do and explicitly ask the user to confirm. \
Do NOT execute the sensitive action in the same turn: stop, send the question, \
and wait for the user's confirmation in a later message. Read-only or clearly \
harmless operations can be executed directly. If the user already confirmed a \
specific action, proceed without re-asking.\
"""


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "graph-agent"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    session_id: str | None = Field(default=None, alias="user")
    temperature: float | None = None
    max_tokens: int | None = None

    model_config = {"populate_by_name": True, "extra": "allow"}


class ResumeRequest(BaseModel):
    approved: bool


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]


def _approval_suffix(approval: ApprovalRequestEvent, session_id: str) -> str:
    return (
        f"[approval required for tool '{approval.name}' with args "
        f"{json.dumps(approval.args, ensure_ascii=False)} - approval_id: "
        f"{approval.approval_id} - resolve via POST "
        f"/v1/sessions/{session_id}/approvals/{approval.approval_id}]"
    )


async def _collect_stream(stream: AsyncIterator[Event]) -> tuple[str, ApprovalRequestEvent | None]:
    parts: list[str] = []
    approval: ApprovalRequestEvent | None = None

    async for event in stream:
        if isinstance(event, TokenEvent):
            parts.append(event.delta)
        elif isinstance(event, ErrorEvent):
            parts.append(f"\n[error] {event.message}")
        elif isinstance(event, FileEvent):
            parts.append(_file_note(event))
        elif isinstance(event, ApprovalRequestEvent):
            approval = event

    return "".join(parts), approval


def _file_note(event: FileEvent) -> str:
    caption = f" ({event.caption})" if event.caption else ""
    return f"\n[file: {event.path}{caption}]"


async def _single_token_stream(text: str) -> AsyncIterator[Event]:
    yield TokenEvent(delta=text)


def _completion_body(
    model: str,
    text: str,
    approval: ApprovalRequestEvent | None,
    session_id: str,
) -> dict[str, Any]:
    if approval is not None:
        suffix = _approval_suffix(approval, session_id)
        text = f"{text}\n{suffix}" if text else suffix

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex}",
        model=model,
        choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content=text))],
    )
    body = response.model_dump()

    if approval is not None:
        body["approval"] = {
            "approval_id": approval.approval_id,
            "name": approval.name,
            "args": approval.args,
            "session_id": session_id,
        }

    return body


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "message": "Invalid or missing API key",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        },
        headers={"WWW-Authenticate": "Bearer"},
    )


def _ws_token(websocket: WebSocket) -> str:
    scheme, _, token = websocket.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token:
        return token
    return websocket.query_params.get("api_key") or websocket.query_params.get("token") or ""


def create_app(
    service: AgentService,
    config: AppConfig,
    sessions: SessionManager,
    realtime: RealtimeBridge | None = None,
) -> FastAPI:
    app = FastAPI(title="graph-agent", version="0.1.0")
    auth_env = config.channels.http.api_key_env

    @app.middleware("http")
    async def bearer_auth(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if auth_env is None or not request.url.path.startswith("/v1/"):
            return await call_next(request)

        expected = os.environ.get(auth_env, "")
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if not expected or scheme.lower() != "bearer" or not hmac.compare_digest(token, expected):
            return _unauthorized()
        return await call_next(request)

    @app.websocket("/v1/realtime")
    async def realtime_endpoint(websocket: WebSocket) -> None:
        if realtime is None:
            await websocket.close(code=1011)
            return
        if auth_env is not None:
            expected = os.environ.get(auth_env, "")
            token = _ws_token(websocket)
            if not expected or not hmac.compare_digest(token, expected):
                await websocket.close(code=1008)
                return

        await websocket.accept()
        client = StarletteWebSocketConnection(websocket)
        try:
            await realtime.handle(client, f"realtime:{uuid4().hex}")
        except WebSocketDisconnect:
            pass
        finally:
            await client.close()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": "graph-agent", "object": "model", "owned_by": "graph-agent"}],
        }

    async def _sse_stream(stream: AsyncIterator[Event], model: str, session_id: str) -> Any:
        completion_id = f"chatcmpl-{uuid4().hex}"
        created = int(time.time())

        def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield chunk({"role": "assistant", "content": ""})

        approval: ApprovalRequestEvent | None = None
        async for event in stream:
            if isinstance(event, TokenEvent):
                yield chunk({"content": event.delta})
            elif isinstance(event, ErrorEvent):
                yield chunk({"content": f"\n[error] {event.message}"})
            elif isinstance(event, FileEvent):
                yield chunk({"content": _file_note(event)})
            elif isinstance(event, ApprovalRequestEvent):
                approval = event

        if approval is not None:
            yield chunk({"content": _approval_suffix(approval, session_id)})

        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        identifier = request.session_id or DEFAULT_SESSION_ID
        user_input = request.messages[-1].content

        if user_input.strip() == NEW_COMMAND:
            thread_id = await sessions.reset(CHANNEL, identifier)
            text = f"[new session] {thread_id}"
            if request.stream:
                return StreamingResponse(
                    _sse_stream(_single_token_stream(text), request.model, identifier),
                    media_type="text/event-stream",
                )
            return _completion_body(request.model, text, None, identifier)

        thread_id = await sessions.thread_id(CHANNEL, identifier)

        if request.stream:
            stream = service.run(
                thread_id,
                user_input,
                approval_mode="auto",
                system_prompt=HTTP_SYSTEM_PROMPT,
            )
            return StreamingResponse(
                _sse_stream(stream, request.model, identifier),
                media_type="text/event-stream",
            )

        text, approval = await _collect_stream(
            service.run(
                thread_id,
                user_input,
                approval_mode="auto",
                system_prompt=HTTP_SYSTEM_PROMPT,
            )
        )
        return _completion_body(request.model, text, approval, identifier)

    @app.post("/v1/sessions/{session_id}/approvals/{approval_id}")
    async def resolve_approval(
        session_id: str, approval_id: str, request: ResumeRequest
    ) -> dict[str, Any]:
        thread_id = await sessions.thread_id(CHANNEL, session_id)
        pending = await service.pending_approval(thread_id)
        if pending is None or pending.approval_id != approval_id:
            raise HTTPException(
                status_code=404,
                detail=f"no pending approval {approval_id} for session {session_id}",
            )

        text, next_approval = await _collect_stream(
            service.resume(thread_id, approval_id, request.approved)
        )
        return _completion_body("graph-agent", text, next_approval, session_id)

    return app
