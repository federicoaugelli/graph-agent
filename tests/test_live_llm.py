from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from graph_agent.config import load_config
from graph_agent.models.llm import OpenAICompatBackend

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_TESTS") != "1",
        reason="live tests need a running model endpoint: RUN_LIVE_TESTS=1",
    ),
]

ECHO_TOOL = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Echo back the given text.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}


@pytest.fixture(scope="module")
def live_backend() -> OpenAICompatBackend:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config.yaml")
    return OpenAICompatBackend(config.models.brain)


async def test_live_text_completion(live_backend: OpenAICompatBackend) -> None:
    result = await live_backend.acomplete([HumanMessage(content="Reply only with: OK")])
    assert result.delta_text.strip()
    assert result.finish_reason == "stop"


async def test_live_tool_roundtrip(live_backend: OpenAICompatBackend) -> None:
    messages: list[BaseMessage] = [
        HumanMessage(content="Usa il tool echo per ripetere esattamente la parola ping.")
    ]
    first = await live_backend.acomplete(messages, tools=[ECHO_TOOL])
    if not first.tool_calls:
        pytest.skip(f"model answered without calling the tool: {first.delta_text!r}")

    call = first.tool_calls[0]
    assert call.name == "echo"
    messages.append(
        AIMessage(
            content=first.delta_text,
            tool_calls=[{"id": call.id, "name": call.name, "args": call.args, "type": "tool_call"}],
        )
    )
    messages.append(ToolMessage(content=str(call.args.get("text", "")), tool_call_id=call.id))

    second = await live_backend.acomplete(messages)
    assert "ping" in second.delta_text.lower()
