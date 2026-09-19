from __future__ import annotations

from typing import Any

import httpx
import pytest

from graph_agent.config import WebSearchToolConfig
from graph_agent.tools.base import ToolContext
from graph_agent.tools.websearch import WebSearchTool


def build_tool(
    handler: Any,
    captured: dict[str, Any],
    api_base: str | None = "http://searx.test",
    provider: str = "searxng",
) -> WebSearchTool:
    def routed(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return handler(request)

    transport = httpx.MockTransport(routed)
    config = WebSearchToolConfig(enabled=True, provider=provider, api_base=api_base)
    return WebSearchTool(config, transport=transport)


def searx_payload(count: int) -> dict[str, Any]:
    return {
        "results": [
            {
                "title": f"Risultato {i}",
                "url": f"https://example.com/{i}",
                "content": f"snippet {i}",
            }
            for i in range(count)
        ]
    }


def ok_handler(count: int = 3) -> Any:
    return lambda request: httpx.Response(200, json=searx_payload(count))


async def make_ctx(minimal_config: Any) -> ToolContext:
    from pathlib import Path

    return ToolContext(
        session_id="t",
        workspace=Path("/tmp"),
        config=minimal_config,
    )


async def test_spec_shape(minimal_config: Any) -> None:
    tool = WebSearchTool(WebSearchToolConfig(enabled=True, api_base="http://x"))
    assert tool.spec.name == "web_search"
    assert tool.spec.requires_approval is False
    assert "query" in tool.spec.parameters["required"]


async def test_normalizes_searxng_results(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}
    tool = build_tool(ok_handler(), captured)
    ctx = await make_ctx(minimal_config)

    results = await tool.execute({"query": "langgraph"}, ctx)

    request = captured["request"]
    assert request.url.path == "/search"
    assert request.url.params["q"] == "langgraph"
    assert request.url.params["format"] == "json"

    assert isinstance(results, list)
    assert len(results) == 3
    assert results[0] == {
        "title": "Risultato 0",
        "url": "https://example.com/0",
        "snippet": "snippet 0",
    }


async def test_caps_at_max_results(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}
    tool = build_tool(ok_handler(7), captured)
    ctx = await make_ctx(minimal_config)

    results = await tool.execute({"query": "x", "max_results": 2}, ctx)

    assert len(results) == 2


async def test_default_max_results_is_five(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}
    tool = build_tool(ok_handler(10), captured)
    ctx = await make_ctx(minimal_config)

    results = await tool.execute({"query": "x"}, ctx)

    assert len(results) == 5


async def test_snippet_defaults_to_empty_string(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"title": "t", "url": "u"}]})

    tool = build_tool(handler, captured)
    ctx = await make_ctx(minimal_config)

    results = await tool.execute({"query": "x"}, ctx)

    assert results == [{"title": "t", "url": "u", "snippet": ""}]


async def test_missing_api_base_raises(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}
    tool = build_tool(ok_handler(), captured, api_base=None)
    ctx = await make_ctx(minimal_config)

    with pytest.raises(ValueError, match="api_base"):
        await tool.execute({"query": "x"}, ctx)

    assert "request" not in captured


async def test_unknown_provider_raises(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}
    tool = build_tool(ok_handler(), captured, provider="duckduckgo")
    ctx = await make_ctx(minimal_config)

    with pytest.raises(ValueError, match="provider"):
        await tool.execute({"query": "x"}, ctx)

    assert "request" not in captured


async def test_empty_results_returns_empty_list(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    tool = build_tool(handler, captured)
    ctx = await make_ctx(minimal_config)

    results = await tool.execute({"query": "asdfqwerty12345zz"}, ctx)

    assert results == []


async def test_http_error_propagates(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}
    tool = build_tool(lambda request: httpx.Response(502), captured)
    ctx = await make_ctx(minimal_config)

    with pytest.raises(httpx.HTTPStatusError):
        await tool.execute({"query": "x"}, ctx)
