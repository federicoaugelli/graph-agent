from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from graph_agent.config import WebFetchToolConfig
from graph_agent.tools.base import ToolContext
from graph_agent.tools.webfetch import WebFetchTool


def build_tool(
    handler: Any,
    captured: dict[str, Any],
    *,
    allow_private_hosts: bool = False,
    timeout_seconds: float = 15.0,
    max_bytes: int = 2_000_000,
    max_chars: int = 8000,
) -> WebFetchTool:
    def routed(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return handler(request)

    transport = httpx.MockTransport(routed)
    config = WebFetchToolConfig(
        enabled=True,
        timeout_seconds=timeout_seconds,
        max_bytes=max_bytes,
        max_chars=max_chars,
        allow_private_hosts=allow_private_hosts,
    )
    return WebFetchTool(config, transport=transport)


def html_handler(body: str, *, status: int = 200, content_type: str = "text/html") -> Any:
    return lambda request: httpx.Response(
        status, content=body.encode(), headers={"content-type": content_type}
    )


def make_ctx(config: Any) -> ToolContext:
    return ToolContext(session_id="t", workspace=Path("/tmp"), config=config)


async def test_spec_shape() -> None:
    tool = WebFetchTool(WebFetchToolConfig(enabled=True))
    assert tool.spec.name == "web_fetch"
    assert tool.spec.requires_approval is False
    assert "url" in tool.spec.parameters["required"]


async def test_extracts_title_and_readable_text(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}
    body = (
        "<html><head><title>Hello Page</title>"
        "<style>.x{color:red}</style></head>"
        "<body><script>alert(1)</script><h1>Title</h1><p>First</p><p>Second</p></body></html>"
    )
    tool = build_tool(html_handler(body), captured)

    result = await tool.execute({"url": "https://example.com/page"}, make_ctx(minimal_config))

    assert captured["request"].url.path == "/page"
    assert result["url"] == "https://example.com/page"
    assert result["title"] == "Hello Page"
    assert "First" in result["text"]
    assert "Second" in result["text"]
    assert "alert" not in result["text"]
    assert ".x" not in result["text"]


async def test_collapses_whitespace(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}
    body = "<html><body><p>a</p>\n\n\n   <p>b</p></body></html>"
    tool = build_tool(html_handler(body), captured)

    result = await tool.execute({"url": "https://example.com/"}, make_ctx(minimal_config))

    assert "a b" in result["text"]
    assert "\n\n" not in result["text"]


async def test_truncates_to_max_chars(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}
    body = "<html><body>" + ("word " * 5000) + "</body></html>"
    tool = build_tool(html_handler(body), captured, max_chars=100)

    result = await tool.execute({"url": "https://example.com/"}, make_ctx(minimal_config))

    assert len(result["text"]) <= 100


async def test_caps_download_bytes(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}
    body = "<html><body>" + ("z" * 10_000) + "ENDOFPAGE</body></html>"
    tool = build_tool(html_handler(body), captured, max_bytes=200, max_chars=10_000)

    result = await tool.execute({"url": "https://example.com/"}, make_ctx(minimal_config))

    assert "ENDOFPAGE" not in result["text"]


async def test_rejects_non_http_scheme(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}
    tool = build_tool(html_handler("<html></html>"), captured)

    with pytest.raises(ValueError, match="scheme"):
        await tool.execute({"url": "file:///etc/passwd"}, make_ctx(minimal_config))

    assert "request" not in captured


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/admin",
        "http://127.0.0.1:4000/v1/models",
        "http://10.0.0.5/",
        "http://192.168.1.10/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
    ],
)
async def test_blocks_private_and_loopback_hosts(minimal_config: Any, url: str) -> None:
    captured: dict[str, Any] = {}
    tool = build_tool(html_handler("<html></html>"), captured)

    with pytest.raises(ValueError, match="private"):
        await tool.execute({"url": url}, make_ctx(minimal_config))

    assert "request" not in captured


async def test_allows_private_hosts_when_enabled(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}
    tool = build_tool(
        html_handler("<html><body>ok</body></html>"), captured, allow_private_hosts=True
    )

    result = await tool.execute({"url": "http://127.0.0.1:4000/"}, make_ctx(minimal_config))

    assert "ok" in result["text"]


async def test_http_error_propagates(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}
    tool = build_tool(html_handler("<html></html>", status=500), captured)

    with pytest.raises(httpx.HTTPStatusError):
        await tool.execute({"url": "https://example.com/"}, make_ctx(minimal_config))


async def test_rejects_unsupported_content_type(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}
    tool = build_tool(
        html_handler("%PDF-1.7", content_type="application/pdf"),
        captured,
    )

    with pytest.raises(ValueError, match="content type"):
        await tool.execute({"url": "https://example.com/doc.pdf"}, make_ctx(minimal_config))


async def test_sends_user_agent(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}
    tool = build_tool(html_handler("<html><body>ok</body></html>"), captured)

    await tool.execute({"url": "https://example.com/"}, make_ctx(minimal_config))

    assert "graph-agent" in captured["request"].headers["user-agent"]


async def test_follows_redirect(minimal_config: Any) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(302, headers={"location": "https://example.com/new"})
        return httpx.Response(
            200, content=b"<html><body>moved</body></html>", headers={"content-type": "text/html"}
        )

    tool = build_tool(handler, captured)

    result = await tool.execute({"url": "https://example.com/old"}, make_ctx(minimal_config))

    assert "moved" in result["text"]
    assert result["url"] == "https://example.com/new"


def test_defaults() -> None:
    config = WebFetchToolConfig()
    assert config.timeout_seconds == 15.0
    assert config.allow_private_hosts is False
