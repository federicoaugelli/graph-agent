from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from graph_agent.config import WebFetchToolConfig
from graph_agent.tools.base import ToolContext, ToolSpec

_USER_AGENT = "graph-agent/0.1 (+web_fetch)"
_ACCEPT = "text/html,application/xhtml+xml,text/plain;q=0.9"
_ACCEPTED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


class WebFetchTool:
    """Fetch a URL and return its readable text.

    Reads a single page (no crawling), strips markup and returns ``{url, title,
    text}`` with the text capped so a page cannot blow up the model context.
    Refuses non-http(s) URLs and, by default, private/loopback hosts to avoid
    the agent reaching internal services (e.g. the local model proxy).
    """

    def __init__(
        self,
        config: WebFetchToolConfig,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_fetch",
            description=(
                "Fetch a URL and return its readable text (title and content). "
                "Use it to read a page found via web_search instead of searching again."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["url"],
            },
            requires_approval=False,
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        url = str(args["url"]).strip()
        self._validate_url(url)
        max_chars = int(args.get("max_chars") or self.config.max_chars)

        headers = {"User-Agent": _USER_AGENT, "Accept": _ACCEPT}
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=self.config.timeout_seconds,
            headers=headers,
            follow_redirects=True,
        ) as client:
            response = await client.get(url)

        response.raise_for_status()
        content_type = self._content_type(response)
        if not content_type.startswith(_ACCEPTED_CONTENT_TYPES):
            raise ValueError(f"web_fetch: unsupported content type: {content_type!r}")

        body = response.content[: self.config.max_bytes]
        encoding = response.encoding or "utf-8"
        document = body.decode(encoding, errors="replace")

        if content_type.startswith("text/plain"):
            title = ""
            text = " ".join(document.split())
        else:
            soup = BeautifulSoup(document, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""
            text = " ".join(soup.get_text(separator=" ").split())

        return {"url": str(response.url), "title": title, "text": text[:max_chars]}

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"web_fetch: unsupported URL scheme: {parsed.scheme!r}")
        if self.config.allow_private_hosts:
            return

        host = parsed.hostname
        if not host:
            raise ValueError("web_fetch: URL is missing a host")
        if host == "localhost":
            raise ValueError("web_fetch: private/loopback host not allowed: localhost")

        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return

        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_unspecified
        ):
            raise ValueError(f"web_fetch: private/loopback host not allowed: {host}")

    @staticmethod
    def _content_type(response: httpx.Response) -> str:
        return response.headers.get("content-type", "").split(";")[0].strip().lower()
