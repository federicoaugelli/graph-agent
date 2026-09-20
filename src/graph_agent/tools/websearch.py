from __future__ import annotations

import asyncio
from typing import Any

import httpx

from graph_agent.config import WebSearchToolConfig
from graph_agent.tools.base import ToolContext, ToolSpec

_USER_AGENT = "graph-agent/0.1 (+web_search)"
_RETRYABLE_STATUS = {429, 503}
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 0.5


class WebSearchTool:
    """Web search via external provider (default: self-hosted SearXNG instance).

    Normalizes provider results to [{title, url, snippet}]. Raises when the provider
    returns no results so the agent sees a failed search instead of an empty answer.
    """

    def __init__(
        self,
        config: WebSearchToolConfig,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_search",
            description="Search the web and return top results with title, url and snippet.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            requires_approval=False,
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        if self.config.provider != "searxng":
            raise ValueError(f"web_search: unsupported provider: {self.config.provider}")
        if not self.config.api_base:
            raise ValueError("web_search: api_base not configured")

        query = str(args["query"])
        max_results = min(max(int(args.get("max_results", 5)), 1), 10)
        return await self._search_searxng(query, max_results)

    async def _search_searxng(self, query: str, max_results: int) -> list[dict[str, str]]:
        base = str(self.config.api_base).rstrip("/")
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=self.config.timeout_seconds,
            headers=headers,
        ) as client:
            response = await self._get_with_retry(
                client,
                f"{base}/search",
                params={"q": query, "format": "json"},
            )
            body = response.json()

        results = body.get("results", [])
        if not results:
            raise RuntimeError(self._no_results_message(body))

        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content") or "",
            }
            for r in results[:max_results]
        ]

    async def _get_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        params: dict[str, str],
    ) -> httpx.Response:
        response = await client.get(url, params=params)
        for attempt in range(1, _MAX_ATTEMPTS):
            if response.status_code not in _RETRYABLE_STATUS:
                break
            await asyncio.sleep(_RETRY_BACKOFF_SECONDS * attempt)
            response = await client.get(url, params=params)

        response.raise_for_status()
        return response

    def _no_results_message(self, body: dict[str, Any]) -> str:
        unresponsive = body.get("unresponsive_engines") or []
        if not unresponsive:
            return "web_search: provider returned no results"
        detail = ", ".join(f"{engine}: {reason}" for engine, reason in unresponsive)
        return f"web_search: no results, engines unavailable ({detail})"
