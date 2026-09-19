from __future__ import annotations

from typing import Any

import httpx

from graph_agent.config import WebSearchToolConfig
from graph_agent.tools.base import ToolContext, ToolSpec


class WebSearchTool:
    """Web search via external provider (default: self-hosted SearXNG instance).

    To implement (Fase 1): normalize provider results to [{title, url, snippet}].
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
        async with httpx.AsyncClient(transport=self._transport, timeout=10.0) as client:
            response = await client.get(
                f"{base}/search",
                params={"q": query, "format": "json"},
            )
            response.raise_for_status()
            body = response.json()

        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content") or "",
            }
            for r in body.get("results", [])[:max_results]
        ]
