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
            description=(
                "Search the web and return top results with title, url and snippet. "
                "Prefer batching several query angles in one call via 'queries' "
                "instead of issuing many one-query calls."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extra query variants fetched in the same call.",
                    },
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

        queries = self._queries(args)
        max_results = min(max(int(args.get("max_results", 5)), 1), 10)
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=self.config.timeout_seconds,
            headers=headers,
        ) as client:
            if len(queries) == 1:
                batches = [await self._search_searxng(client, queries[0], max_results)]
            else:
                batches = await self._search_batch(client, queries, max_results)

        return self._merge(batches, ctx, max_results * len(queries))

    def _queries(self, args: dict[str, Any]) -> list[str]:
        queries: list[str] = []
        single = args.get("query")
        if isinstance(single, str) and single.strip():
            queries.append(single.strip())
        batch = args.get("queries")
        if isinstance(batch, list):
            queries.extend(str(item).strip() for item in batch if str(item).strip())
        unique = list(dict.fromkeys(queries))
        if not unique:
            raise ValueError("web_search: query must not be empty")
        return unique[: self.config.max_queries_per_call]

    async def _search_batch(
        self, client: httpx.AsyncClient, queries: list[str], max_results: int
    ) -> list[list[dict[str, str]]]:
        settled = await asyncio.gather(
            *(self._search_searxng(client, query, max_results) for query in queries),
            return_exceptions=True,
        )
        batches: list[list[dict[str, str]]] = []
        errors: list[BaseException] = []
        for item in settled:
            if isinstance(item, BaseException):
                errors.append(item)
            else:
                batches.append(item)
        if not batches and errors:
            raise errors[0]
        return batches

    def _merge(
        self, batches: list[list[dict[str, str]]], ctx: ToolContext, max_total: int
    ) -> list[dict[str, str]]:
        seen = set(ctx.state.seen_urls) if ctx.state is not None else set()
        merged: list[dict[str, str]] = []
        for batch in batches:
            for item in batch:
                url = item["url"]
                if url and url in seen:
                    continue
                if url:
                    seen.add(url)
                merged.append(item)
                if max_total > 0 and len(merged) >= max_total:
                    self._remember(ctx, seen)
                    return merged
        self._remember(ctx, seen)
        return merged

    def _remember(self, ctx: ToolContext, seen: set[str]) -> None:
        if ctx.state is not None:
            ctx.state.seen_urls.update(seen)

    async def _search_searxng(
        self, client: httpx.AsyncClient, query: str, max_results: int
    ) -> list[dict[str, str]]:
        base = str(self.config.api_base).rstrip("/")
        response = await self._get_with_retry(
            client,
            f"{base}/search",
            params={"q": query, "format": "json"},
        )
        body = response.json()

        results = body.get("results", [])
        if not results:
            raise RuntimeError(self._no_results_message(body))

        normalized: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for result in results:
            url = result.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            snippet = result.get("content") or ""
            normalized.append(
                {
                    "title": result.get("title", ""),
                    "url": url,
                    "snippet": snippet[: self.config.max_snippet_chars],
                }
            )
            if len(normalized) >= max_results:
                break

        return normalized

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
