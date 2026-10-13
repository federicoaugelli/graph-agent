from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

_WORD = re.compile(r"\w+")


def normalize_query(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _tokens(text: str) -> set[str]:
    return set(normalize_query(text).split())


def query_similarity(a: str, b: str) -> float:
    """Similarity of two search queries, biased towards near-duplicate phrasing.

    Uses the max of the Jaccard index and the overlap coefficient so that a query
    that only adds qualifiers to an already-seen one ("x" vs "x 2026") still counts
    as a duplicate. Queries with fewer than two words are never treated as similar.
    """
    left, right = _tokens(a), _tokens(b)
    if len(left) < 2 or len(right) < 2:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    union = len(left | right)
    return max(intersection / union, intersection / min(len(left), len(right)))


def arg_queries(args: dict[str, Any]) -> list[str]:
    """Queries carried by a web_search call: the primary ``query`` plus ``queries``."""
    queries: list[str] = []
    single = args.get("query")
    if isinstance(single, str) and single.strip():
        queries.append(single.strip())
    batch = args.get("queries")
    if isinstance(batch, list):
        queries.extend(str(item).strip() for item in batch if str(item).strip())
    return list(dict.fromkeys(queries))


@dataclass
class RunState:
    """Per-run, in-memory state shared by tool policies and stateful tools.

    Lives on ``RunMetrics`` (a contextvar) so it never reaches the checkpointer.
    Policies that consume a shared budget mutate it under ``lock``: ``ToolNode``
    runs the tool calls of a round concurrently.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    counts: dict[str, int] = field(default_factory=dict)
    search_queries: list[str] = field(default_factory=list)
    seen_urls: set[str] = field(default_factory=set)


@dataclass(slots=True)
class ToolCallInfo:
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class PolicyDecision:
    """Verdict for one tool call.

    ``args`` are the (possibly rewritten) arguments to run with; ``result``, when
    set, short-circuits execution and is returned to the model instead.
    """

    args: dict[str, Any]
    result: str | None = None
    is_error: bool = False

    @property
    def blocks(self) -> bool:
        return self.result is not None


class ToolPolicy(Protocol):
    def applies_to(self, name: str) -> bool: ...

    async def apply(self, call: ToolCallInfo, state: RunState) -> PolicyDecision: ...


class ToolPolicyRegistry:
    """Ordered policies applied to a tool call before it executes.

    Each policy may rewrite the arguments or short-circuit the call; the first
    short-circuit wins. Mirrors ``ToolRegistry`` and is the chokepoint where
    model-based guardrails (system-1) plug in later.
    """

    def __init__(self) -> None:
        self._policies: list[ToolPolicy] = []

    def register(self, policy: ToolPolicy) -> None:
        self._policies.append(policy)

    async def apply(self, call: ToolCallInfo, state: RunState) -> PolicyDecision:
        current = call
        for policy in self._policies:
            if not policy.applies_to(current.name):
                continue
            decision = await policy.apply(current, state)
            current = ToolCallInfo(name=current.name, args=decision.args)
            if decision.blocks:
                return decision
        return PolicyDecision(args=current.args)


class SearchBudgetPolicy:
    """Cap the number of ``web_search`` calls in a single run."""

    def __init__(self, max_calls_per_run: int) -> None:
        self.max_calls = max_calls_per_run

    def applies_to(self, name: str) -> bool:
        return name == "web_search"

    async def apply(self, call: ToolCallInfo, state: RunState) -> PolicyDecision:
        async with state.lock:
            used = state.counts.get("web_search", 0)
            if self.max_calls > 0 and used >= self.max_calls:
                return PolicyDecision(
                    args=call.args,
                    result=(
                        f"web_search budget reached for this turn ({self.max_calls} calls). "
                        "Stop searching and answer with the information you already gathered, "
                        "or ask the user for guidance."
                    ),
                )
            state.counts["web_search"] = used + 1
        return PolicyDecision(args=call.args)


class SearchDedupPolicy:
    """Drop ``web_search`` queries that repeat (approximately) one already issued."""

    def __init__(self, similarity: float) -> None:
        self.similarity = similarity

    def applies_to(self, name: str) -> bool:
        return name == "web_search"

    async def apply(self, call: ToolCallInfo, state: RunState) -> PolicyDecision:
        queries = arg_queries(call.args)
        if not queries:
            return PolicyDecision(args=call.args)

        kept: list[str] = []
        dropped: list[str] = []
        async with state.lock:
            for query in queries:
                if self._is_duplicate(query, state.search_queries):
                    dropped.append(query)
                else:
                    kept.append(query)
                    state.search_queries.append(normalize_query(query))
            if not kept:
                return PolicyDecision(
                    args=call.args,
                    result=(
                        "Already searched this turn for: "
                        + "; ".join(dropped)
                        + ". Do not repeat it: reuse the results already in the "
                        "conversation, rephrase with new terms, or answer now."
                    ),
                )

        return PolicyDecision(args=_with_kept_queries(call.args, kept))

    def _is_duplicate(self, query: str, seen: list[str]) -> bool:
        return any(query_similarity(query, previous) >= self.similarity for previous in seen)


def _with_kept_queries(args: dict[str, Any], kept: list[str]) -> dict[str, Any]:
    updated = dict(args)
    if isinstance(args.get("queries"), list):
        updated["queries"] = kept
    updated["query"] = kept[0]
    return updated
