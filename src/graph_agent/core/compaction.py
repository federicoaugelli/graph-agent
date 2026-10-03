from __future__ import annotations

from langchain_core.messages import BaseMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from graph_agent.models.llm import LLMBackend

SUMMARY_INSTRUCTIONS = (
    "Summarize the conversation so far for your own future reference. Preserve "
    "durable facts, user preferences, decisions, open tasks and any detail needed "
    "to continue. Be concise. Reply with the summary only."
)

SUMMARY_PREFIX = "[Conversation summary]\n"


async def compact(
    backend: LLMBackend,
    messages: list[BaseMessage],
    *,
    token_limit: int,
    keep: int,
) -> list[BaseMessage] | None:
    """Summarize the oldest messages once the context exceeds ``token_limit``.

    Older messages are replaced by a single summary (produced by the same model)
    while the last ``keep`` messages are kept verbatim. The split never lands on a
    ``ToolMessage`` so tool calls stay paired with their results. Returns None when
    there is nothing to compact.
    """
    if token_limit <= 0 or count_tokens_approximately(messages) <= token_limit:
        return None

    split = len(messages) - max(keep, 0)
    while 0 < split < len(messages) and isinstance(messages[split], ToolMessage):
        split -= 1
    if split <= 0:
        return None

    old, recent = messages[:split], messages[split:]
    response = await backend.acomplete([SystemMessage(content=SUMMARY_INSTRUCTIONS), *old])
    summary = SystemMessage(content=f"{SUMMARY_PREFIX}{response.delta_text.strip()}")
    return [RemoveMessage(id=REMOVE_ALL_MESSAGES), summary, *recent]
