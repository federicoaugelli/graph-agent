from __future__ import annotations

import asyncio
from collections import defaultdict

from graph_agent.events import Event


class EventBus:
    """In-process pub/sub: transports subscribe to topics, the core publishes events."""

    def __init__(self) -> None:
        self._topics: dict[str, list[asyncio.Queue[Event]]] = defaultdict(list)

    def subscribe(self, topic: str) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue()
        self._topics[topic].append(queue)
        return queue

    def unsubscribe(self, topic: str, queue: asyncio.Queue[Event]) -> None:
        subscribers = self._topics.get(topic, [])
        if queue in subscribers:
            subscribers.remove(queue)

    async def publish(self, topic: str, event: Event) -> None:
        for queue in self._topics.get(topic, []):
            await queue.put(event)

    async def publish_all(self, topic: str, events: list[Event]) -> None:
        for event in events:
            await self.publish(topic, event)
