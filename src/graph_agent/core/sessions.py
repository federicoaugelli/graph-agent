from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class ThreadStore(Protocol):
    async def delete_thread(self, thread_id: str) -> None: ...


@dataclass(slots=True)
class _Session:
    generation: int
    last_seen: float


class SessionManager:
    """Maps transport sessions to LangGraph thread ids and resets them.

    session key = "{channel}:{identifier}", thread id = "{key}:{generation}".
    A reset bumps the generation and purges the previous thread's checkpoints,
    so a fresh manager (e.g. a CLI process) starts clean by resetting before use.
    """

    def __init__(
        self,
        store: ThreadStore,
        ttl_minutes: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.ttl_seconds = ttl_minutes * 60 if ttl_minutes else None
        self._clock = clock
        self._sessions: dict[str, _Session] = {}

    async def thread_id(self, channel: str, identifier: str) -> str:
        key = self._key(channel, identifier)
        now = self._clock()
        session = self._sessions.get(key)
        if session is not None and self._expired(session, now):
            return await self.reset(channel, identifier)
        if session is None:
            session = _Session(generation=1, last_seen=now)
            self._sessions[key] = session
        else:
            session.last_seen = now
        return self._thread(key, session.generation)

    async def reset(self, channel: str, identifier: str) -> str:
        key = self._key(channel, identifier)
        previous = self._sessions.pop(key, None)
        if previous is None:
            generation = 1
            await self.store.delete_thread(self._thread(key, generation))
        else:
            await self.store.delete_thread(self._thread(key, previous.generation))
            generation = previous.generation + 1
        thread = self._thread(key, generation)
        self._sessions[key] = _Session(generation=generation, last_seen=self._clock())
        return thread

    def _expired(self, session: _Session, now: float) -> bool:
        if self.ttl_seconds is None:
            return False
        return now - session.last_seen > self.ttl_seconds

    @staticmethod
    def _key(channel: str, identifier: str) -> str:
        return f"{channel}:{identifier}"

    @staticmethod
    def _thread(key: str, generation: int) -> str:
        return f"{key}:{generation}"