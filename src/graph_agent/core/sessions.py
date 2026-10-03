from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

SHARED_SESSION_KEY = "shared"


class ThreadStore(Protocol):
    async def delete_thread(self, thread_id: str) -> None: ...


@dataclass(slots=True)
class _Session:
    generation: int
    last_seen: float


class SessionManager:
    """Owns the single shared LangGraph thread used by every transport.

    All channels (CLI, Telegram, HTTP, realtime, voice) talk to the same brain
    session: thread id = "{key}:{generation}". A reset bumps the generation and
    purges the previous thread's checkpoints; an idle TTL resets it automatically.
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
        self._session: _Session | None = None

    async def thread_id(self) -> str:
        now = self._clock()
        session = self._session
        if session is not None and self._expired(session, now):
            return await self.reset()
        if session is None:
            session = _Session(generation=1, last_seen=now)
            self._session = session
        else:
            session.last_seen = now
        return self._thread(session.generation)

    async def reset(self) -> str:
        previous = self._session
        if previous is None:
            generation = 1
            await self.store.delete_thread(self._thread(generation))
        else:
            await self.store.delete_thread(self._thread(previous.generation))
            generation = previous.generation + 1
        self._session = _Session(generation=generation, last_seen=self._clock())
        return self._thread(generation)

    def _expired(self, session: _Session, now: float) -> bool:
        if self.ttl_seconds is None:
            return False
        return now - session.last_seen > self.ttl_seconds

    @staticmethod
    def _thread(generation: int) -> str:
        return f"{SHARED_SESSION_KEY}:{generation}"