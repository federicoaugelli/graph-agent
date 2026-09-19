from __future__ import annotations

import sqlite3

from conftest import fake_text_response
from graph_agent.config import AppConfig
from graph_agent.core.service import AgentService
from graph_agent.core.sessions import SessionManager


class FakeStore:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)


async def test_thread_id_is_stable() -> None:
    store = FakeStore()
    sessions = SessionManager(store)

    first = await sessions.thread_id("cli", "local")
    second = await sessions.thread_id("cli", "local")

    assert first == "cli:local:1"
    assert second == first
    assert store.deleted == []


async def test_reset_bumps_generation_and_purges_previous() -> None:
    store = FakeStore()
    sessions = SessionManager(store)
    first = await sessions.thread_id("cli", "local")

    second = await sessions.reset("cli", "local")

    assert first == "cli:local:1"
    assert second == "cli:local:2"
    assert store.deleted == ["cli:local:1"]


async def test_reset_from_cold_purges_target_thread() -> None:
    store = FakeStore()
    sessions = SessionManager(store)

    assert await sessions.reset("cli", "local") == "cli:local:1"
    assert store.deleted == ["cli:local:1"]


async def test_ttl_expiry_resets() -> None:
    store = FakeStore()
    now = [0.0]
    sessions = SessionManager(store, ttl_minutes=1, clock=lambda: now[0])
    assert await sessions.thread_id("cli", "local") == "cli:local:1"

    now[0] = 61.0

    assert await sessions.thread_id("cli", "local") == "cli:local:2"
    assert store.deleted == ["cli:local:1"]


async def test_no_ttl_keeps_thread_id() -> None:
    store = FakeStore()
    now = [0.0]
    sessions = SessionManager(store, clock=lambda: now[0])
    assert await sessions.thread_id("cli", "local") == "cli:local:1"

    now[0] = 10_000.0

    assert await sessions.thread_id("cli", "local") == "cli:local:1"


async def test_delete_thread_purges_checkpoints(app_config: AppConfig) -> None:
    service = AgentService(app_config)
    await service.setup(fake_text_response("ok"))
    await service.set_approval_mode("drop:1", "auto")
    await service.set_approval_mode("keep:1", "auto")

    await service.delete_thread("drop:1")
    await service.shutdown()

    con = sqlite3.connect(app_config.agent.checkpointer_db)
    try:
        dropped = con.execute(
            "SELECT count(*) FROM checkpoints WHERE thread_id = ?", ("drop:1",)
        ).fetchone()[0]
        kept = con.execute(
            "SELECT count(*) FROM checkpoints WHERE thread_id = ?", ("keep:1",)
        ).fetchone()[0]
    finally:
        con.close()

    assert dropped == 0
    assert kept > 0