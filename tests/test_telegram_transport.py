from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from aiogram.types import CallbackQuery, FSInputFile, InlineKeyboardMarkup, Message

from conftest import ScriptedLLMBackend
from graph_agent.config import AppConfig, TelegramChannelConfig
from graph_agent.core.service import AgentService
from graph_agent.core.sessions import SessionManager
from graph_agent.models.llm import StreamChunk, ToolCallRequest
from graph_agent.tools.base import ToolContext, ToolSpec
from graph_agent.transports.telegram import TelegramBot
from graph_agent.transports.telegram.bot import APPROVE_PREFIX, DENY_PREFIX

TOKEN = "42:TEST"
CHAT_ID = 7
USER_ID = 7


class ConfirmTool:
    def __init__(self) -> None:
        self.executed: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="confirm",
            description="Do something dangerous.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            requires_approval=True,
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        self.executed.append(args)
        return args["text"]


class RecordingBot:
    """Stands in for aiogram Bot methods; handlers must use self.bot.* calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _make_message(self, text: str) -> Message:
        return make_message(text)

    async def send_message(self, *args: Any, **kwargs: Any) -> Message:
        entry = self._record("send", args, kwargs)
        return self._make_message(entry["text"])

    async def edit_message_text(self, *args: Any, **kwargs: Any) -> Message:
        self._record("edit", args, kwargs)
        return self._make_message("")

    async def answer_callback_query(self, *args: Any, **kwargs: Any) -> bool:
        self._record("answer", args, kwargs)
        return True

    async def send_document(self, *args: Any, **kwargs: Any) -> Message:
        self._record("send_document", args, kwargs)
        return self._make_message("")

    async def download(self, *args: Any, **kwargs: Any) -> Any:
        entry = self._record("download", args, kwargs)
        destination = entry["destination"]
        if destination is not None:
            Path(destination).write_bytes(b"payload")
        return destination

    def _record(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "method": method,
            "chat_id": kwargs.get("chat_id"),
            "text": kwargs.get("text", ""),
            "reply_markup": kwargs.get("reply_markup"),
            "document": kwargs.get("document"),
            "caption": kwargs.get("caption"),
            "destination": kwargs.get("destination"),
        }
        self.calls.append(entry)
        return entry


def make_message(text: str, chat_id: int = CHAT_ID, user_id: int = USER_ID) -> Message:
    return Message.model_validate(
        {
            "message_id": 55,
            "date": "2024-01-01T00:00:00Z",
            "chat": {"id": chat_id, "type": "private"},
            "from_user": {"id": user_id, "is_bot": False, "first_name": "Tester"},
            "text": text,
        }
    )


def make_document_message(
    file_name: str = "note.txt",
    chat_id: int = CHAT_ID,
    user_id: int = USER_ID,
    caption: str | None = None,
) -> Message:
    return Message.model_validate(
        {
            "message_id": 57,
            "date": "2024-01-01T00:00:00Z",
            "chat": {"id": chat_id, "type": "private"},
            "from_user": {"id": user_id, "is_bot": False, "first_name": "Tester"},
            "caption": caption,
            "document": {
                "file_id": "doc-1",
                "file_unique_id": "doc-unique",
                "file_name": file_name,
            },
        }
    )


def make_photo_message(chat_id: int = CHAT_ID, user_id: int = USER_ID) -> Message:
    return Message.model_validate(
        {
            "message_id": 58,
            "date": "2024-01-01T00:00:00Z",
            "chat": {"id": chat_id, "type": "private"},
            "from_user": {"id": user_id, "is_bot": False, "first_name": "Tester"},
            "photo": [
                {
                    "file_id": "photo-1",
                    "file_unique_id": "photo-unique",
                    "width": 90,
                    "height": 90,
                }
            ],
        }
    )


def make_callback(data: str, user_id: int = USER_ID, chat_id: int = CHAT_ID) -> CallbackQuery:
    return CallbackQuery.model_validate(
        {
            "id": "cb-1",
            "chat_instance": "inst",
            "from_user": {"id": user_id, "is_bot": False, "first_name": "Tester"},
            "message": {
                "message_id": 56,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": chat_id, "type": "private"},
            },
            "data": data,
        }
    )


def text_chunks(text: str) -> list[StreamChunk]:
    return [StreamChunk(delta_text=text, finish_reason="stop")]


def approval_script() -> list[list[StreamChunk]]:
    return [
        [
            StreamChunk(
                tool_calls=[ToolCallRequest(id="call_1", name="confirm", args={"text": "boom"})],
                finish_reason="tool_calls",
            )
        ],
        text_chunks("end"),
    ]


@pytest.fixture
def telegram_config() -> TelegramChannelConfig:
    return TelegramChannelConfig(allowed_user_ids=[USER_ID])


@pytest.fixture
async def bot_env(
    app_config: AppConfig,
    telegram_config: TelegramChannelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    backend = ScriptedLLMBackend(
        [[StreamChunk(delta_text="Hello "), StreamChunk(delta_text="world", finish_reason="stop")]]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    bot = TelegramBot(service, telegram_config, SessionManager(service))
    recorder = RecordingBot()
    bot.bot = recorder  # type: ignore[assignment]
    yield bot, recorder, backend, service
    await service.shutdown()


async def test_is_allowed_respects_allowlist(bot_env: Any) -> None:
    bot, _, _, _ = bot_env
    assert bot.is_allowed(USER_ID) is True
    assert bot.is_allowed(999) is False

    bot.config = TelegramChannelConfig(allowed_user_ids=[])
    assert bot.is_allowed(1234) is False


async def test_disallowed_user_gets_silence(bot_env: Any) -> None:
    bot, recorder, backend, _ = bot_env

    await bot.handle_message(make_message("hello", user_id=999))

    assert recorder.calls == []
    assert backend.calls == []


async def test_empty_allowlist_denies_everyone(bot_env: Any) -> None:
    bot, recorder, backend, _ = bot_env
    bot.config = TelegramChannelConfig(allowed_user_ids=[])

    await bot.handle_message(make_message("hello"))

    assert recorder.calls == []
    assert backend.calls == []


async def test_message_streams_final_answer(bot_env: Any) -> None:
    bot, recorder, backend, _ = bot_env

    await bot.handle_message(make_message("saluta"))

    assert recorder.calls, "handler must reply via bot.send_message/edit_message_text"
    assert recorder.calls[-1]["text"] == "Hello world"
    assert recorder.calls[-1]["chat_id"] == CHAT_ID
    assert any(call["method"] == "edit" for call in recorder.calls), "use edits for streaming"

    history = [message.content for message in backend.calls[-1]]
    assert "saluta" in history


async def test_telegram_session_is_chat_id_and_persists(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    backend = ScriptedLLMBackend(
        [text_chunks("prima"), text_chunks("seconda"), text_chunks("terza")]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    bot = TelegramBot(
        service, TelegramChannelConfig(allowed_user_ids=[USER_ID]), SessionManager(service)
    )
    recorder = RecordingBot()
    bot.bot = recorder  # type: ignore[assignment]

    await bot.handle_message(make_message("uno", chat_id=4242))
    await bot.handle_message(make_message("due", chat_id=4242))
    await bot.handle_message(make_message("tre", chat_id=9999))

    second_turn = [m.content for m in backend.calls[1]]
    assert "uno" in second_turn
    assert "prima" in second_turn
    assert "due" in second_turn

    third_turn = [m.content for m in backend.calls[2]]
    assert "tre" in third_turn
    assert "uno" not in third_turn

    await service.shutdown()


async def make_approval_bot(app_config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    tool = ConfirmTool()
    backend = ScriptedLLMBackend(approval_script())
    service = AgentService(app_config)
    await service.setup(backend)
    service.registry.register(tool)
    bot = TelegramBot(
        service,
        TelegramChannelConfig(allowed_user_ids=[USER_ID]),
        SessionManager(service),
    )
    recorder = RecordingBot()
    bot.bot = recorder  # type: ignore[assignment]
    return bot, recorder, tool, service


async def test_approval_sends_inline_keyboard(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot, recorder, tool, service = await make_approval_bot(app_config, monkeypatch)

    await bot.handle_message(make_message("do it"))

    keyboard_call = next(c for c in recorder.calls if c["reply_markup"] is not None)
    markup = keyboard_call["reply_markup"]
    assert isinstance(markup, InlineKeyboardMarkup)
    datas = [
        btn.callback_data for row in markup.inline_keyboard for btn in row if btn.callback_data
    ]
    assert f"{APPROVE_PREFIX}call_1" in datas
    assert f"{DENY_PREFIX}call_1" in datas
    assert tool.executed == []

    await service.shutdown()


async def test_callback_approve_resumes_and_executes(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot, recorder, tool, service = await make_approval_bot(app_config, monkeypatch)
    await bot.handle_message(make_message("do it"))

    await bot.handle_approval_callback(make_callback(f"{APPROVE_PREFIX}call_1"))

    assert tool.executed == [{"text": "boom"}]
    assert any(call["method"] == "answer" for call in recorder.calls)
    assert recorder.calls[-1]["text"] == "end"

    await service.shutdown()


async def test_callback_deny_refuses_tool(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot, recorder, tool, service = await make_approval_bot(app_config, monkeypatch)
    await bot.handle_message(make_message("do it"))

    await bot.handle_approval_callback(make_callback(f"{DENY_PREFIX}call_1"))

    assert tool.executed == []
    assert any(call["method"] == "answer" for call in recorder.calls)
    assert recorder.calls[-1]["text"] == "end"

    await service.shutdown()


async def test_callback_from_disallowed_user_is_not_resumed(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot, _, tool, service = await make_approval_bot(app_config, monkeypatch)
    await bot.handle_message(make_message("do it"))

    await bot.handle_approval_callback(make_callback(f"{APPROVE_PREFIX}call_1", user_id=999))

    assert tool.executed == []

    await service.shutdown()


async def test_auto_command_sets_session_mode_without_running_agent(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot, recorder, tool, service = await make_approval_bot(app_config, monkeypatch)

    await bot.handle_message(make_message("/auto"))

    assert len(recorder.calls) == 1
    assert "auto" in recorder.calls[0]["text"].lower()
    assert "manual" not in recorder.calls[0]["text"].lower()
    assert tool.executed == []

    await bot.handle_message(make_message("do it"))

    assert tool.executed == [{"text": "boom"}]
    assert all(call["reply_markup"] is None for call in recorder.calls)
    assert recorder.calls[-1]["text"] == "end"

    await service.shutdown()


async def test_manual_command_reenables_approvals(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot, recorder, tool, service = await make_approval_bot(app_config, monkeypatch)
    await bot.handle_message(make_message("/auto"))

    recorder.calls.clear()
    await bot.handle_message(make_message("/manual"))

    assert "manual" in recorder.calls[-1]["text"].lower()
    assert "auto" not in recorder.calls[-1]["text"].lower()

    await bot.handle_message(make_message("do it"))

    assert tool.executed == []
    assert any(call["reply_markup"] is not None for call in recorder.calls)

    await service.shutdown()


async def test_approval_mode_is_per_chat(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot, recorder, tool, service = await make_approval_bot(app_config, monkeypatch)
    await bot.handle_message(make_message("/auto", chat_id=111, user_id=USER_ID))

    await bot.handle_message(make_message("do it", chat_id=222, user_id=USER_ID))

    assert tool.executed == []
    assert any(call["reply_markup"] is not None for call in recorder.calls)

    await service.shutdown()


async def test_new_command_starts_a_fresh_thread(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    backend = ScriptedLLMBackend([text_chunks("prima"), text_chunks("dopo")])
    service = AgentService(app_config)
    await service.setup(backend)
    bot = TelegramBot(
        service, TelegramChannelConfig(allowed_user_ids=[USER_ID]), SessionManager(service)
    )
    recorder = RecordingBot()
    bot.bot = recorder  # type: ignore[assignment]

    await bot.handle_message(make_message("uno", chat_id=4242))
    await bot.handle_message(make_message("/new", chat_id=4242))
    await bot.handle_message(make_message("due", chat_id=4242))

    assert any("new session" in call["text"] for call in recorder.calls)

    last_turn = [message.content for message in backend.calls[-1]]
    assert "due" in last_turn
    assert "uno" not in last_turn
    assert "prima" not in last_turn

    await service.shutdown()


async def test_agent_sends_workspace_file(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    workspace = app_config.agent.workspace
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "report.txt").write_text("data")
    backend = ScriptedLLMBackend(
        [
            [
                StreamChunk(
                    tool_calls=[
                        ToolCallRequest(
                            id="call_send",
                            name="send_file",
                            args={"path": "report.txt", "caption": "ecco"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            ],
            text_chunks("inviato"),
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    bot = TelegramBot(
        service, TelegramChannelConfig(allowed_user_ids=[USER_ID]), SessionManager(service)
    )
    recorder = RecordingBot()
    bot.bot = recorder  # type: ignore[assignment]

    await bot.handle_message(make_message("manda il report"))

    doc_calls = [call for call in recorder.calls if call["method"] == "send_document"]
    assert len(doc_calls) == 1
    assert doc_calls[0]["chat_id"] == CHAT_ID
    assert doc_calls[0]["caption"] == "ecco"
    document = doc_calls[0]["document"]
    assert isinstance(document, FSInputFile)
    assert document.path == str((workspace / "report.txt").resolve())

    await service.shutdown()


async def test_incoming_document_is_saved_and_announced(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    backend = ScriptedLLMBackend([text_chunks("ricevuto")])
    service = AgentService(app_config)
    await service.setup(backend)
    bot = TelegramBot(
        service, TelegramChannelConfig(allowed_user_ids=[USER_ID]), SessionManager(service)
    )
    recorder = RecordingBot()
    bot.bot = recorder  # type: ignore[assignment]

    await bot.handle_file(make_document_message("note.txt", caption="leggi"))

    download_calls = [call for call in recorder.calls if call["method"] == "download"]
    assert len(download_calls) == 1
    destination = Path(download_calls[0]["destination"])
    assert destination == app_config.agent.workspace / "incoming" / str(CHAT_ID) / "note.txt"
    assert destination.exists()

    history = " ".join(str(message.content) for message in backend.calls[-1])
    assert "leggi" in history
    assert "incoming" in history

    await service.shutdown()


async def test_incoming_photo_is_saved_as_jpg(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    backend = ScriptedLLMBackend([text_chunks("ok")])
    service = AgentService(app_config)
    await service.setup(backend)
    bot = TelegramBot(
        service, TelegramChannelConfig(allowed_user_ids=[USER_ID]), SessionManager(service)
    )
    recorder = RecordingBot()
    bot.bot = recorder  # type: ignore[assignment]

    await bot.handle_file(make_photo_message())

    download_calls = [call for call in recorder.calls if call["method"] == "download"]
    destination = Path(download_calls[0]["destination"])
    assert destination.name == "photo_photo-unique.jpg"
    assert destination.exists()

    await service.shutdown()


async def test_disallowed_user_file_is_ignored(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    backend = ScriptedLLMBackend([text_chunks("ok")])
    service = AgentService(app_config)
    await service.setup(backend)
    bot = TelegramBot(
        service, TelegramChannelConfig(allowed_user_ids=[USER_ID]), SessionManager(service)
    )
    recorder = RecordingBot()
    bot.bot = recorder  # type: ignore[assignment]

    await bot.handle_file(make_document_message(user_id=999))

    assert recorder.calls == []
    assert backend.calls == []

    await service.shutdown()
