from __future__ import annotations

from graph_agent.config import RealtimeChannelConfig
from graph_agent.models.realtime import build_openai_url, build_realtime_backend
from graph_agent.models.realtime_qwen import (
    QwenRealtimeBackend,
    build_qwen_url,
    from_qwen,
    to_qwen,
)
from graph_agent.transports.realtime.bridge import DELEGATE_TOOL


def test_realtime_config_defaults() -> None:
    config = RealtimeChannelConfig()
    assert config.backend == "openai"
    assert config.api_base == "http://localhost:4000"
    assert config.api_key_env is None


def test_build_openai_url_from_base() -> None:
    config = RealtimeChannelConfig(api_base="http://localhost:4000", model="gpt-realtime")
    assert build_openai_url(config) == "ws://localhost:4000/v1/realtime?model=gpt-realtime"


def test_build_openai_url_keeps_existing_realtime_path() -> None:
    config = RealtimeChannelConfig(api_base="https://example.test/v1/realtime", model="m")
    assert build_openai_url(config) == "wss://example.test/v1/realtime?model=m"


def test_build_qwen_url_uses_full_api_base() -> None:
    config = RealtimeChannelConfig(
        backend="qwen",
        api_base="wss://ws123.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime",
        model="qwen-audio-3.0-realtime-plus",
    )
    assert build_qwen_url(config) == (
        "wss://ws123.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime"
        "?model=qwen-audio-3.0-realtime-plus"
    )


def test_factory_selects_backend() -> None:
    backend = build_realtime_backend(RealtimeChannelConfig(backend="qwen"))
    assert isinstance(backend, QwenRealtimeBackend)


def test_to_qwen_adapts_session_update() -> None:
    event = {
        "type": "session.update",
        "session": {
            "instructions": "be brief",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "tools": [DELEGATE_TOOL],
        },
    }
    translated = to_qwen(event)
    session = translated["session"]
    assert translated["type"] == "session.update"
    assert session["instructions"] == "be brief"
    assert session["input_audio_format"] == "pcm"
    assert session["output_audio_format"] == "pcm"
    assert session["tools"][0]["function"]["name"] == "delegate_to_brain"
    assert session["tools"][0]["function"]["parameters"] == DELEGATE_TOOL["parameters"]


def test_to_qwen_disables_search_when_tools_present() -> None:
    event = {
        "type": "session.update",
        "session": {"enable_search": True, "tools": [DELEGATE_TOOL]},
    }
    session = to_qwen(event)["session"]
    assert session["enable_search"] is False


def test_to_qwen_passthrough_for_other_events() -> None:
    event = {"type": "input_audio_buffer.append", "audio": "AAAA"}
    assert to_qwen(event) == event


def test_from_qwen_is_identity_per_event() -> None:
    message = {"type": "response.audio.delta", "delta": "AAAA"}
    assert from_qwen(message) == [message]
