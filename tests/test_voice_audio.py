from __future__ import annotations

from array import array

from graph_agent.transports.voice.audio import resample_pcm16


def _samples(data: bytes) -> array[int]:
    out: array[int] = array("h")
    out.frombytes(data)
    return out


def test_resample_identity_when_rates_match() -> None:
    data = bytes(range(0, 20))
    assert resample_pcm16(data, 8000, 8000) == data


def test_resample_empty_input() -> None:
    assert resample_pcm16(b"", 8000, 16000) == b""


def test_resample_upsample_doubles_sample_count() -> None:
    source = array("h", [0, 100, -100, 200]).tobytes()
    result = _samples(resample_pcm16(source, 8000, 16000))
    assert len(result) == 8


def test_resample_downsample_halves_sample_count() -> None:
    source = array("h", list(range(32))).tobytes()
    result = _samples(resample_pcm16(source, 16000, 8000))
    assert len(result) == 16


def test_resample_interpolates_midpoints() -> None:
    source = array("h", [0, 1000]).tobytes()
    result = _samples(resample_pcm16(source, 8000, 16000))
    assert result[0] == 0
    assert result[1] == 500
    assert result[-1] == 1000


def test_resample_keeps_last_sample_on_short_input() -> None:
    source = array("h", [7]).tobytes()
    assert _samples(resample_pcm16(source, 8000, 16000)) == array("h", [7, 7])
