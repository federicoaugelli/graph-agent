from __future__ import annotations

import sys
from array import array

PCM16_SAMPLE_BYTES = 2


def resample_pcm16(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample signed 16-bit little-endian mono PCM with linear interpolation.

    Telephony (AudioSocket) carries 8 kHz slin; the realtime backend expects a
    higher-rate PCM stream. Pure Python keeps the transport dependency-free.
    """
    if src_rate == dst_rate or not data:
        return data

    samples = array("h")
    samples.frombytes(data[: len(data) - len(data) % PCM16_SAMPLE_BYTES])
    if sys.byteorder == "big":
        samples.byteswap()

    count = len(samples)
    if count == 0:
        return b""

    out_count = max(1, round(count * dst_rate / src_rate))
    result = array("h", bytes(PCM16_SAMPLE_BYTES * out_count))
    step = src_rate / dst_rate
    last = count - 1
    for index in range(out_count):
        position = index * step
        left = int(position)
        if left >= last:
            result[index] = samples[last]
            continue
        fraction = position - left
        result[index] = int(samples[left] * (1 - fraction) + samples[left + 1] * fraction)

    if sys.byteorder == "big":
        result.byteswap()
    return result.tobytes()
