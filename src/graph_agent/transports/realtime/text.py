from __future__ import annotations

import re

_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_BOLD = re.compile(r"(\*\*|__)(.+?)\1")
_ITALIC = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)|(?<!_)_(?!\s)(.+?)(?<!\s)_(?!_)")
_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_HR = re.compile(r"^\s*(?:[-*_]\s*){3,}$", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_LEFTOVER_BRACKETS = re.compile(r"[\[\]]")
_EMOJI = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "\U00002190-\U000021ff"
    "\U00002b00-\U00002bff"
    "\U0000fe00-\U0000fe0f"
    "\U0000200d"
    "\U000020e3"
    "\U000024c2"
    "\U0000fe0f"
    "]+",
    flags=re.UNICODE,
)


def normalize_for_speech(text: str) -> str:
    """Convert markdown-ish LLM output into plain prose suitable for TTS.

    Strips formatting markers, links, emojis and code fences so a realtime model
    or a TTS engine does not read them aloud (``**bold**``, ``#``, ``- item``...).
    """
    if not text:
        return ""

    out = text.replace("```", " ")
    out = _IMAGE.sub(r"\1", out)
    out = _MD_LINK.sub(r"\1", out)
    out = _INLINE_CODE.sub(r"\1", out)
    out = _HEADER.sub("", out)
    out = _BLOCKQUOTE.sub("", out)
    out = _HR.sub(" ", out)
    out = _BULLET.sub("", out)
    out = _BOLD.sub(r"\2", out)
    out = _ITALIC.sub(lambda match: match.group(1) or match.group(2) or "", out)
    out = _EMOJI.sub("", out)
    out = out.replace("*", "").replace("_", " ").replace("`", "").replace("#", "")
    out = _LEFTOVER_BRACKETS.sub("", out)
    out = re.sub(r"[ \t]*\n+[ \t]*", ". ", out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    out = re.sub(r"\.\s*\.", ".", out)
    return out.strip()
