from __future__ import annotations

import html
import re

TELEGRAM_MESSAGE_LIMIT = 4096

_FENCED = re.compile(r"```([^\n`]*)\n?(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)
_STRIKE = re.compile(r"~~(.+?)~~", re.DOTALL)
_ITALIC = re.compile(
    r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])"
    r"|(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])",
    re.DOTALL,
)
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_QUOTE = re.compile(r"^\s{0,3}>\s?(.*)$")
_ULIST = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_OLIST = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
_HR = re.compile(r"^\s{0,3}([-*_])\s*(?:\1\s*){2,}$")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_TAG_NAME = re.compile(r"</?([a-zA-Z]+)")


def escape_html(text: str) -> str:
    """Escape the three characters Telegram's HTML parser treats specially."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def html_to_plain(text: str) -> str:
    """Strip generated tags and entities, for a readable plain-text fallback."""
    return html.unescape(_TAG.sub("", text))


def markdown_to_telegram_html(text: str) -> str:
    """Convert a markdown-ish agent reply into the Telegram HTML subset.

    Supports bold, italic, strikethrough, inline/block code, links, headings,
    lists and blockquotes. Everything else is escaped and left as plain text.
    """
    blocks: list[str] = []

    def stash_block(match: re.Match[str]) -> str:
        blocks.append(_render_code_block(match))
        return f"\x00B{len(blocks) - 1}\x00"

    text = _FENCED.sub(stash_block, text)

    rendered: list[str] = []
    quote: list[str] = []

    def flush_quote() -> None:
        if quote:
            rendered.append("<blockquote>" + "\n".join(quote) + "</blockquote>")
            quote.clear()

    for line in text.split("\n"):
        quote_match = _QUOTE.match(line)
        if quote_match:
            quote.append(_inline(quote_match.group(1)))
            continue
        flush_quote()
        if _HR.match(line):
            rendered.append("———")
            continue
        heading_match = _HEADING.match(line)
        if heading_match:
            content = _inline(heading_match.group(2))
            rendered.append(f"<b>{content}</b>" if content else "")
            continue
        ul_match = _ULIST.match(line)
        if ul_match:
            rendered.append(f"{ul_match.group(1)}• {_inline(ul_match.group(2))}")
            continue
        ol_match = _OLIST.match(line)
        if ol_match:
            rendered.append(f"{ol_match.group(1)}{ol_match.group(2)}. {_inline(ol_match.group(3))}")
            continue
        rendered.append(_inline(line))
    flush_quote()

    result = "\n".join(rendered)
    for index, block in enumerate(blocks):
        result = result.replace(f"\x00B{index}\x00", block, 1)
    return result


def split_telegram_html(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split an HTML message into chunks of at most `limit` characters.

    Splitting only happens between tokens, and any tag left open at a boundary
    is closed and reopened in the next chunk, so every chunk stays parseable.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    open_tags: list[str] = []

    def flush() -> None:
        nonlocal length
        chunks.append("".join(current) + _closing(open_tags))
        current.clear()
        current.extend(open_tags)
        length = sum(len(tag) for tag in open_tags)

    for token in _tokenize(text):
        if _TAG.fullmatch(token):
            name = _tag_name(token)
            if token.startswith("</"):
                for index in range(len(open_tags) - 1, -1, -1):
                    if _tag_name(open_tags[index]) == name:
                        del open_tags[index]
                        break
            else:
                open_tags.append(token)
            current.append(token)
            length += len(token)
            continue

        rest = token
        while rest:
            available = limit - length - len(_closing(open_tags))
            if available <= 0:
                flush()
                continue
            if len(rest) <= available:
                current.append(rest)
                length += len(rest)
                rest = ""
            else:
                cut = rest.rfind("\n", 0, available + 1)
                if cut <= 0:
                    cut = available
                current.append(rest[:cut])
                flush()
                rest = rest[cut:].lstrip("\n")

    if "".join(current).strip():
        chunks.append("".join(current) + _closing(open_tags))
    return chunks


def _inline(text: str) -> str:
    codes: list[str] = []

    def stash(match: re.Match[str]) -> str:
        codes.append(escape_html(match.group(1)))
        return f"\x00{len(codes) - 1}\x00"

    text = _INLINE_CODE.sub(stash, text)
    text = escape_html(text)
    text = _LINK.sub(_render_link, text)
    text = _BOLD.sub(lambda match: f"<b>{match.group(1) or match.group(2)}</b>", text)
    text = _STRIKE.sub(lambda match: f"<s>{match.group(1)}</s>", text)
    text = _ITALIC.sub(lambda match: f"<i>{match.group(1) or match.group(2)}</i>", text)
    for index, code in enumerate(codes):
        text = text.replace(f"\x00{index}\x00", f"<code>{code}</code>", 1)
    return text


def _render_link(match: re.Match[str]) -> str:
    label = match.group(1)
    url = match.group(2).replace('"', "&quot;")
    return f'<a href="{url}">{label}</a>'


def _render_code_block(match: re.Match[str]) -> str:
    language = match.group(1).strip()
    code = escape_html(match.group(2).rstrip("\n"))
    if language:
        return f'<pre><code class="language-{escape_html(language)}">{code}</code></pre>'
    return f"<pre>{code}</pre>"


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    position = 0
    for match in _TAG.finditer(text):
        if match.start() > position:
            tokens.append(text[position : match.start()])
        tokens.append(match.group())
        position = match.end()
    if position < len(text):
        tokens.append(text[position:])
    return tokens


def _tag_name(tag: str) -> str:
    match = _TAG_NAME.match(tag)
    return match.group(1) if match else ""


def _closing(open_tags: list[str]) -> str:
    return "".join(f"</{_tag_name(tag)}>" for tag in reversed(open_tags))
