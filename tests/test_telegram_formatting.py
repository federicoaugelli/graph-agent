from __future__ import annotations

from graph_agent.transports.telegram.formatting import (
    escape_html,
    markdown_to_telegram_html,
    split_telegram_html,
)


def test_plain_text_is_left_alone() -> None:
    assert markdown_to_telegram_html("hello world") == "hello world"


def test_escape_html_covers_specials() -> None:
    assert escape_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_inline_styles() -> None:
    rendered = markdown_to_telegram_html("**bold** _italic_ ~~gone~~")
    assert rendered == "<b>bold</b> <i>italic</i> <s>gone</s>"


def test_inline_code_and_escaping() -> None:
    assert markdown_to_telegram_html("use `a<b>`") == "use <code>a&lt;b&gt;</code>"


def test_fenced_code_block_with_language() -> None:
    rendered = markdown_to_telegram_html("```python\nprint(1)\n```")
    assert rendered == '<pre><code class="language-python">print(1)</code></pre>'


def test_fenced_code_block_without_language_escapes() -> None:
    assert markdown_to_telegram_html("```\nx < y\n```") == "<pre>x &lt; y</pre>"


def test_heading_becomes_bold() -> None:
    assert markdown_to_telegram_html("## Title") == "<b>Title</b>"


def test_link_is_rendered_with_escaped_url() -> None:
    rendered = markdown_to_telegram_html("[docs](https://x.com/a?b=1&c=2)")
    assert rendered == '<a href="https://x.com/a?b=1&amp;c=2">docs</a>'


def test_unordered_and_ordered_lists() -> None:
    assert markdown_to_telegram_html("- one\n- two") == "• one\n• two"
    assert markdown_to_telegram_html("1. one") == "1. one"


def test_blockquote_groups_consecutive_lines() -> None:
    assert markdown_to_telegram_html("> hi\n> there") == "<blockquote>hi\nthere</blockquote>"


def test_plain_html_is_escaped() -> None:
    assert markdown_to_telegram_html("5 < 6 & 7 > 3") == "5 &lt; 6 &amp; 7 &gt; 3"


def test_intraword_underscore_is_not_italic() -> None:
    assert markdown_to_telegram_html("snake_case_name") == "snake_case_name"


def test_split_returns_short_text_unchanged() -> None:
    assert split_telegram_html("short", limit=100) == ["short"]


def test_split_respects_the_limit() -> None:
    text = "a" * 5000
    chunks = split_telegram_html(text, limit=1000)
    assert "".join(chunks) == text
    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_split_balances_tags_across_chunks() -> None:
    text = "<b>" + "x" * 100 + "</b>"
    chunks = split_telegram_html(text, limit=40)
    assert len(chunks) > 1
    assert all(chunk.startswith("<b>") and chunk.endswith("</b>") for chunk in chunks)
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_convert_then_split_keeps_code_parseable() -> None:
    rendered = markdown_to_telegram_html("```\n" + "<x>\n" * 50 + "```")
    chunks = split_telegram_html(rendered, limit=120)
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert all("<pre>" in chunk and "</pre>" in chunk for chunk in chunks)
