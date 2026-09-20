from __future__ import annotations

import base64
from pathlib import Path

from graph_agent.media import image_content_part, is_image


def test_is_image_detects_image_extensions(tmp_path: Path) -> None:
    assert is_image(tmp_path / "photo.jpg") is True
    assert is_image(tmp_path / "scan.PNG") is True
    assert is_image(tmp_path / "notes.txt") is False
    assert is_image(tmp_path / "archive.zip") is False


def test_image_content_part_builds_base64_data_url(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0payload")

    part = image_content_part(path)

    assert part["type"] == "image_url"
    url = part["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    encoded = url.split(",", 1)[1]
    assert base64.b64decode(encoded) == b"\xff\xd8\xff\xe0payload"


def test_image_content_part_falls_back_to_jpeg_mime(tmp_path: Path) -> None:
    path = tmp_path / "drawing.unknown"
    path.write_bytes(b"data")

    part = image_content_part(path)

    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")
