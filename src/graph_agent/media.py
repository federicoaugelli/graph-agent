from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

FALLBACK_IMAGE_MIME = "image/jpeg"


def is_image(path: Path) -> bool:
    """Whether a path looks like an image, by mime type (extension based)."""
    mime, _ = mimetypes.guess_type(path.name)
    return mime is not None and mime.startswith("image/")


def image_content_part(path: Path) -> dict[str, Any]:
    """Build an OpenAI-style ``image_url`` content part with a base64 data URL."""
    mime = mimetypes.guess_type(path.name)[0] or FALLBACK_IMAGE_MIME
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
