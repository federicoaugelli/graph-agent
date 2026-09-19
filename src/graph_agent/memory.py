from __future__ import annotations

from pathlib import Path

MEMORY_FILENAME = "MEMORY.md"

_TEMPLATE = "# Memory\n\nLong-term notes carried across sessions.\n"

_INSTRUCTIONS = (
    "## Persistent memory\n\n"
    "You have a persistent memory file at `{path}`. Read it at the start of every "
    "session to recover prior context, and update it whenever you learn durable facts "
    "about the user or the project (preferences, decisions, ongoing work). Keep entries "
    "concise. Use the filesystem tool to read and write it."
)


class MemoryStore:
    """File-based cross-session memory stored as markdown in the workspace."""

    def __init__(self, root: Path, directory: Path = Path("memory")) -> None:
        self.root = root
        self.directory = directory

    @property
    def path(self) -> Path:
        return self.root / self.directory / MEMORY_FILENAME

    @property
    def relative_path(self) -> str:
        return (self.directory / MEMORY_FILENAME).as_posix()

    def ensure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(_TEMPLATE)

    def instructions(self) -> str:
        return _INSTRUCTIONS.format(path=self.relative_path)

    def compose(self, base: str | None) -> str:
        if not base:
            return self.instructions()
        return f"{base}\n\n{self.instructions()}"