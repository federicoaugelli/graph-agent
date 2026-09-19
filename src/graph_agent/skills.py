from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

SKILL_FILENAME = "SKILL.md"

_INDEX_HEADER = (
    "## Skills\n\n"
    "You have the following skills available. When a task matches a skill's "
    "description, read its `SKILL.md` with the filesystem tool and follow its "
    "instructions."
)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    relative_path: str


class SkillStore:
    """Progressive-disclosure skills stored as markdown in the workspace.

    Only the index (name, description, path) is injected into the system prompt;
    the SKILL.md body is read on demand via the filesystem tool.
    """

    def __init__(self, root: Path, directory: Path = Path("skills")) -> None:
        self.root = root
        self.directory = directory

    @property
    def directory_path(self) -> Path:
        return self.root / self.directory

    def ensure(self) -> None:
        self.directory_path.mkdir(parents=True, exist_ok=True)

    def discover(self) -> list[Skill]:
        base = self.directory_path
        if not base.is_dir():
            return []

        skills: list[Skill] = []
        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            skill_file = entry / SKILL_FILENAME
            if not skill_file.is_file():
                continue
            name, description = _parse_frontmatter(skill_file.read_text(), skill_file)
            skills.append(
                Skill(
                    name=name,
                    description=description,
                    relative_path=skill_file.relative_to(self.root).as_posix(),
                )
            )

        skills.sort(key=lambda skill: skill.name)
        return skills

    def instructions(self) -> str:
        skills = self.discover()
        if not skills:
            return ""

        lines = [_INDEX_HEADER, ""]
        lines.extend(
            f"- {skill.name} ({skill.relative_path}): {skill.description}" for skill in skills
        )
        return "\n".join(lines)

    def compose(self, base: str | None) -> str:
        parts = [part for part in (base, self.instructions()) if part]
        return "\n\n".join(parts)


def _parse_frontmatter(text: str, path: Path) -> tuple[str, str]:
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing YAML frontmatter")

    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{path}: unterminated YAML frontmatter")

    meta = yaml.safe_load(parts[1])
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: frontmatter is not a mapping")

    name = meta.get("name")
    description = meta.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{path}: frontmatter needs a non-empty 'name'")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{path}: frontmatter needs a non-empty 'description'")

    return name.strip(), description.strip()