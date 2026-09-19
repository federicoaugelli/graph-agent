from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    extra: dict[str, object] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    brain: ModelConfig
    router: ModelConfig | None = None


class AgentConfig(BaseModel):
    max_iterations: int = 25
    workspace: Path = Path("data/workspace")
    checkpointer_db: Path = Path("data/checkpoints.sqlite")
    system_prompt: str | None = None
    persona_file: Path | None = None
    session_ttl_minutes: int | None = None


class MemoryConfig(BaseModel):
    enabled: bool = False
    dir: Path = Path("memory")


class SkillsConfig(BaseModel):
    enabled: bool = False
    dir: Path = Path("skills")


class SandboxConfig(BaseModel):
    timeout_seconds: int = 120
    max_output_bytes: int = 20_000
    approval_required: bool = True
    mask_paths: list[Path] = Field(default_factory=list)
    deny_patterns: list[str] = Field(default_factory=list)


class HttpChannelConfig(BaseModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8080
    api_key_env: str | None = None


class TelegramChannelConfig(BaseModel):
    enabled: bool = False
    token_env: str = "TELEGRAM_BOT_TOKEN"
    allowed_user_ids: list[int] = Field(default_factory=list)


class RealtimeChannelConfig(BaseModel):
    enabled: bool = False
    model: str = ""
    litellm_base: str = "http://localhost:4000"


class ChannelsConfig(BaseModel):
    http: HttpChannelConfig = Field(default_factory=HttpChannelConfig)
    telegram: TelegramChannelConfig = Field(default_factory=TelegramChannelConfig)
    realtime: RealtimeChannelConfig = Field(default_factory=RealtimeChannelConfig)


class SchedulerConfig(BaseModel):
    enabled: bool = False
    jobstore_db: Path = Path("data/scheduler.sqlite")


class MCPServerConfig(BaseModel):
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class MCPConfig(BaseModel):
    servers: list[MCPServerConfig] = Field(default_factory=list)


class FilesystemToolConfig(BaseModel):
    enabled: bool = True
    root: Path = Path("data/workspace")


class ShellToolConfig(BaseModel):
    enabled: bool = True


class WebSearchToolConfig(BaseModel):
    enabled: bool = False
    provider: str = "searxng"
    api_base: str | None = None
    api_key_env: str | None = None


class ToolsConfig(BaseModel):
    filesystem: FilesystemToolConfig = Field(default_factory=FilesystemToolConfig)
    shell: ShellToolConfig = Field(default_factory=ShellToolConfig)
    web_search: WebSearchToolConfig = Field(default_factory=WebSearchToolConfig)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    level: str = "INFO"
    json_output: bool = Field(default=False, alias="json")


class AppConfig(BaseModel):
    models: ModelsConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    load_dotenv()
    config_path = Path(path)
    raw: dict[str, object] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text())
        if loaded:
            raw = loaded
    local_path = config_path.with_name("config.local.yaml")
    if local_path.exists():
        local = yaml.safe_load(local_path.read_text())
        if local:
            raw = _deep_merge(raw, local)
    return AppConfig.model_validate(raw)


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged
