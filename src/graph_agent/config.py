from __future__ import annotations

from pathlib import Path
from typing import Literal

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
    context_token_limit: int = 200_000
    compaction_keep_messages: int = 20
    latency_metrics: bool = False
    max_tool_output_chars: int = 20_000
    repeat_tool_call_limit: int = 3


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
    network: bool = False
    writable_binds: dict[str, str] = Field(default_factory=dict)
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
    approval_mode: Literal["manual", "auto"] = "manual"


class RealtimeChannelConfig(BaseModel):
    enabled: bool = False
    backend: Literal["openai", "qwen"] = "openai"
    model: str = ""
    api_base: str = "http://localhost:4000"
    api_key_env: str | None = None
    instructions: str | None = None
    voice: str | None = None


class VoiceChannelConfig(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8101
    sample_rate: int = 16000
    output_sample_rate: int | None = None
    greeting: str | None = None
    advertise_host: str | None = None
    trunk: str = "trunk"
    caller_id: str | None = None
    default_target: str | None = None
    ami_host: str = "127.0.0.1"
    ami_port: int = 5038
    ami_username: str = "graphagent"
    ami_secret_env: str = "ASTERISK_AMI_SECRET"


class ChannelsConfig(BaseModel):
    http: HttpChannelConfig = Field(default_factory=HttpChannelConfig)
    telegram: TelegramChannelConfig = Field(default_factory=TelegramChannelConfig)
    realtime: RealtimeChannelConfig = Field(default_factory=RealtimeChannelConfig)
    voice: VoiceChannelConfig = Field(default_factory=VoiceChannelConfig)


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
    timeout_seconds: float = 30.0
    max_snippet_chars: int = 1000


class WebFetchToolConfig(BaseModel):
    enabled: bool = False
    timeout_seconds: float = 15.0
    max_bytes: int = 2_000_000
    max_chars: int = 8000
    allow_private_hosts: bool = False


class ToolsConfig(BaseModel):
    filesystem: FilesystemToolConfig = Field(default_factory=FilesystemToolConfig)
    shell: ShellToolConfig = Field(default_factory=ShellToolConfig)
    web_search: WebSearchToolConfig = Field(default_factory=WebSearchToolConfig)
    web_fetch: WebFetchToolConfig = Field(default_factory=WebFetchToolConfig)


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
