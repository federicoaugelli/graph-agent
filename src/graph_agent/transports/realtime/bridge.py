from __future__ import annotations

from graph_agent.config import RealtimeChannelConfig
from graph_agent.core.service import AgentService


class RealtimeBridge:
    """User WS <-> realtime session bridge (OpenAI Realtime protocol via LiteLLM).

    To implement (Phase 6):
    - connect to {litellm_base}/v1/realtime?model=... with websockets
    - session.update declaring the custom delegate_to_brain tool
    - forward audio/text events between client and realtime session
    - on function_call(delegate_to_brain): run service.run() async,
      then send response.function_call_output with the result
    """

    def __init__(self, service: AgentService, config: RealtimeChannelConfig) -> None:
        self.service = service
        self.config = config

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError
