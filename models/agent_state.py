
from typing import Any

from openai import BaseModel


class AgentState(BaseModel):
    messages: list[dict[str,Any]]
    iteration_count: int = 0
    status: str = "RUNNING"