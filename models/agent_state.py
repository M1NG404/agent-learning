
from enum import Enum
from typing import Any, Optional

from openai import BaseModel


class AgentStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"
    

class AgentState(BaseModel):
    messages: list[dict[str,Any]]
    iteration_count: int = 0
    status: AgentStatus = AgentStatus.RUNNING
    final_answer: Optional[str] = None