from pydantic import BaseModel
from typing import Optional, Dict, Any


class AgentMessage(BaseModel):
    """Message exchanged between agents."""

    sender: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class SystemMessage(BaseModel):
    """Message from the system to an agent."""

    content: str
    metadata: Optional[Dict[str, Any]] = None
