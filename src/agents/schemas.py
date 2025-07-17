from typing import Any, Dict, Optional

from pydantic import BaseModel


class AgentMessage(BaseModel):
    """Message exchanged between agents."""

    sender: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class SystemMessage(BaseModel):
    """Message from the system to an agent."""

    content: str
    metadata: Optional[Dict[str, Any]] = None
