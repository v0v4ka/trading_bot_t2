from datetime import datetime
from typing import Any, Dict, List, Optional

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


class Signal(BaseModel):
    """Trading signal from Signal Detection Agent."""

    timestamp: datetime
    type: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0-1.0
    details: Dict[str, Any]


class TradingDecision(BaseModel):
    """Final trading decision from Decision Maker Agent."""

    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0-1.0
    reasoning: str
    signals_used: List[Signal]
    timestamp: datetime
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None  # Reverse pyramiding position size
