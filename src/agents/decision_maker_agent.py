"""
Decision Maker Agent for the trading bot.

This agent processes signals from the Signal Detection Agent and makes final
trading decisions based on signal confluence and Bill Williams methodology.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from ..data.models import OHLCV
from ..logging.decision_logger import DecisionLogger, DecisionType
from .base_agent import BaseAgent
from .schemas import Signal, TradingDecision

logger = logging.getLogger("trading_bot.decision_maker")


class DecisionMakerAgent(BaseAgent):
    """
    Decision Maker Agent that processes signals and makes trading decisions.

    This agent evaluates signal confluence, applies Bill Williams methodology,
    and makes final BUY/SELL/HOLD decisions with confidence scoring.
    """

    def __init__(
        self,
        client=None,
        decision_logger: Optional[DecisionLogger] = None,
        confluence_threshold: float = 0.7,
    ):
        """
        Initialize the Decision Maker Agent.

        Args:
            client: Optional OpenAI client for complex reasoning
            decision_logger: Logger for decision tracking
            confluence_threshold: Minimum confluence score for trading decisions
        """
        super().__init__(
            name="DecisionMaker",
            system_prompt=(
                "You are a trading decision maker using Bill Williams methodology. "
                "Analyze signal confluence and market context to make informed trading decisions. "
                "Always provide clear reasoning for your decisions."
            ),
            client=client,
        )
        self.decision_logger = decision_logger
        self.confluence_threshold = confluence_threshold

    def make_decision(
        self, signals: List[Signal], market_data: OHLCV
    ) -> TradingDecision:
        """
        Make a trading decision based on signals and market data.

        Args:
            signals: List of signals from Signal Detection Agent
            market_data: Current market data (OHLCV)

        Returns:
            TradingDecision with action, confidence, and reasoning
        """
        if not signals:
            decision = TradingDecision(
                action="HOLD",
                confidence=0.0,
                reasoning="No signals available for decision making",
                signals_used=[],
                timestamp=datetime.now(),
                entry_price=market_data.close,
            )
            self._log_decision(decision, market_data)
            return decision

        # Evaluate signal confluence
        confluence_score = self.evaluate_confluence(signals)

        # Determine trading action
        action = self.determine_action(confluence_score, signals)

        # Calculate confidence score
        confidence = self.calculate_confidence(confluence_score, signals)

        # Generate reasoning
        reasoning = self.generate_reasoning(signals, confluence_score, action)

        # Create decision
        decision = TradingDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            signals_used=signals,
            timestamp=datetime.now(),
            entry_price=market_data.close if action != "HOLD" else None,
            stop_loss=(
                self.calculate_stop_loss(market_data, action)
                if action != "HOLD"
                else None
            ),
        )

        # Log the decision
        self._log_decision(decision, market_data)

        return decision

    def evaluate_confluence(self, signals: List[Signal]) -> float:
        """
        Calculate signal confluence score based on signal agreement.

        Args:
            signals: List of signals to evaluate

        Returns:
            Confluence score between 0.0 and 1.0
        """
        if not signals:
            return 0.0

        # Group signals by type
        buy_signals = [s for s in signals if s.type == "BUY"]
        sell_signals = [s for s in signals if s.type == "SELL"]
        hold_signals = [s for s in signals if s.type == "HOLD"]

        total_signals = len(signals)

        # Calculate weighted confluence based on signal confidence
        if buy_signals:
            buy_weight = sum(s.confidence for s in buy_signals)
            buy_count = len(buy_signals)
            buy_score = buy_weight * buy_count / total_signals
        else:
            buy_score = 0.0

        if sell_signals:
            sell_weight = sum(s.confidence for s in sell_signals)
            sell_count = len(sell_signals)
            sell_score = sell_weight * sell_count / total_signals
        else:
            sell_score = 0.0

        # Confluence is the maximum agreement in one direction
        confluence = max(buy_score, sell_score)

        # Penalty for conflicting signals
        if buy_signals and sell_signals:
            conflict_penalty = (
                min(len(buy_signals), len(sell_signals)) / total_signals * 0.2
            )
            confluence = max(0.0, confluence - conflict_penalty)

        return min(1.0, confluence)

    def determine_action(self, confluence_score: float, signals: List[Signal]) -> str:
        """
        Determine trading action based on confluence and signals.

        Args:
            confluence_score: Signal confluence score
            signals: List of signals

        Returns:
            Trading action: "BUY", "SELL", or "HOLD"
        """
        if confluence_score < self.confluence_threshold:
            return "HOLD"

        # Count signal types
        buy_count = sum(1 for s in signals if s.type == "BUY")
        sell_count = sum(1 for s in signals if s.type == "SELL")

        # Weighted decision based on signal confidence
        buy_weight = sum(s.confidence for s in signals if s.type == "BUY")
        sell_weight = sum(s.confidence for s in signals if s.type == "SELL")

        if buy_weight > sell_weight and buy_count >= 2:
            return "BUY"
        elif sell_weight > buy_weight and sell_count >= 2:
            return "SELL"
        else:
            return "HOLD"

    def calculate_confidence(
        self, confluence_score: float, signals: List[Signal]
    ) -> float:
        """
        Calculate confidence score for the decision.

        Args:
            confluence_score: Signal confluence score
            signals: List of signals

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not signals:
            return 0.0

        # Base confidence from confluence
        base_confidence = confluence_score

        # Boost confidence based on number of signals
        signal_count = len(signals)
        if signal_count >= 3:
            signal_boost = 0.2
        elif signal_count >= 2:
            signal_boost = 0.1
        else:
            signal_boost = 0.0

        # Average signal confidence
        avg_signal_confidence = sum(s.confidence for s in signals) / len(signals)

        # Combined confidence
        confidence = (base_confidence + signal_boost + avg_signal_confidence) / 3

        return min(1.0, confidence)

    def generate_reasoning(
        self, signals: List[Signal], confluence_score: float, action: str
    ) -> str:
        """
        Generate human-readable reasoning for the decision.

        Args:
            signals: List of signals used
            confluence_score: Calculated confluence score
            action: Chosen action

        Returns:
            Reasoning string
        """
        signal_summary: Dict[str, List[str]] = {}
        for signal in signals:
            if signal.type not in signal_summary:
                signal_summary[signal.type] = []
            signal_summary[signal.type].append(f"{signal.confidence:.2f}")

        reasoning_parts = [
            f"Decision: {action}",
            f"Signal confluence: {confluence_score:.2f}",
            f"Total signals analyzed: {len(signals)}",
        ]

        for signal_type, confidences in signal_summary.items():
            reasoning_parts.append(
                f"{signal_type} signals: {len(confidences)} (confidences: {', '.join(confidences)})"
            )

        if action == "HOLD":
            if confluence_score < self.confluence_threshold:
                reasoning_parts.append(
                    f"Action: HOLD - Confluence below threshold ({self.confluence_threshold:.2f})"
                )
            else:
                reasoning_parts.append("Action: HOLD - Insufficient signal agreement")
        else:
            reasoning_parts.append(
                f"Action: {action} - Strong signal confluence detected"
            )

        return " | ".join(reasoning_parts)

    def calculate_stop_loss(self, market_data: OHLCV, action: str) -> Optional[float]:
        """
        Calculate stop loss level based on Bill Williams methodology.

        Args:
            market_data: Current market data
            action: Trading action

        Returns:
            Stop loss price or None
        """
        if action == "HOLD":
            return None

        # Simple stop loss: 2% from entry price
        stop_loss_pct = 0.02

        if action == "BUY":
            return market_data.close * (1 - stop_loss_pct)
        elif action == "SELL":
            return market_data.close * (1 + stop_loss_pct)

        return None

    def _log_decision(self, decision: TradingDecision, market_data: OHLCV) -> None:
        """
        Log the trading decision if logger is available.

        Args:
            decision: Trading decision to log
            market_data: Market data context
        """
        if not self.decision_logger:
            return

        try:
            self.decision_logger.log_decision(
                agent_name=self.name,
                agent_type="decision_maker",
                decision_type=DecisionType.TRADE_EXECUTION,
                action_taken=decision.action,
                confidence_score=decision.confidence,
                reasoning_summary=f"{decision.action} decision with {len(decision.signals_used)} signals",
                full_reasoning=decision.reasoning,
                context_data={
                    "action": decision.action,
                    "entry_price": decision.entry_price,
                    "stop_loss": decision.stop_loss,
                    "signals_count": len(decision.signals_used),
                    "market_price": market_data.close,
                    "market_volume": market_data.volume,
                },
            )
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
