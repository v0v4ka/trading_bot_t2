"""
Decision Maker Agent for the trading bot.

This agent processes signals from the Signal Detection Agent and makes final
trading decisions based on signal confluence and Bill Williams methodology.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from ..data.models import OHLCV
from ..decision_logging.decision_logger import DecisionLogger, DecisionType
from .base_agent import BaseAgent
from .schemas import Signal, TradingDecision

logger = logging.getLogger("trading_bot.decision_maker")


class DecisionMakerAgent(BaseAgent):
    def calculate_position_size(self, staged_action: str) -> float:
        """
        Calculate position size based on staged entry (reverse pyramiding).
        First entry: 1.0 (full size), Second: 0.5, Third: 0.25, else: 1.0
        """
        if staged_action == 'BUY':
            return 1.0
        elif staged_action == 'BUY_ADDON':
            return 0.5
        elif staged_action == 'BUY_ADDON2':
            return 0.25
        else:
            return 1.0
    def detect_alligator_state(self, signals: List[Signal], market_data: OHLCV) -> str:
        """
        Detect Alligator state: 'awake' or 'sleeping'.
        Returns 'awake' if Alligator lines are sufficiently separated, else 'sleeping'.
        Assumes signal.details['indicator'] == 'alligator' contains line values.
        """
        for s in signals:
            if s.details.get('indicator') == 'alligator':
                jaw = s.details.get('jaw')
                teeth = s.details.get('teeth')
                lips = s.details.get('lips')
                # Example logic: Alligator is awake if lines are separated by a threshold
                if jaw is not None and teeth is not None and lips is not None:
                    sep1 = abs(jaw - teeth)
                    sep2 = abs(teeth - lips)
                    threshold = 0.5  # This can be tuned
                    if sep1 > threshold and sep2 > threshold:
                        return 'awake'
                    else:
                        return 'sleeping'
        return 'unknown'
    def detect_three_wise_men(self, signals: List[Signal], market_data: OHLCV) -> Dict[str, bool]:
        """
        Detect Three Wise Men staged entry signals.
        Returns a dict: {'first': bool, 'second': bool, 'third': bool}
        """
        result = {'first': False, 'second': False, 'third': False}
        # First Wise Man: Reversal bar outside Alligator’s mouth, confirmed by AO color
        for s in signals:
            if (
                s.details.get('indicator') == 'alligator'
                and s.details.get('outside_mouth', False)
                and s.details.get('ao_color') in ['green', 'red']
            ):
                result['first'] = True
                break
        # Second Wise Man: AO saucer pattern for add-on entry
        for s in signals:
            if s.details.get('indicator') == 'ao' and s.details.get('saucer', False):
                result['second'] = True
                break
        # Third Wise Man: Fractal breakout for further add-on
        for s in signals:
            if s.details.get('indicator') == 'fractal' and s.details.get('breakout', False):
                result['third'] = True
                break
        return result
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

        # Detect Alligator state
        alligator_state = self.detect_alligator_state(signals, market_data)

        # Detect Three Wise Men staged entry signals
        wise_men = self.detect_three_wise_men(signals, market_data)

        # Determine staged action
        staged_action = None
        staged_reason = []
        if wise_men['first']:
            staged_action = 'BUY'
            staged_reason.append('First Wise Man: reversal bar outside Alligator’s mouth, AO color confirmed')
        if wise_men['second']:
            staged_action = 'BUY_ADDON'
            staged_reason.append('Second Wise Man: AO saucer pattern detected')
        if wise_men['third']:
            staged_action = 'BUY_ADDON2'
            staged_reason.append('Third Wise Man: Fractal breakout detected')

        # Calculate position size for reverse pyramiding
        position_size = self.calculate_position_size(staged_action) if staged_action else 1.0

        # Filter trades based on Alligator state
        if alligator_state == 'sleeping':
            action = 'HOLD'
            staged_reason.append('Alligator sleeping: trade filtered')
        elif alligator_state == 'awake':
            # If Alligator is awake, allow trade if:
            # - staged_action is present
            # - OR at least one strong BUY/SELL signal (confidence >= 0.8)
            if staged_action:
                action = staged_action
            else:
                strong_buy = any(s.type == 'BUY' and s.confidence >= 0.8 for s in signals)
                strong_sell = any(s.type == 'SELL' and s.confidence >= 0.8 for s in signals)
                if strong_buy:
                    action = 'BUY'
                elif strong_sell:
                    action = 'SELL'
                else:
                    action = self.determine_action(confluence_score, signals)
        else:
            # If state unknown, allow staged entry or confluence logic
            action = staged_action if staged_action else self.determine_action(confluence_score, signals)

        # Calculate confidence score
        confidence = self.calculate_confidence(confluence_score, signals)

        # Generate reasoning
        reasoning = self.generate_reasoning(signals, confluence_score, action)
        if staged_reason:
            reasoning += ' | Staged Entry: ' + ', '.join(staged_reason)
        reasoning += f' | Alligator state: {alligator_state}'

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
            position_size=position_size if action != "HOLD" else None,
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

    def calculate_stop_loss(self, market_data: OHLCV, action: str, signals: Optional[List[Signal]] = None, staged_action: Optional[str] = None) -> Optional[float]:
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

        reversal_bar = None
        trailing_fractal = None
        if signals:
            for s in signals:
                if s.details.get('indicator') == 'alligator' and s.details.get('outside_mouth', False):
                    reversal_bar = s.details.get('bar_low') if action.startswith('BUY') else s.details.get('bar_high')
                    break
            for s in reversed(signals):
                if s.details.get('indicator') == 'fractal' and s.details.get('breakout', False):
                    trailing_fractal = s.details.get('fractal_low') if action.startswith('BUY') else s.details.get('fractal_high')
                    break

        # Initial entry: stop below/above reversal bar
        if staged_action == 'BUY' or staged_action == 'SELL':
            if reversal_bar is not None:
                if action.startswith('BUY'):
                    return reversal_bar * 0.995  # 0.5% below
                else:
                    return reversal_bar * 1.005  # 0.5% above
            else:
                # fallback to 2% rule
                stop_loss_pct = 0.02
                if action == "BUY":
                    return market_data.close * (1 - stop_loss_pct)
                elif action == "SELL":
                    return market_data.close * (1 + stop_loss_pct)
        # Add-ons: trailing stop to most recent fractal
        elif staged_action in ['BUY_ADDON', 'BUY_ADDON2', 'SELL_ADDON', 'SELL_ADDON2']:
            if trailing_fractal is not None:
                if action.startswith('BUY'):
                    return trailing_fractal * 0.995
                else:
                    return trailing_fractal * 1.005
            else:
                # fallback to 2% rule
                stop_loss_pct = 0.02
                if action == "BUY":
                    return market_data.close * (1 - stop_loss_pct)
                elif action == "SELL":
                    return market_data.close * (1 + stop_loss_pct)
        # fallback for unknown
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
