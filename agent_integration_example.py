"""
Integration example showing how agents would use the decision logging system.

This module demonstrates how to integrate the decision logging system with
trading agents in the multi-agent architecture.
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.logging.decision_logger import DecisionLogger, DecisionType, LogLevel


class BaseAgent:
    """
    Base class for trading agents with integrated decision logging.

    This shows how agents would integrate with the decision logging system.
    """

    def __init__(
        self,
        agent_name: str,
        agent_type: str,
        decision_logger: Optional[DecisionLogger] = None,
    ):
        """
        Initialize the agent with decision logging capability.

        Args:
            agent_name: Unique name for this agent instance
            agent_type: Type/category of the agent
            decision_logger: Optional decision logger instance
        """
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.decision_logger = decision_logger or DecisionLogger()

    def _log_decision(
        self,
        decision_type: DecisionType,
        action_taken: str,
        confidence_score: float,
        reasoning_summary: str,
        context_data: Optional[Dict[str, Any]] = None,
        full_reasoning: Optional[str] = None,
        llm_prompt: Optional[str] = None,
        llm_response: Optional[str] = None,
    ):
        """Log a decision made by this agent."""
        if self.decision_logger:
            return self.decision_logger.log_decision(
                agent_name=self.agent_name,
                agent_type=self.agent_type,
                decision_type=decision_type,
                action_taken=action_taken,
                confidence_score=confidence_score,
                reasoning_summary=reasoning_summary,
                context_data=context_data,
                full_reasoning=full_reasoning,
                llm_prompt=llm_prompt,
                llm_response=llm_response,
            )


class SignalGenerationAgent(BaseAgent):
    """
    Example signal generation agent with decision logging.
    """

    def __init__(self, symbol: str, decision_logger: Optional[DecisionLogger] = None):
        super().__init__(
            agent_name=f"SignalAgent_{symbol}",
            agent_type="TechnicalAnalysisAgent",
            decision_logger=decision_logger,
        )
        self.symbol = symbol

    def analyze_market(self, market_data: Dict[str, Any]) -> str:
        """
        Analyze market data and generate trading signals.

        Args:
            market_data: Current market data including OHLCV and indicators

        Returns:
            Signal: 'BUY', 'SELL', or 'HOLD'
        """
        # Simulate market analysis
        price = market_data.get("price", 1.0)
        ma_5 = market_data.get("ma_5", price * 0.999)
        ma_20 = market_data.get("ma_20", price * 0.995)
        rsi = market_data.get("rsi", 50)

        # Simple signal logic
        if ma_5 > ma_20 and rsi < 70:
            signal = "BUY"
            confidence = 0.8 if rsi < 30 else 0.6
            reasoning = f"Bullish: MA5({ma_5:.4f}) > MA20({ma_20:.4f}), RSI({rsi:.1f}) not overbought"
        elif ma_5 < ma_20 and rsi > 30:
            signal = "SELL"
            confidence = 0.8 if rsi > 70 else 0.6
            reasoning = f"Bearish: MA5({ma_5:.4f}) < MA20({ma_20:.4f}), RSI({rsi:.1f}) not oversold"
        else:
            signal = "HOLD"
            confidence = 0.5
            reasoning = f"Neutral: Mixed signals, MA5({ma_5:.4f}) vs MA20({ma_20:.4f}), RSI({rsi:.1f})"

        # Log the decision
        self._log_decision(
            decision_type=DecisionType.SIGNAL_GENERATION,
            action_taken=f"Generated {signal} signal for {self.symbol}",
            confidence_score=confidence,
            reasoning_summary=reasoning,
            context_data={
                "symbol": self.symbol,
                "signal": signal,
                "market_data": market_data,
                "analysis_timestamp": datetime.now().isoformat(),
            },
        )

        return signal


class RiskManagementAgent(BaseAgent):
    """
    Example risk management agent with decision logging.
    """

    def __init__(self, decision_logger: Optional[DecisionLogger] = None):
        super().__init__(
            agent_name="RiskManager_Main",
            agent_type="RiskManagementAgent",
            decision_logger=decision_logger,
        )
        self.max_position_size = 0.05  # 5% max position
        self.max_daily_risk = 0.02  # 2% max daily risk

    def assess_trade_risk(self, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk for a proposed trade.

        Args:
            trade_request: Trade details including symbol, direction, size, stops

        Returns:
            Risk assessment with approved position size and risk metrics
        """
        symbol = trade_request["symbol"]
        direction = trade_request["direction"]
        requested_size = trade_request["requested_size"]
        stop_loss = trade_request.get("stop_loss")
        take_profit = trade_request.get("take_profit")
        current_price = trade_request["current_price"]

        # Calculate risk
        if stop_loss:
            risk_per_unit = abs(current_price - stop_loss)
            risk_reward_ratio = (
                abs(take_profit - current_price) / risk_per_unit if take_profit else 1.0
            )
        else:
            risk_per_unit = current_price * 0.02  # 2% default stop
            risk_reward_ratio = 1.5  # Default R:R

        # Determine approved position size
        max_risk_size = self.max_daily_risk / (risk_per_unit / current_price)
        approved_size = min(requested_size, self.max_position_size, max_risk_size)

        # Risk assessment
        if risk_reward_ratio >= 2.0 and approved_size >= requested_size * 0.8:
            risk_level = "LOW"
            approval = "APPROVED"
            confidence = 0.9
        elif risk_reward_ratio >= 1.5 and approved_size >= requested_size * 0.5:
            risk_level = "MEDIUM"
            approval = "APPROVED_REDUCED"
            confidence = 0.7
        else:
            risk_level = "HIGH"
            approval = "REJECTED"
            confidence = 0.8
            approved_size = 0

        result = {
            "approval": approval,
            "approved_size": approved_size,
            "risk_level": risk_level,
            "risk_reward_ratio": risk_reward_ratio,
            "risk_per_unit": risk_per_unit,
        }

        # Log the decision
        self._log_decision(
            decision_type=DecisionType.RISK_ASSESSMENT,
            action_taken=f"{approval}: {symbol} {direction} position, size {approved_size:.3f}",
            confidence_score=confidence,
            reasoning_summary=f"Risk level: {risk_level}, R:R: {risk_reward_ratio:.1f}, "
            f"size {approved_size:.3f}/{requested_size:.3f}",
            context_data={
                "trade_request": trade_request,
                "risk_assessment": result,
                "risk_parameters": {
                    "max_position_size": self.max_position_size,
                    "max_daily_risk": self.max_daily_risk,
                },
            },
        )

        return result


def demo_agent_integration():
    """Demonstrate how agents integrate with the decision logging system."""
    print("=== Agent Integration Demo ===")

    # Create shared decision logger
    logger = DecisionLogger(
        log_file_path="logs/agent_integration.log", log_level=LogLevel.STANDARD
    )

    # Create agents
    signal_agent = SignalGenerationAgent("EURUSD", logger)
    risk_agent = RiskManagementAgent(logger)

    # Simulate market analysis
    market_data = {
        "price": 1.0895,
        "ma_5": 1.0890,
        "ma_20": 1.0875,
        "rsi": 45.2,
        "volume": 1500000,
    }

    print("Signal Agent analyzing market...")
    signal = signal_agent.analyze_market(market_data)
    print(f"Generated signal: {signal}")

    # Simulate trade request based on signal
    if signal in ["BUY", "SELL"]:
        trade_request = {
            "symbol": "EURUSD",
            "direction": signal,
            "requested_size": 0.02,
            "current_price": 1.0895,
            "stop_loss": 1.0850 if signal == "BUY" else 1.0940,
            "take_profit": 1.0950 if signal == "BUY" else 1.0840,
        }

        print("\nRisk Agent assessing trade...")
        risk_assessment = risk_agent.assess_trade_risk(trade_request)
        print(f"Risk assessment: {risk_assessment['approval']}")
        print(f"Approved size: {risk_assessment['approved_size']}")

    print("\nDecisions logged to:", logger.get_log_file_path())


if __name__ == "__main__":
    demo_agent_integration()
