"""
RiskAssessmentAgent: Evaluates position sizing, risk metrics, and stop-loss levels for trading decisions.
Phase 4, Task 4.1 (APM Implementation Plan)
"""
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class RiskAssessmentResult:
    position_size: float
    risk_score: float
    stop_loss: Optional[float]
    notes: Optional[str] = None

class RiskAssessmentAgent:
    def __init__(self, max_risk_per_trade: float = 0.02, account_balance: float = 10000.0):
        self.max_risk_per_trade = max_risk_per_trade  # e.g., 2% of account
        self.account_balance = account_balance

    def assess(self, decision: Dict, backtest_metrics: Optional[Dict] = None) -> RiskAssessmentResult:
        """
        Assess risk for a given TradingDecision and backtest metrics.
        Args:
            decision: TradingDecision dict with 'action', 'entry_price', 'stop_loss', 'position_size', etc.
            backtest_metrics: Optional dict with performance metrics.
        Returns:
            RiskAssessmentResult
        """
        action = decision.get('action')
        entry_price = decision.get('entry_price')
        stop_loss = decision.get('stop_loss')
        position_size = decision.get('position_size', 1.0)

        # Basic position sizing: risk per trade
        if stop_loss and entry_price and action in ('BUY', 'SELL'):
            risk_per_unit = abs(entry_price - stop_loss)
            max_loss = self.account_balance * self.max_risk_per_trade
            size = max_loss / risk_per_unit if risk_per_unit > 0 else 0.0
            size = min(size, position_size)
        else:
            size = 0.0

        # Simple risk score: 1.0 = low risk, 0.0 = high risk
        risk_score = 1.0 if size > 0.0 else 0.0
        notes = None
        if backtest_metrics:
            win_rate = backtest_metrics.get('win_rate', 0.5)
            if win_rate < 0.4:
                risk_score *= 0.5
                notes = 'Low historical win rate; risk reduced.'

        return RiskAssessmentResult(
            position_size=size,
            risk_score=risk_score,
            stop_loss=stop_loss,
            notes=notes
        )
