"""
Performance metrics and analysis for backtesting results.

This module provides comprehensive performance analysis capabilities including
returns calculation, risk metrics, trade statistics, and reporting functionality.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

logger = logging.getLogger("trading_bot.backtesting.metrics")


@dataclass 
class Trade:
    """Represents a completed trade."""
    
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    trade_type: str  # "BUY" or "SELL"
    transaction_cost: float = 0.0
    stop_loss_triggered: bool = False
    
    @property
    def duration(self) -> timedelta:
        """Trade duration."""
        return self.exit_time - self.entry_time
    
    @property
    def duration_hours(self) -> float:
        """Trade duration in hours."""
        return self.duration.total_seconds() / 3600
    
    @property
    def pnl(self) -> float:
        """Profit and loss for this trade."""
        if self.trade_type == "BUY":
            gross_pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # SELL
            gross_pnl = (self.entry_price - self.exit_price) * self.quantity
        
        return gross_pnl - self.transaction_cost
    
    @property
    def return_percentage(self) -> float:
        """Return percentage for this trade."""
        investment = self.entry_price * self.quantity
        return (self.pnl / investment) * 100 if investment > 0 else 0.0


@dataclass
class BacktestResults:
    """Container for all backtesting results."""
    
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    config: Optional[Dict[str, Any]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def total_trades(self) -> int:
        """Total number of trades."""
        return len(self.trades)
    
    @property
    def winning_trades(self) -> List[Trade]:
        """List of profitable trades."""
        return [trade for trade in self.trades if trade.pnl > 0]
    
    @property
    def losing_trades(self) -> List[Trade]:
        """List of losing trades."""
        return [trade for trade in self.trades if trade.pnl < 0]


class PerformanceAnalyzer:
    """Analyzes backtesting performance and calculates metrics."""
    
    def __init__(self, results: BacktestResults, initial_capital: float):
        """
        Initialize performance analyzer.
        
        Args:
            results: Backtesting results to analyze
            initial_capital: Starting capital amount
        """
        self.results = results
        self.initial_capital = initial_capital
        self._metrics_cache: Optional[Dict[str, Any]] = None
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if self._metrics_cache is not None:
            return self._metrics_cache
        
        if not self.results.trades:
            return self._empty_metrics()
        
        metrics = {}
        
        # Basic trade statistics
        metrics.update(self._calculate_trade_stats())
        
        # Return metrics
        metrics.update(self._calculate_return_metrics())
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics())
        
        # Time-based metrics
        metrics.update(self._calculate_time_metrics())
        
        # Additional statistics
        metrics.update(self._calculate_additional_stats())
        
        self._metrics_cache = metrics
        return metrics
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when no trades."""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "total_return_percentage": 0.0,
            "average_trade_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
        }
    
    def _calculate_trade_stats(self) -> Dict[str, Any]:
        """Calculate basic trade statistics."""
        winning_trades = self.results.winning_trades
        losing_trades = self.results.losing_trades
        
        return {
            "total_trades": self.results.total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / self.results.total_trades if self.results.total_trades > 0 else 0.0,
        }
    
    def _calculate_return_metrics(self) -> Dict[str, Any]:
        """Calculate return-based metrics."""
        total_pnl = sum(trade.pnl for trade in self.results.trades)
        final_capital = self.initial_capital + total_pnl
        
        trade_returns = [trade.return_percentage / 100 for trade in self.results.trades]
        
        return {
            "total_return": total_pnl,
            "total_return_percentage": (total_pnl / self.initial_capital) * 100,
            "final_capital": final_capital,
            "average_trade_return": np.mean(trade_returns) * 100 if trade_returns else 0.0,
            "median_trade_return": np.median(trade_returns) * 100 if trade_returns else 0.0,
            "best_trade": max(self.results.trades, key=lambda t: t.pnl).pnl if self.results.trades else 0.0,
            "worst_trade": min(self.results.trades, key=lambda t: t.pnl).pnl if self.results.trades else 0.0,
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate risk-based metrics."""
        if not self.results.trades:
            return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0}
        
        trade_returns = [trade.return_percentage / 100 for trade in self.results.trades]
        
        # Sharpe ratio (assuming risk-free rate of 0)
        if np.std(trade_returns) > 0:
            sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Profit factor
        gross_profit = sum(trade.pnl for trade in self.results.winning_trades)
        gross_loss = abs(sum(trade.pnl for trade in self.results.losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_percentage": (max_drawdown / self.initial_capital) * 100,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not self.results.equity_curve:
            return 0.0
        
        equity_values = [equity for _, equity in self.results.equity_curve]
        peak = equity_values[0]
        max_dd = 0.0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _calculate_time_metrics(self) -> Dict[str, Any]:
        """Calculate time-based metrics."""
        if not self.results.trades:
            return {"average_trade_duration_hours": 0.0}
        
        durations = [trade.duration_hours for trade in self.results.trades]
        
        return {
            "average_trade_duration_hours": np.mean(durations),
            "median_trade_duration_hours": np.median(durations),
            "min_trade_duration_hours": np.min(durations),
            "max_trade_duration_hours": np.max(durations),
        }
    
    def _calculate_additional_stats(self) -> Dict[str, Any]:
        """Calculate additional statistics."""
        if not self.results.trades:
            return {}
        
        # Stop loss statistics
        stop_loss_trades = [trade for trade in self.results.trades if trade.stop_loss_triggered]
        
        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades()
        
        return {
            "stop_loss_trades": len(stop_loss_trades),
            "stop_loss_rate": len(stop_loss_trades) / self.results.total_trades,
            "max_consecutive_wins": consecutive_wins,
            "max_consecutive_losses": consecutive_losses,
        }
    
    def _calculate_consecutive_trades(self) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        if not self.results.trades:
            return 0, 0
        
        max_wins = current_wins = 0
        max_losses = current_losses = 0
        
        for trade in self.results.trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def generate_report(self, detailed: bool = True) -> str:
        """Generate a human-readable performance report."""
        metrics = self.calculate_metrics()
        
        report = []
        report.append("="*60)
        report.append("BACKTESTING PERFORMANCE REPORT")
        report.append("="*60)
        
        # Summary section
        report.append("\nSUMMARY:")
        report.append(f"Total Trades: {metrics['total_trades']}")
        report.append(f"Win Rate: {metrics['win_rate']:.2%}")
        report.append(f"Total Return: ${metrics['total_return']:.2f} ({metrics['total_return_percentage']:.2f}%)")
        report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        report.append(f"Max Drawdown: ${metrics['max_drawdown']:.2f} ({metrics.get('max_drawdown_percentage', 0):.2f}%)")
        
        if detailed and metrics['total_trades'] > 0:
            report.append("\nDETAILED METRICS:")
            report.append(f"Winning Trades: {metrics['winning_trades']}")
            report.append(f"Losing Trades: {metrics['losing_trades']}")
            report.append(f"Average Trade Return: {metrics['average_trade_return']:.2f}%")
            report.append(f"Best Trade: ${metrics['best_trade']:.2f}")
            report.append(f"Worst Trade: ${metrics['worst_trade']:.2f}")
            report.append(f"Profit Factor: {metrics['profit_factor']:.2f}")
            report.append(f"Average Trade Duration: {metrics['average_trade_duration_hours']:.1f} hours")
            
            if 'stop_loss_trades' in metrics:
                report.append(f"Stop Loss Triggered: {metrics['stop_loss_trades']} ({metrics['stop_loss_rate']:.2%})")
            
            report.append(f"Max Consecutive Wins: {metrics['max_consecutive_wins']}")
            report.append(f"Max Consecutive Losses: {metrics['max_consecutive_losses']}")
        
        report.append("="*60)
        
        return "\n".join(report)
    
    def get_trade_summary(self) -> pd.DataFrame:
        """Get a DataFrame summary of all trades."""
        if not self.results.trades:
            return pd.DataFrame()
        
        trade_data = []
        for i, trade in enumerate(self.results.trades):
            trade_data.append({
                "trade_id": i + 1,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "duration_hours": trade.duration_hours,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "trade_type": trade.trade_type,
                "pnl": trade.pnl,
                "return_pct": trade.return_percentage,
                "stop_loss_triggered": trade.stop_loss_triggered,
            })
        
        return pd.DataFrame(trade_data)
