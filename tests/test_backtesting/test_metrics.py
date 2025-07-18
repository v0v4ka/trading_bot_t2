"""Test suite for backtesting metrics and performance analysis."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.backtesting.metrics import BacktestResults, PerformanceAnalyzer, Trade


class TestTrade:
    """Test cases for Trade class."""

    def test_trade_creation(self):
        """Test creating a trade instance."""
        entry_time = datetime(2023, 1, 1, 10, 0)
        exit_time = datetime(2023, 1, 1, 15, 30)

        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=100.0,
            exit_price=105.0,
            quantity=10.0,
            trade_type="BUY",
            transaction_cost=2.0,
        )

        assert trade.entry_time == entry_time
        assert trade.exit_time == exit_time
        assert trade.entry_price == 100.0
        assert trade.exit_price == 105.0
        assert trade.quantity == 10.0
        assert trade.trade_type == "BUY"
        assert trade.transaction_cost == 2.0
        assert trade.stop_loss_triggered is False

    def test_trade_duration(self):
        """Test trade duration calculation."""
        entry_time = datetime(2023, 1, 1, 10, 0)
        exit_time = datetime(2023, 1, 1, 15, 30)

        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=100.0,
            exit_price=105.0,
            quantity=10.0,
            trade_type="BUY",
        )

        expected_duration = timedelta(hours=5, minutes=30)
        assert trade.duration == expected_duration
        assert trade.duration_hours == 5.5

    def test_buy_trade_pnl(self):
        """Test P&L calculation for BUY trade."""
        trade = Trade(
            entry_time=datetime(2023, 1, 1, 10, 0),
            exit_time=datetime(2023, 1, 1, 15, 0),
            entry_price=100.0,
            exit_price=105.0,
            quantity=10.0,
            trade_type="BUY",
            transaction_cost=2.0,
        )

        # P&L = (exit_price - entry_price) * quantity - transaction_cost
        # P&L = (105 - 100) * 10 - 2 = 50 - 2 = 48
        assert trade.pnl == 48.0

    def test_sell_trade_pnl(self):
        """Test P&L calculation for SELL trade."""
        trade = Trade(
            entry_time=datetime(2023, 1, 1, 10, 0),
            exit_time=datetime(2023, 1, 1, 15, 0),
            entry_price=100.0,
            exit_price=95.0,
            quantity=10.0,
            trade_type="SELL",
            transaction_cost=2.0,
        )

        # P&L = (entry_price - exit_price) * quantity - transaction_cost
        # P&L = (100 - 95) * 10 - 2 = 50 - 2 = 48
        assert trade.pnl == 48.0

    def test_losing_trade_pnl(self):
        """Test P&L calculation for losing trade."""
        trade = Trade(
            entry_time=datetime(2023, 1, 1, 10, 0),
            exit_time=datetime(2023, 1, 1, 15, 0),
            entry_price=100.0,
            exit_price=97.0,
            quantity=10.0,
            trade_type="BUY",
            transaction_cost=2.0,
        )

        # P&L = (97 - 100) * 10 - 2 = -30 - 2 = -32
        assert trade.pnl == -32.0

    def test_return_percentage(self):
        """Test return percentage calculation."""
        trade = Trade(
            entry_time=datetime(2023, 1, 1, 10, 0),
            exit_time=datetime(2023, 1, 1, 15, 0),
            entry_price=100.0,
            exit_price=105.0,
            quantity=10.0,
            trade_type="BUY",
            transaction_cost=2.0,
        )

        # Investment = 100 * 10 = 1000
        # Return % = (48 / 1000) * 100 = 4.8%
        assert trade.return_percentage == 4.8


class TestBacktestResults:
    """Test cases for BacktestResults class."""

    def test_empty_results(self):
        """Test empty backtest results."""
        results = BacktestResults()

        assert results.total_trades == 0
        assert results.winning_trades == []
        assert results.losing_trades == []
        assert results.equity_curve == []
        assert results.config is None

    def test_results_with_trades(self):
        """Test backtest results with trades."""
        winning_trade = Trade(
            entry_time=datetime(2023, 1, 1),
            exit_time=datetime(2023, 1, 2),
            entry_price=100.0,
            exit_price=105.0,
            quantity=10.0,
            trade_type="BUY",
        )

        losing_trade = Trade(
            entry_time=datetime(2023, 1, 3),
            exit_time=datetime(2023, 1, 4),
            entry_price=100.0,
            exit_price=95.0,
            quantity=10.0,
            trade_type="BUY",
        )

        results = BacktestResults(trades=[winning_trade, losing_trade])

        assert results.total_trades == 2
        assert len(results.winning_trades) == 1
        assert len(results.losing_trades) == 1
        assert results.winning_trades[0] == winning_trade
        assert results.losing_trades[0] == losing_trade


class TestPerformanceAnalyzer:
    """Test cases for PerformanceAnalyzer class."""

    def create_sample_trades(self):
        """Create sample trades for testing."""
        trades = [
            # Winning trade
            Trade(
                entry_time=datetime(2023, 1, 1),
                exit_time=datetime(2023, 1, 2),
                entry_price=100.0,
                exit_price=110.0,
                quantity=10.0,
                trade_type="BUY",
                transaction_cost=2.0,
            ),
            # Losing trade
            Trade(
                entry_time=datetime(2023, 1, 3),
                exit_time=datetime(2023, 1, 4),
                entry_price=100.0,
                exit_price=90.0,
                quantity=10.0,
                trade_type="BUY",
                transaction_cost=2.0,
            ),
            # Another winning trade
            Trade(
                entry_time=datetime(2023, 1, 5),
                exit_time=datetime(2023, 1, 6),
                entry_price=100.0,
                exit_price=105.0,
                quantity=10.0,
                trade_type="BUY",
                transaction_cost=2.0,
            ),
        ]
        return trades

    def test_empty_results_metrics(self):
        """Test metrics calculation with no trades."""
        results = BacktestResults()
        analyzer = PerformanceAnalyzer(results, initial_capital=10000.0)

        metrics = analyzer.calculate_metrics()

        assert metrics["total_trades"] == 0
        assert metrics["winning_trades"] == 0
        assert metrics["losing_trades"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["total_return"] == 0.0
        assert metrics["total_return_percentage"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["max_drawdown"] == 0.0

    def test_metrics_with_trades(self):
        """Test metrics calculation with sample trades."""
        trades = self.create_sample_trades()
        results = BacktestResults(trades=trades)
        analyzer = PerformanceAnalyzer(results, initial_capital=10000.0)

        metrics = analyzer.calculate_metrics()

        # Basic trade stats
        assert metrics["total_trades"] == 3
        assert metrics["winning_trades"] == 2
        assert metrics["losing_trades"] == 1
        assert metrics["win_rate"] == 2 / 3

        # Return metrics
        # Trade 1: (110-100)*10 - 2 = 98
        # Trade 2: (90-100)*10 - 2 = -102
        # Trade 3: (105-100)*10 - 2 = 48
        # Total = 98 - 102 + 48 = 44
        assert metrics["total_return"] == 44.0
        assert metrics["total_return_percentage"] == 0.44  # 44/10000 * 100
        assert metrics["final_capital"] == 10044.0

        # Check that other metrics are calculated
        assert "average_trade_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "profit_factor" in metrics

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        trades = [
            Trade(
                entry_time=datetime(2023, 1, 1),
                exit_time=datetime(2023, 1, 2),
                entry_price=100.0,
                exit_price=105.0,
                quantity=10.0,
                trade_type="BUY",
            )
        ]

        results = BacktestResults(trades=trades)
        analyzer = PerformanceAnalyzer(results, initial_capital=10000.0)

        metrics = analyzer.calculate_metrics()

        # With only one trade, standard deviation is 0, so Sharpe ratio should be 0
        assert metrics["sharpe_ratio"] == 0.0

    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""
        trades = self.create_sample_trades()
        results = BacktestResults(trades=trades)
        analyzer = PerformanceAnalyzer(results, initial_capital=10000.0)

        metrics = analyzer.calculate_metrics()

        # Gross profit: 98 + 48 = 146
        # Gross loss: 102
        # Profit factor: 146 / 102 ≈ 1.43
        expected_profit_factor = 146.0 / 102.0
        assert abs(metrics["profit_factor"] - expected_profit_factor) < 0.01

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create equity curve with drawdown
        equity_curve = [
            (datetime(2023, 1, 1), 10000.0),
            (datetime(2023, 1, 2), 10500.0),  # Peak
            (datetime(2023, 1, 3), 10200.0),  # Drawdown
            (datetime(2023, 1, 4), 9800.0),  # Max drawdown: 700
            (datetime(2023, 1, 5), 10100.0),
        ]

        results = BacktestResults(equity_curve=equity_curve)
        analyzer = PerformanceAnalyzer(results, initial_capital=10000.0)

        max_dd = analyzer._calculate_max_drawdown()

        # Max drawdown should be 10500 - 9800 = 700
        assert max_dd == 700.0

    def test_consecutive_trades_calculation(self):
        """Test consecutive wins/losses calculation."""
        # Create trades with pattern: Win, Win, Loss, Win, Loss, Loss
        trades = [
            Trade(
                entry_time=datetime(2023, 1, 1),
                exit_time=datetime(2023, 1, 2),
                entry_price=100.0,
                exit_price=105.0,
                quantity=10.0,
                trade_type="BUY",
            ),  # Win
            Trade(
                entry_time=datetime(2023, 1, 2),
                exit_time=datetime(2023, 1, 3),
                entry_price=100.0,
                exit_price=108.0,
                quantity=10.0,
                trade_type="BUY",
            ),  # Win
            Trade(
                entry_time=datetime(2023, 1, 3),
                exit_time=datetime(2023, 1, 4),
                entry_price=100.0,
                exit_price=95.0,
                quantity=10.0,
                trade_type="BUY",
            ),  # Loss
            Trade(
                entry_time=datetime(2023, 1, 4),
                exit_time=datetime(2023, 1, 5),
                entry_price=100.0,
                exit_price=103.0,
                quantity=10.0,
                trade_type="BUY",
            ),  # Win
            Trade(
                entry_time=datetime(2023, 1, 5),
                exit_time=datetime(2023, 1, 6),
                entry_price=100.0,
                exit_price=97.0,
                quantity=10.0,
                trade_type="BUY",
            ),  # Loss
            Trade(
                entry_time=datetime(2023, 1, 6),
                exit_time=datetime(2023, 1, 7),
                entry_price=100.0,
                exit_price=96.0,
                quantity=10.0,
                trade_type="BUY",
            ),  # Loss
        ]

        results = BacktestResults(trades=trades)
        analyzer = PerformanceAnalyzer(results, initial_capital=10000.0)

        max_wins, max_losses = analyzer._calculate_consecutive_trades()

        # Max consecutive wins: 2 (first two trades)
        # Max consecutive losses: 2 (last two trades)
        assert max_wins == 2
        assert max_losses == 2

    def test_generate_report(self):
        """Test report generation."""
        trades = self.create_sample_trades()
        results = BacktestResults(trades=trades)
        analyzer = PerformanceAnalyzer(results, initial_capital=10000.0)

        report = analyzer.generate_report()

        assert "BACKTESTING PERFORMANCE REPORT" in report
        assert "Total Trades: 3" in report
        assert "Win Rate:" in report
        assert "Total Return:" in report
        assert "Sharpe Ratio:" in report
        assert "Max Drawdown:" in report

    def test_get_trade_summary(self):
        """Test trade summary DataFrame generation."""
        trades = self.create_sample_trades()
        results = BacktestResults(trades=trades)
        analyzer = PerformanceAnalyzer(results, initial_capital=10000.0)

        df = analyzer.get_trade_summary()

        assert len(df) == 3
        assert "trade_id" in df.columns
        assert "entry_time" in df.columns
        assert "exit_time" in df.columns
        assert "pnl" in df.columns
        assert "return_pct" in df.columns

        # Check first trade data
        assert df.iloc[0]["trade_id"] == 1
        assert df.iloc[0]["entry_price"] == 100.0
        assert df.iloc[0]["exit_price"] == 110.0
        assert df.iloc[0]["pnl"] == 98.0
