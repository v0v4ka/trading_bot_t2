"""Test suite for backtesting engine."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.schemas import Signal, TradingDecision
from src.backtesting.config import BacktestConfig
from src.backtesting.engine import BacktestEngine, run_simple_backtest
from src.backtesting.metrics import BacktestResults, Trade
from src.data.models import OHLCV


class TestBacktestEngine:
    """Test cases for BacktestEngine class."""

    def create_sample_config(self):
        """Create a sample configuration for testing."""
        return BacktestConfig(
            symbol="AAPL",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 2, 1),
            initial_capital=10000.0,
            position_size=1000.0,
            signal_confidence_threshold=0.6,
            decision_confidence_threshold=0.7,
        )

    def create_sample_dataframe(self):
        """Create sample OHLCV DataFrame for testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")
        data = []

        for date in dates:
            # Generate realistic OHLCV data
            base_price = 150.0 + np.random.normal(0, 5)
            high = base_price + np.random.uniform(0, 2)
            low = base_price - np.random.uniform(0, 2)
            open_price = base_price + np.random.uniform(-1, 1)
            close_price = base_price + np.random.uniform(-1, 1)
            volume = int(np.random.uniform(1000000, 5000000))

            data.append(
                {
                    "timestamp": date,
                    "Open": open_price,
                    "High": high,
                    "Low": low,
                    "Close": close_price,
                    "Volume": volume,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def test_engine_initialization(self):
        """Test engine initialization with valid config."""
        config = self.create_sample_config()
        engine = BacktestEngine(config)

        assert engine.config == config
        assert engine.current_capital == config.initial_capital
        assert engine.open_positions == []
        assert isinstance(engine.results, BacktestResults)
        assert engine.results.config == config.to_dict()

    def test_engine_initialization_invalid_config(self):
        """Test engine initialization with invalid config."""
        invalid_config = BacktestConfig(initial_capital=-1000.0)

        with pytest.raises(ValueError) as exc_info:
            BacktestEngine(invalid_config)

        assert "Invalid configuration" in str(exc_info.value)

    @patch("src.backtesting.engine.DataProvider")
    def test_load_historical_data(self, mock_data_provider):
        """Test loading historical data."""
        config = self.create_sample_config()
        sample_df = self.create_sample_dataframe()

        # Create mock OHLCVSeries that returns the sample DataFrame
        mock_series = Mock()
        mock_series.to_dataframe.return_value = sample_df

        # Mock data provider
        mock_provider_instance = Mock()
        mock_provider_instance.fetch.return_value = mock_series
        mock_data_provider.return_value = mock_provider_instance

        engine = BacktestEngine(config)
        df = engine._load_historical_data()

        assert not df.empty
        assert len(df) == len(sample_df)
        mock_provider_instance.fetch.assert_called_once()

    @patch("src.backtesting.engine.DataProvider")
    def test_load_historical_data_empty(self, mock_data_provider):
        """Test loading historical data when no data is available."""
        config = self.create_sample_config()

        # Mock empty data
        mock_provider_instance = Mock()
        mock_provider_instance.get_historical_data.return_value = pd.DataFrame()
        mock_data_provider.return_value = mock_provider_instance

        engine = BacktestEngine(config)

        with pytest.raises(ValueError) as exc_info:
            engine._load_historical_data()

        assert "No data available" in str(exc_info.value)

    @patch("src.backtesting.engine.SignalDetectionAgent")
    @patch("src.backtesting.engine.DecisionMakerAgent")
    def test_initialize_agents(self, mock_decision_agent, mock_signal_agent):
        """Test agent initialization."""
        config = self.create_sample_config()
        engine = BacktestEngine(config)
        sample_df = self.create_sample_dataframe()

        # Mock the agent classes
        mock_signal_instance = Mock()
        mock_decision_instance = Mock()
        mock_signal_agent.return_value = mock_signal_instance
        mock_decision_agent.return_value = mock_decision_instance

        engine._initialize_agents(sample_df)

        # Verify agents were initialized
        assert engine.signal_agent is not None
        assert engine.decision_agent is not None

        # Verify agents were called with correct parameters
        mock_signal_agent.assert_called_once()
        mock_decision_agent.assert_called_once()

    def test_calculate_stop_loss(self):
        """Test stop loss calculation."""
        config = self.create_sample_config()
        engine = BacktestEngine(config)

        # Test BUY trade stop loss
        buy_stop_loss = engine._calculate_stop_loss(100.0, "BUY")
        expected_buy_stop_loss = 100.0 * (1 - config.stop_loss_percentage)
        assert buy_stop_loss == expected_buy_stop_loss

        # Test SELL trade stop loss
        sell_stop_loss = engine._calculate_stop_loss(100.0, "SELL")
        expected_sell_stop_loss = 100.0 * (1 + config.stop_loss_percentage)
        assert sell_stop_loss == expected_sell_stop_loss

    def test_update_equity_curve(self):
        """Test equity curve update."""
        config = self.create_sample_config()
        engine = BacktestEngine(config)

        timestamp = datetime(2023, 1, 15)
        initial_length = len(engine.results.equity_curve)

        engine._update_equity_curve(timestamp)

        assert len(engine.results.equity_curve) == initial_length + 1
        assert engine.results.equity_curve[-1][0] == timestamp
        assert engine.results.equity_curve[-1][1] == config.initial_capital

    @patch("src.backtesting.engine.SignalDetectionAgent")
    @patch("src.backtesting.engine.DecisionMakerAgent")
    def test_get_signals_at_timestamp(self, mock_decision_agent, mock_signal_agent):
        """Test getting signals at specific timestamp."""
        config = self.create_sample_config()
        engine = BacktestEngine(config)
        sample_df = self.create_sample_dataframe()

        # Mock signal agent
        mock_signal_instance = Mock()
        mock_signals = [
            Signal(
                timestamp=datetime(2023, 1, 15),
                type="BUY",
                confidence=0.8,
                details={"indicator": "fractal"},
            )
        ]
        mock_signal_instance.detect_signals.return_value = mock_signals
        mock_signal_agent.return_value = mock_signal_instance

        engine._initialize_agents(sample_df)
        signals = engine._get_signals_at_timestamp(15, sample_df)

        assert isinstance(signals, list)
        # Should return some signals (mocked to return last 3)

    def test_get_decisions(self):
        """Test getting decisions from Decision Maker Agent."""
        config = self.create_sample_config()
        engine = BacktestEngine(config)

        # Create mock signals
        signals = [
            Signal(
                timestamp=datetime(2023, 1, 15),
                type="BUY",
                confidence=0.8,
                details={"indicator": "fractal"},
            )
        ]

        # Create mock current bar
        current_bar = pd.Series(
            {
                "timestamp": datetime(2023, 1, 15),
                "Open": 150.0,
                "High": 152.0,
                "Low": 148.0,
                "Close": 151.0,
                "Volume": 1000000,
            }
        )

        # Mock decision agent
        mock_decision_agent = Mock()
        mock_decision = TradingDecision(
            action="BUY",
            confidence=0.8,
            reasoning="Strong signal confluence",
            signals_used=signals,
            timestamp=datetime(2023, 1, 15),
            entry_price=151.0,
        )
        mock_decision_agent.make_decision.return_value = mock_decision
        engine.decision_agent = mock_decision_agent

        decisions = engine._get_decisions(signals, current_bar)

        assert len(decisions) == 1
        assert decisions[0].action == "BUY"
        assert decisions[0].confidence >= config.decision_confidence_threshold

    def test_execute_single_trade_insufficient_capital(self):
        """Test trade execution with insufficient capital."""
        config = self.create_sample_config()
        engine = BacktestEngine(config)
        engine.current_capital = 50.0  # Very low capital

        decision = TradingDecision(
            action="BUY",
            confidence=0.8,
            reasoning="Test trade",
            signals_used=[],
            timestamp=datetime(2023, 1, 15),
            entry_price=1000.0,  # High price
        )

        current_bar = pd.Series({"Close": 1000.0, "timestamp": datetime(2023, 1, 15)})

        initial_trades = len(engine.results.trades)
        engine._execute_single_trade(decision, current_bar, datetime(2023, 1, 15))

        # Should not execute trade due to insufficient capital
        assert len(engine.results.trades) == initial_trades

    def test_close_all_positions(self):
        """Test closing all positions at end of backtest."""
        config = self.create_sample_config()
        engine = BacktestEngine(config)

        # Add mock open position
        mock_position = {
            "entry_time": datetime(2023, 1, 10),
            "entry_price": 150.0,
            "quantity": 10.0,
            "trade_type": "BUY",
            "transaction_cost": 2.0,
            "stop_loss_price": 142.5,
        }
        engine.open_positions.append(mock_position)

        final_bar = pd.Series(
            {"Close": 155.0, "timestamp": datetime(2023, 1, 30)},
            name=datetime(2023, 1, 30),
        )

        initial_trades = len(engine.results.trades)
        engine._close_all_positions(final_bar)

        assert len(engine.open_positions) == 0
        assert len(engine.results.trades) == initial_trades + 1

    def test_get_performance_analyzer(self):
        """Test getting performance analyzer."""
        config = self.create_sample_config()
        engine = BacktestEngine(config)

        analyzer = engine.get_performance_analyzer()

        assert analyzer is not None
        assert analyzer.initial_capital == config.initial_capital
        assert analyzer.results == engine.results

    def test_generate_summary_report(self):
        """Test generating summary report."""
        config = self.create_sample_config()
        engine = BacktestEngine(config)

        # Add a sample trade
        trade = Trade(
            entry_time=datetime(2023, 1, 10),
            exit_time=datetime(2023, 1, 11),
            entry_price=150.0,
            exit_price=155.0,
            quantity=10.0,
            trade_type="BUY",
            transaction_cost=2.0,
        )
        engine.results.trades.append(trade)

        report = engine.generate_summary_report()

        assert "BACKTESTING PERFORMANCE REPORT" in report
        assert "Total Trades: 1" in report


class TestRunSimpleBacktest:
    """Test cases for the run_simple_backtest convenience function."""

    @patch("src.backtesting.engine.BacktestEngine")
    def test_run_simple_backtest_default(self, mock_engine_class):
        """Test run_simple_backtest with default parameters."""
        mock_engine = Mock()
        mock_results = BacktestResults()
        mock_engine.run_backtest.return_value = mock_results
        mock_engine_class.return_value = mock_engine

        results = run_simple_backtest()

        # Verify engine was created and backtest was run
        mock_engine_class.assert_called_once()
        mock_engine.run_backtest.assert_called_once()
        assert results == mock_results

        # Check that config was created with default symbol and date range
        args, kwargs = mock_engine_class.call_args
        config = args[0]
        assert config.symbol == "AAPL"
        assert isinstance(config.start_date, datetime)
        assert isinstance(config.end_date, datetime)

    @patch("src.backtesting.engine.BacktestEngine")
    def test_run_simple_backtest_custom(self, mock_engine_class):
        """Test run_simple_backtest with custom parameters."""
        mock_engine = Mock()
        mock_results = BacktestResults()
        mock_engine.run_backtest.return_value = mock_results
        mock_engine_class.return_value = mock_engine

        results = run_simple_backtest(
            symbol="TSLA", days=180, initial_capital=20000.0, position_size=2000.0
        )

        # Verify custom parameters were used
        args, kwargs = mock_engine_class.call_args
        config = args[0]
        assert config.symbol == "TSLA"
        assert config.initial_capital == 20000.0
        assert config.position_size == 2000.0

        # Check that date range is approximately 180 days
        date_diff = config.end_date - config.start_date
        assert abs(date_diff.days - 180) <= 1  # Allow for small variations
