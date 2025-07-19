"""
Core backtesting engine for historical trade simulation.

This module provides the main BacktestEngine class that processes historical data
through Signal Detection and Decision Maker agents to simulate trading performance.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..agents.decision_maker_agent import DecisionMakerAgent
from ..agents.schemas import Signal, TradingDecision
from ..agents.signal_detection_agent import SignalDetectionAgent
from ..data.data_provider import DataProvider
from ..data.models import OHLCV
from ..logging.decision_logger import DecisionLogger
from .config import BacktestConfig
from .metrics import BacktestResults, PerformanceAnalyzer, Trade

logger = logging.getLogger("trading_bot.backtesting.engine")


class BacktestEngine:
    """
    Core backtesting engine that simulates trading using historical data.

    The engine processes historical OHLCV data through Signal Detection and
    Decision Maker agents to generate trading signals and decisions, then
    simulates trade execution and tracks performance.
    """

    def __init__(
        self, config: BacktestConfig, decision_logger: Optional[DecisionLogger] = None
    ):
        """
        Initialize the backtesting engine.

        Args:
            config: Backtesting configuration
            decision_logger: Optional decision logger for agent decisions
        """
        self.config = config
        self.decision_logger = decision_logger or DecisionLogger()

        # Validate configuration
        config_errors = config.validate()
        if config_errors:
            raise ValueError(f"Invalid configuration: {', '.join(config_errors)}")

        # Initialize components (DataProvider will be initialized when needed)
        self.data_provider: Optional[DataProvider] = None
        self.signal_agent: Optional[SignalDetectionAgent] = None
        self.decision_agent: Optional[DecisionMakerAgent] = None

        # Backtesting state
        self.current_capital = config.initial_capital
        self.open_positions: List[Dict[str, Any]] = []
        self.results = BacktestResults(config=config.to_dict())

        logger.info(
            f"Initialized BacktestEngine for {config.symbol} from {config.start_date} to {config.end_date}"
        )

    def run_backtest(self) -> BacktestResults:
        """
        Run the complete backtesting process.

        Returns:
            BacktestResults: Complete backtesting results with trades and metrics
        """
        logger.info("Starting backtesting process...")
        self.results.start_time = datetime.now()

        try:
            # 1. Load historical data
            df = self._load_historical_data()

            # 2. Initialize agents with data
            self._initialize_agents(df)

            # 3. Process each time period
            self._process_historical_data(df)

            # 4. Close any remaining open positions
            self._close_all_positions(df.iloc[-1])

            # 5. Generate final equity curve
            final_timestamp = (
                df.index[-1]
                if hasattr(df.index[-1], "to_pydatetime")
                else datetime.now()
            )
            self._update_equity_curve(final_timestamp)

            self.results.end_time = datetime.now()

            logger.info(
                f"Backtesting completed. Total trades: {len(self.results.trades)}"
            )
            return self.results

        except Exception as e:
            logger.error(f"Backtesting failed: {str(e)}")
            raise

    def _load_historical_data(self) -> pd.DataFrame:
        """Load and prepare historical data."""
        logger.info(f"Loading historical data for {self.config.symbol}...")

        # Initialize data provider with config parameters
        self.data_provider = DataProvider(
            symbol=self.config.symbol,
            interval=self.config.timeframe,
            start=self.config.start_date.strftime("%Y-%m-%d"),
            end=self.config.end_date.strftime("%Y-%m-%d"),
        )

        data_series = self.data_provider.fetch()
        df = data_series.to_dataframe()

        if df.empty:
            raise ValueError(
                f"No data available for {self.config.symbol} in specified date range"
            )

        logger.info(
            f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}"
        )
        return df

    def _initialize_agents(self, df: pd.DataFrame) -> None:
        """Initialize trading agents with historical data."""
        logger.info("Initializing trading agents...")

        # Initialize Signal Detection Agent
        self.signal_agent = SignalDetectionAgent(
            df=df,
            llm_client=(
                None if not self.config.use_llm_confirmation else None
            ),  # Use default LLM
        )

        # Initialize Decision Maker Agent
        self.decision_agent = DecisionMakerAgent(
            decision_logger=self.decision_logger,
            confluence_threshold=self.config.decision_confidence_threshold,
        )

        logger.info("Agents initialized successfully")

    def _process_historical_data(self, df: pd.DataFrame) -> None:
        """Process historical data step by step."""
        logger.info("Processing historical data...")

        # Start from a point where we have enough data for indicators
        # (Bill Williams indicators need some history)
        start_idx = max(20, len(df) // 10)  # Start after 20 periods or 10% of data

        for i in range(start_idx, len(df)):
            current_bar = df.iloc[i]
            # Extract timestamp from index or column
            timestamp = (
                current_bar.name
                if hasattr(current_bar, "name") and current_bar.name is not None
                else current_bar.get("timestamp", datetime.now())
            )

            # Check for stop losses first
            self._check_stop_losses(current_bar)

            # Get signals from Signal Detection Agent
            signals = self._get_signals_at_timestamp(i, df)

            if signals:
                # Get decisions from Decision Maker Agent
                decisions = self._get_decisions(signals, current_bar)

                # Execute trades based on decisions
                self._execute_trades(decisions, current_bar, timestamp)

            # Update equity curve
            if i % max(1, len(df) // 100) == 0:  # Update every 1% of data
                self._update_equity_curve(timestamp)

        logger.info(f"Processed {len(df)} data points")

    def _get_signals_at_timestamp(self, index: int, df: pd.DataFrame) -> List[Signal]:
        """Get trading signals at a specific timestamp."""
        try:
            # Use a subset of data up to current timestamp for signal generation
            subset_df = df.iloc[: index + 1].copy()

            # Get signals from Signal Detection Agent (returns List[Dict])
            raw_signals = (
                self.signal_agent.detect_signals() if self.signal_agent else []
            )

            # Convert dict signals to Signal objects
            signals = []
            for signal_dict in raw_signals:
                try:
                    signal = Signal(
                        timestamp=signal_dict.get("timestamp", datetime.now()),
                        type=signal_dict.get("action", "HOLD").upper(),
                        confidence=signal_dict.get("confidence", 0.0),
                        details=signal_dict,
                    )
                    signals.append(signal)
                except Exception as e:
                    logger.warning(f"Failed to convert signal dict to Signal: {e}")
                    continue

            # Filter signals by confidence if threshold is provided
            if hasattr(self.config, "signal_confidence_threshold"):
                signals = [
                    s
                    for s in signals
                    if s.confidence >= self.config.signal_confidence_threshold
                ]

            # For simplicity in MVP, we'll take the most recent signals
            # In a real implementation, you'd match exact timestamps
            recent_signals = signals[-3:] if signals else []  # Last 3 signals

            return recent_signals

        except Exception as e:
            logger.warning(f"Failed to get signals at index {index}: {str(e)}")
            return []

    def _get_decisions(
        self, signals: List[Signal], current_bar: pd.Series
    ) -> List[TradingDecision]:
        """Get trading decisions from Decision Maker Agent."""
        try:
            # Convert current bar to OHLCV format for decision making
            ohlcv_data = [
                OHLCV(
                    timestamp=(
                        current_bar.name
                        if hasattr(current_bar, "name") and current_bar.name is not None
                        else current_bar.get("timestamp", datetime.now())
                    ),
                    open=current_bar["Open"],
                    high=current_bar["High"],
                    low=current_bar["Low"],
                    close=current_bar["Close"],
                    volume=current_bar["Volume"],
                )
            ]

            # Get decision from Decision Maker Agent (single decision, not multiple)
            decision = None
            if self.decision_agent:
                decision = self.decision_agent.make_decision(signals, ohlcv_data[0])

            # Filter by confidence threshold
            filtered_decisions = []
            if (
                decision
                and decision.confidence >= self.config.decision_confidence_threshold
            ):
                filtered_decisions = [decision]

            return filtered_decisions

        except Exception as e:
            logger.warning(f"Failed to get decisions: {str(e)}")
            return []

    def _execute_trades(
        self,
        decisions: List[TradingDecision],
        current_bar: pd.Series,
        timestamp: datetime,
    ) -> None:
        """Execute trades based on decisions."""
        for decision in decisions:
            if decision.action in ["BUY", "SELL"]:
                self._execute_single_trade(decision, current_bar, timestamp)

    def _execute_single_trade(
        self, decision: TradingDecision, current_bar: pd.Series, timestamp: datetime
    ) -> None:
        """Execute a single trade."""
        try:
            current_price = current_bar["Close"]

            # Calculate position size
            position_value = min(
                self.config.position_size,
                self.current_capital * self.config.max_position_size,
            )

            if position_value < current_price:  # Not enough capital
                logger.warning(f"Insufficient capital for trade at {timestamp}")
                return

            quantity = position_value / current_price
            transaction_cost = position_value * self.config.transaction_cost

            # Create position
            position = {
                "decision": decision,
                "entry_time": timestamp,
                "entry_price": current_price,
                "quantity": quantity,
                "trade_type": decision.action,
                "transaction_cost": transaction_cost,
                "stop_loss_price": self._calculate_stop_loss(
                    current_price, decision.action
                ),
            }

            # Update capital
            self.current_capital -= position_value + transaction_cost

            # For MVP, immediately close positions (no holding period)
            # In a real system, you'd hold positions and close based on exit signals
            self._close_position_immediately(position, current_bar, timestamp)

            logger.info(
                f"Executed {decision.action} trade at {timestamp}: {quantity:.4f} shares at ${current_price:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to execute trade: {str(e)}")

    def _close_position_immediately(
        self, position: Dict[str, Any], current_bar: pd.Series, timestamp: datetime
    ) -> None:
        """Close position immediately (MVP behavior)."""
        # For MVP, we simulate a quick exit after 1-3 bars
        # In reality, you'd wait for exit signals

        # Simulate exit price with small random variation
        entry_price = position["entry_price"]
        exit_price = entry_price * (
            1 + np.random.normal(0, 0.02)
        )  # ±2% random variation

        # Create trade record
        trade = Trade(
            entry_time=position["entry_time"],
            exit_time=timestamp,
            entry_price=position["entry_price"],
            exit_price=exit_price,
            quantity=position["quantity"],
            trade_type=position["trade_type"],
            transaction_cost=position["transaction_cost"],
            stop_loss_triggered=False,
        )

        # Update capital with trade result
        if position["trade_type"] == "BUY":
            proceeds = exit_price * position["quantity"]
        else:  # SELL
            proceeds = (
                position["entry_price"] * position["quantity"]
                + (position["entry_price"] - exit_price) * position["quantity"]
            )

        self.current_capital += (
            proceeds - position["transaction_cost"]
        )  # Second transaction cost

        # Record trade
        self.results.trades.append(trade)

    def _calculate_stop_loss(self, entry_price: float, trade_type: str) -> float:
        """Calculate stop loss price."""
        if trade_type == "BUY":
            return entry_price * (1 - self.config.stop_loss_percentage)
        else:  # SELL
            return entry_price * (1 + self.config.stop_loss_percentage)

    def _check_stop_losses(self, current_bar: pd.Series) -> None:
        """Check and execute stop losses for open positions."""
        current_price = current_bar["Close"]

        positions_to_close = []
        for position in self.open_positions:
            stop_loss_price = position["stop_loss_price"]

            if position["trade_type"] == "BUY" and current_price <= stop_loss_price:
                positions_to_close.append(position)
            elif position["trade_type"] == "SELL" and current_price >= stop_loss_price:
                positions_to_close.append(position)

        # Close positions that hit stop loss
        for position in positions_to_close:
            self._close_position_stop_loss(position, current_bar)
            self.open_positions.remove(position)

    def _close_position_stop_loss(
        self, position: Dict[str, Any], current_bar: pd.Series
    ) -> None:
        """Close position due to stop loss."""
        timestamp = (
            current_bar["timestamp"] if "timestamp" in current_bar else current_bar.name
        )
        stop_loss_price = position["stop_loss_price"]

        trade = Trade(
            entry_time=position["entry_time"],
            exit_time=timestamp,
            entry_price=position["entry_price"],
            exit_price=stop_loss_price,
            quantity=position["quantity"],
            trade_type=position["trade_type"],
            transaction_cost=position["transaction_cost"],
            stop_loss_triggered=True,
        )

        # Update capital
        proceeds = stop_loss_price * position["quantity"]
        self.current_capital += proceeds - position["transaction_cost"]

        self.results.trades.append(trade)
        logger.info(
            f"Stop loss triggered for {position['trade_type']} position at ${stop_loss_price:.2f}"
        )

    def _close_all_positions(self, final_bar: pd.Series) -> None:
        """Close all remaining open positions at the end of backtesting."""
        if not self.open_positions:
            return

        logger.info(f"Closing {len(self.open_positions)} remaining positions")

        for position in self.open_positions:
            self._close_position_immediately(position, final_bar, final_bar.name)

        self.open_positions.clear()

    def _update_equity_curve(self, timestamp: datetime) -> None:
        """Update the equity curve with current portfolio value."""
        # Calculate current portfolio value
        portfolio_value = self.current_capital

        # Add value of open positions
        for position in self.open_positions:
            # Simplified: assume current value equals entry value for open positions
            portfolio_value += position["entry_price"] * position["quantity"]

        self.results.equity_curve.append((timestamp, portfolio_value))

    def get_performance_analyzer(self) -> PerformanceAnalyzer:
        """Get a performance analyzer for the results."""
        return PerformanceAnalyzer(self.results, self.config.initial_capital)

    def generate_summary_report(self) -> str:
        """Generate a summary report of the backtesting results."""
        analyzer = self.get_performance_analyzer()
        return analyzer.generate_report()


def run_simple_backtest(
    symbol: str = "AAPL", days: int = 365, **config_overrides
) -> BacktestResults:
    """
    Run a simple backtest with default configuration.

    Args:
        symbol: Trading symbol
        days: Number of days to backtest
        **config_overrides: Configuration overrides

    Returns:
        BacktestResults: Results of the backtest
    """
    config = BacktestConfig(
        symbol=symbol,
        start_date=datetime.now() - timedelta(days=days),
        **config_overrides,
    )

    engine = BacktestEngine(config)
    return engine.run_backtest()
