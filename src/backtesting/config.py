"""
Backtesting configuration management.

This module provides configuration classes for setting up backtesting scenarios
with flexible parameters for timeframes, symbols, date ranges, and agent settings.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class BacktestConfig:
    """Configuration for backtesting scenarios."""

    # Data configuration
    symbol: str = "AAPL"
    timeframe: str = "1d"  # 1d, 1h, 15m, etc.
    start_date: datetime = field(
        default_factory=lambda: datetime.now() - timedelta(days=365)
    )
    end_date: datetime = field(default_factory=datetime.now)

    # Trading configuration
    initial_capital: float = 10000.0
    position_size: float = 1000.0  # Fixed position size for MVP
    transaction_cost: float = 0.001  # 0.1% transaction cost

    # Agent configuration
    signal_confidence_threshold: float = 0.6
    decision_confidence_threshold: float = 0.7
    use_llm_confirmation: bool = True

    # Risk management (basic for MVP)
    max_position_size: float = 0.2  # 20% of capital per trade
    stop_loss_percentage: float = 0.05  # 5% stop loss

    # Backtesting behavior
    allow_partial_fills: bool = False
    require_confirmation: bool = True  # Require both signal and decision

    def validate(self) -> List[str]:
        """Validate configuration parameters and return any errors."""
        errors = []

        if self.start_date >= self.end_date:
            errors.append("start_date must be before end_date")

        if self.initial_capital <= 0:
            errors.append("initial_capital must be positive")

        if self.position_size <= 0:
            errors.append("position_size must be positive")

        if not 0 <= self.signal_confidence_threshold <= 1:
            errors.append("signal_confidence_threshold must be between 0 and 1")

        if not 0 <= self.decision_confidence_threshold <= 1:
            errors.append("decision_confidence_threshold must be between 0 and 1")

        if not 0 <= self.max_position_size <= 1:
            errors.append("max_position_size must be between 0 and 1")

        if self.stop_loss_percentage < 0:
            errors.append("stop_loss_percentage must be non-negative")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "position_size": self.position_size,
            "transaction_cost": self.transaction_cost,
            "signal_confidence_threshold": self.signal_confidence_threshold,
            "decision_confidence_threshold": self.decision_confidence_threshold,
            "use_llm_confirmation": self.use_llm_confirmation,
            "max_position_size": self.max_position_size,
            "stop_loss_percentage": self.stop_loss_percentage,
            "allow_partial_fills": self.allow_partial_fills,
            "require_confirmation": self.require_confirmation,
        }


@dataclass
class ScenarioConfig:
    """Configuration for specific backtesting scenarios."""

    name: str
    description: str
    config: BacktestConfig
    expected_metrics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        errors = self.config.validate()
        if errors:
            raise ValueError(
                f"Invalid configuration for scenario '{self.name}': {', '.join(errors)}"
            )


# Predefined scenario configurations
DEFAULT_SCENARIOS = {
    "conservative": ScenarioConfig(
        name="Conservative Strategy",
        description="Low-risk strategy with high confidence thresholds",
        config=BacktestConfig(
            signal_confidence_threshold=0.8,
            decision_confidence_threshold=0.8,
            position_size=500.0,
            stop_loss_percentage=0.03,
            max_position_size=0.1,
        ),
    ),
    "aggressive": ScenarioConfig(
        name="Aggressive Strategy",
        description="Higher-risk strategy with lower confidence thresholds",
        config=BacktestConfig(
            signal_confidence_threshold=0.5,
            decision_confidence_threshold=0.6,
            position_size=2000.0,
            stop_loss_percentage=0.08,
            max_position_size=0.3,
        ),
    ),
    "balanced": ScenarioConfig(
        name="Balanced Strategy",
        description="Balanced approach with moderate risk",
        config=BacktestConfig(
            signal_confidence_threshold=0.65,
            decision_confidence_threshold=0.7,
            position_size=1000.0,
            stop_loss_percentage=0.05,
            max_position_size=0.2,
        ),
    ),
}


def get_scenario_config(scenario_name: str) -> ScenarioConfig:
    """Get a predefined scenario configuration by name."""
    if scenario_name not in DEFAULT_SCENARIOS:
        available = ", ".join(DEFAULT_SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")

    return DEFAULT_SCENARIOS[scenario_name]


def create_custom_config(**kwargs) -> BacktestConfig:
    """Create a custom backtesting configuration with overrides."""
    return BacktestConfig(**kwargs)
