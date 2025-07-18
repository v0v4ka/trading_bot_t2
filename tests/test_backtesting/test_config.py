"""Test suite for backtesting configuration module."""

from datetime import datetime, timedelta

import pytest

from src.backtesting.config import (
    DEFAULT_SCENARIOS,
    BacktestConfig,
    ScenarioConfig,
    create_custom_config,
    get_scenario_config,
)


class TestBacktestConfig:
    """Test cases for BacktestConfig class."""

    def test_default_config_creation(self):
        """Test creating a config with default values."""
        config = BacktestConfig()

        assert config.symbol == "AAPL"
        assert config.timeframe == "1d"
        assert config.initial_capital == 10000.0
        assert config.position_size == 1000.0
        assert config.transaction_cost == 0.001
        assert config.signal_confidence_threshold == 0.6
        assert config.decision_confidence_threshold == 0.7
        assert config.use_llm_confirmation is True
        assert config.max_position_size == 0.2
        assert config.stop_loss_percentage == 0.05

    def test_custom_config_creation(self):
        """Test creating a config with custom values."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        config = BacktestConfig(
            symbol="TSLA",
            timeframe="1h",
            start_date=start_date,
            end_date=end_date,
            initial_capital=50000.0,
            position_size=5000.0,
            signal_confidence_threshold=0.8,
            decision_confidence_threshold=0.9,
        )

        assert config.symbol == "TSLA"
        assert config.timeframe == "1h"
        assert config.start_date == start_date
        assert config.end_date == end_date
        assert config.initial_capital == 50000.0
        assert config.position_size == 5000.0
        assert config.signal_confidence_threshold == 0.8
        assert config.decision_confidence_threshold == 0.9

    def test_config_validation_valid(self):
        """Test validation of valid configuration."""
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=10000.0,
            position_size=1000.0,
            signal_confidence_threshold=0.7,
            decision_confidence_threshold=0.8,
        )

        errors = config.validate()
        assert errors == []

    def test_config_validation_invalid_dates(self):
        """Test validation with invalid date range."""
        config = BacktestConfig(
            start_date=datetime(2023, 12, 31),
            end_date=datetime(2023, 1, 1),  # End before start
        )

        errors = config.validate()
        assert "start_date must be before end_date" in errors

    def test_config_validation_invalid_capital(self):
        """Test validation with invalid capital."""
        config = BacktestConfig(initial_capital=-1000.0)

        errors = config.validate()
        assert "initial_capital must be positive" in errors

    def test_config_validation_invalid_position_size(self):
        """Test validation with invalid position size."""
        config = BacktestConfig(position_size=-500.0)

        errors = config.validate()
        assert "position_size must be positive" in errors

    def test_config_validation_invalid_confidence_thresholds(self):
        """Test validation with invalid confidence thresholds."""
        config = BacktestConfig(
            signal_confidence_threshold=1.5,  # > 1
            decision_confidence_threshold=-0.1,  # < 0
        )

        errors = config.validate()
        assert "signal_confidence_threshold must be between 0 and 1" in errors
        assert "decision_confidence_threshold must be between 0 and 1" in errors

    def test_config_validation_invalid_max_position_size(self):
        """Test validation with invalid max position size."""
        config = BacktestConfig(max_position_size=1.5)  # > 1

        errors = config.validate()
        assert "max_position_size must be between 0 and 1" in errors

    def test_config_validation_invalid_stop_loss(self):
        """Test validation with invalid stop loss."""
        config = BacktestConfig(stop_loss_percentage=-0.05)

        errors = config.validate()
        assert "stop_loss_percentage must be non-negative" in errors

    def test_config_to_dict(self):
        """Test conversion to dictionary."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        config = BacktestConfig(symbol="MSFT", start_date=start_date, end_date=end_date)

        config_dict = config.to_dict()

        assert config_dict["symbol"] == "MSFT"
        assert config_dict["start_date"] == start_date.isoformat()
        assert config_dict["end_date"] == end_date.isoformat()
        assert "initial_capital" in config_dict
        assert "position_size" in config_dict


class TestScenarioConfig:
    """Test cases for ScenarioConfig class."""

    def test_scenario_config_creation(self):
        """Test creating a scenario config."""
        config = BacktestConfig(symbol="TSLA")
        scenario = ScenarioConfig(
            name="Test Scenario", description="A test scenario", config=config
        )

        assert scenario.name == "Test Scenario"
        assert scenario.description == "A test scenario"
        assert scenario.config.symbol == "TSLA"
        assert scenario.expected_metrics is None

    def test_scenario_config_with_invalid_config(self):
        """Test scenario config with invalid configuration."""
        invalid_config = BacktestConfig(initial_capital=-1000.0)

        with pytest.raises(ValueError) as exc_info:
            ScenarioConfig(
                name="Invalid Scenario",
                description="Invalid scenario",
                config=invalid_config,
            )

        assert "Invalid configuration for scenario 'Invalid Scenario'" in str(
            exc_info.value
        )


class TestPredefinedScenarios:
    """Test cases for predefined scenarios."""

    def test_default_scenarios_exist(self):
        """Test that default scenarios are defined."""
        assert "conservative" in DEFAULT_SCENARIOS
        assert "aggressive" in DEFAULT_SCENARIOS
        assert "balanced" in DEFAULT_SCENARIOS

    def test_conservative_scenario(self):
        """Test conservative scenario configuration."""
        scenario = DEFAULT_SCENARIOS["conservative"]

        assert scenario.name == "Conservative Strategy"
        assert scenario.config.signal_confidence_threshold == 0.8
        assert scenario.config.decision_confidence_threshold == 0.8
        assert scenario.config.position_size == 500.0
        assert scenario.config.stop_loss_percentage == 0.03
        assert scenario.config.max_position_size == 0.1

    def test_aggressive_scenario(self):
        """Test aggressive scenario configuration."""
        scenario = DEFAULT_SCENARIOS["aggressive"]

        assert scenario.name == "Aggressive Strategy"
        assert scenario.config.signal_confidence_threshold == 0.5
        assert scenario.config.decision_confidence_threshold == 0.6
        assert scenario.config.position_size == 2000.0
        assert scenario.config.stop_loss_percentage == 0.08
        assert scenario.config.max_position_size == 0.3

    def test_balanced_scenario(self):
        """Test balanced scenario configuration."""
        scenario = DEFAULT_SCENARIOS["balanced"]

        assert scenario.name == "Balanced Strategy"
        assert scenario.config.signal_confidence_threshold == 0.65
        assert scenario.config.decision_confidence_threshold == 0.7
        assert scenario.config.position_size == 1000.0
        assert scenario.config.stop_loss_percentage == 0.05
        assert scenario.config.max_position_size == 0.2

    def test_all_scenarios_valid(self):
        """Test that all predefined scenarios have valid configurations."""
        for scenario_name, scenario in DEFAULT_SCENARIOS.items():
            errors = scenario.config.validate()
            assert (
                errors == []
            ), f"Scenario '{scenario_name}' has invalid configuration: {errors}"


class TestScenarioFunctions:
    """Test cases for scenario utility functions."""

    def test_get_scenario_config_valid(self):
        """Test getting a valid scenario configuration."""
        scenario = get_scenario_config("conservative")

        assert scenario.name == "Conservative Strategy"
        assert scenario.config.signal_confidence_threshold == 0.8

    def test_get_scenario_config_invalid(self):
        """Test getting an invalid scenario configuration."""
        with pytest.raises(ValueError) as exc_info:
            get_scenario_config("nonexistent")

        assert "Unknown scenario 'nonexistent'" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_create_custom_config(self):
        """Test creating custom configuration with overrides."""
        config = create_custom_config(
            symbol="GOOGL", initial_capital=25000.0, position_size=2500.0
        )

        assert config.symbol == "GOOGL"
        assert config.initial_capital == 25000.0
        assert config.position_size == 2500.0
        # Check that other values are defaults
        assert config.timeframe == "1d"
        assert config.signal_confidence_threshold == 0.6
