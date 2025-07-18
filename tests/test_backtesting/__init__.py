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
