"""Test suite for backtesting configuration module."""

import pytest
from datetime import datetime, timedelta
from src.backtesting.config import (
    BacktestConfig, 
    ScenarioConfig, 
    DEFAULT_SCENARIOS,
    get_scenario_config,
    create_custom_config
)
