"""
Environment configuration loader for trading bot.

This module provides utilities to load and validate environment variables
with proper defaults and type conversion.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    Load environment variables from .env file.

    Args:
        env_path: Optional path to .env file. Defaults to .env in project root.
    """
    if env_path is None:
        env_path = Path(__file__).parent.parent / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        print(f"Warning: .env file not found at {env_path}")
        return

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Only set if not already in environment
                if key not in os.environ:
                    os.environ[key] = value


def get_env_var(
    key: str, default: Any = None, var_type: type = str, required: bool = False
) -> Any:
    """
    Get environment variable with type conversion and validation.

    Args:
        key: Environment variable name
        default: Default value if not found
        var_type: Type to convert to (str, int, float, bool)
        required: Whether the variable is required

    Returns:
        Environment variable value converted to specified type

    Raises:
        ValueError: If required variable is missing or conversion fails
    """
    value = os.getenv(key)

    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{key}' not found")
        return default

    if var_type == bool:
        return value.lower() in ("true", "1", "yes", "on")
    elif var_type == int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Cannot convert '{key}={value}' to int")
    elif var_type == float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Cannot convert '{key}={value}' to float")
    else:
        return str(value)


class TradingBotConfig:
    """Configuration class that loads all trading bot settings from environment."""

    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration by loading environment variables."""
        load_env_file(env_file)

        # OpenAI Configuration
        self.openai_api_key = get_env_var("OPENAI_API_KEY", required=False)
        self.openai_model = get_env_var("OPENAI_MODEL", "gpt-4o")
        self.openai_temperature = get_env_var("OPENAI_TEMPERATURE", 0.1, float)
        self.openai_max_tokens = get_env_var("OPENAI_MAX_TOKENS", 500, int)

        # Trading Bot Configuration
        self.test_mode = get_env_var("TBOT_TEST_MODE", True, bool)
        self.log_level = get_env_var("TBOT_LOG_LEVEL", "INFO")
        self.log_file = get_env_var("TBOT_LOG_FILE", "logs/trading_bot.log")

        # Backtesting Configuration
        self.default_initial_capital = get_env_var(
            "DEFAULT_INITIAL_CAPITAL", 10000.0, float
        )
        self.default_backtest_days = get_env_var("DEFAULT_BACKTEST_DAYS", 60, int)
        self.default_symbol = get_env_var("DEFAULT_SYMBOL", "AAPL")

        # Agent Configuration
        self.signal_confidence_threshold = get_env_var(
            "SIGNAL_CONFIDENCE_THRESHOLD", 0.6, float
        )
        self.decision_confidence_threshold = get_env_var(
            "DECISION_CONFIDENCE_THRESHOLD", 0.7, float
        )
        self.use_llm_confirmation = get_env_var("USE_LLM_CONFIRMATION", True, bool)

        # Chart Configuration
        self.default_chart_theme = get_env_var("DEFAULT_CHART_THEME", "professional")
        self.default_chart_width = get_env_var("DEFAULT_CHART_WIDTH", 16, int)
        self.default_chart_height = get_env_var("DEFAULT_CHART_HEIGHT", 12, int)

        # Performance Configuration
        self.max_concurrent_requests = get_env_var("MAX_CONCURRENT_REQUESTS", 5, int)
        self.request_timeout = get_env_var("REQUEST_TIMEOUT", 30, int)
        self.max_retries = get_env_var("MAX_RETRIES", 3, int)

        # Directory Paths (deprecated config_dir - now using .env only)
        self.logs_dir = get_env_var("LOGS_DIR", "logs/")
        self.outputs_dir = get_env_var("OUTPUTS_DIR", "outputs/")
        self.cache_dir = get_env_var("CACHE_DIR", "cache/")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith("_") and not callable(getattr(self, key))
        }

    def is_production_mode(self) -> bool:
        """Check if running in production mode (LLM enabled)."""
        return not self.test_mode and bool(self.openai_api_key)

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.test_mode and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required when TBOT_TEST_MODE=false")

        if self.signal_confidence_threshold < 0 or self.signal_confidence_threshold > 1:
            errors.append("SIGNAL_CONFIDENCE_THRESHOLD must be between 0 and 1")

        if (
            self.decision_confidence_threshold < 0
            or self.decision_confidence_threshold > 1
        ):
            errors.append("DECISION_CONFIDENCE_THRESHOLD must be between 0 and 1")

        if self.default_initial_capital <= 0:
            errors.append("DEFAULT_INITIAL_CAPITAL must be positive")

        return errors


# Global configuration instance
config = TradingBotConfig()


# Convenience functions
def get_openai_model() -> str:
    """Get the configured OpenAI model."""
    return config.openai_model


def is_test_mode() -> bool:
    """Check if running in test mode."""
    return config.test_mode


def get_api_key() -> Optional[str]:
    """Get the OpenAI API key."""
    return config.openai_api_key
