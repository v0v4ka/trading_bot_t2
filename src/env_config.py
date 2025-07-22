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

        # Detailed Logging Configuration
        self.log_level_global = get_env_var("LOG_LEVEL", "INFO")
        self.log_format = get_env_var(
            "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.log_to_console = get_env_var("LOG_TO_CONSOLE", True, bool)
        self.log_to_file = get_env_var("LOG_TO_FILE", True, bool)
        self.log_file_max_size = get_env_var("LOG_FILE_MAX_SIZE", 10485760, int)  # 10MB
        self.log_file_backup_count = get_env_var("LOG_FILE_BACKUP_COUNT", 5, int)
        
        # Module-specific Log Levels
        self.log_level_backtesting = get_env_var("LOG_LEVEL_BACKTESTING", "INFO")
        self.log_level_agents = get_env_var("LOG_LEVEL_AGENTS", "INFO") 
        self.log_level_data = get_env_var("LOG_LEVEL_DATA", "WARNING")
        self.log_level_visualization = get_env_var("LOG_LEVEL_VISUALIZATION", "WARNING")
        self.log_level_cli = get_env_var("LOG_LEVEL_CLI", "INFO")

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

        # Backtesting Scenarios
        self.bull_market_start = get_env_var("BULL_MARKET_START", "2020-03-01")
        self.bull_market_end = get_env_var("BULL_MARKET_END", "2021-12-31")
        self.bull_market_capital = get_env_var("BULL_MARKET_CAPITAL", 10000.0, float)
        self.bull_market_description = get_env_var(
            "BULL_MARKET_DESCRIPTION", "Bull market recovery period"
        )

        self.bear_market_start = get_env_var("BEAR_MARKET_START", "2022-01-01")
        self.bear_market_end = get_env_var("BEAR_MARKET_END", "2022-12-31")
        self.bear_market_capital = get_env_var("BEAR_MARKET_CAPITAL", 10000.0, float)
        self.bear_market_description = get_env_var(
            "BEAR_MARKET_DESCRIPTION", "Bear market correction period"
        )

        self.sideways_market_start = get_env_var("SIDEWAYS_MARKET_START", "2019-01-01")
        self.sideways_market_end = get_env_var("SIDEWAYS_MARKET_END", "2019-12-31")
        self.sideways_market_capital = get_env_var(
            "SIDEWAYS_MARKET_CAPITAL", 10000.0, float
        )
        self.sideways_market_description = get_env_var(
            "SIDEWAYS_MARKET_DESCRIPTION", "Sideways/consolidation period"
        )

        # Environment Configuration
        self.environment = get_env_var("ENVIRONMENT", "development")

        # Development Environment
        self.dev_data_source = get_env_var("DEV_DATA_SOURCE", "yahoo")
        self.dev_cache_enabled = get_env_var("DEV_CACHE_ENABLED", True, bool)
        self.dev_debug_mode = get_env_var("DEV_DEBUG_MODE", True, bool)

        # Staging Environment
        self.staging_data_source = get_env_var("STAGING_DATA_SOURCE", "yahoo")
        self.staging_cache_enabled = get_env_var("STAGING_CACHE_ENABLED", True, bool)
        self.staging_debug_mode = get_env_var("STAGING_DEBUG_MODE", False, bool)

        # Production Environment
        self.prod_data_source = get_env_var("PROD_DATA_SOURCE", "premium_api")
        self.prod_cache_enabled = get_env_var("PROD_CACHE_ENABLED", False, bool)
        self.prod_debug_mode = get_env_var("PROD_DEBUG_MODE", False, bool)

        # Agent Workflow Configurations
        self.conservative_signal_confirmation_steps = get_env_var(
            "CONSERVATIVE_SIGNAL_CONFIRMATION_STEPS", 3, int
        )
        self.conservative_risk_management = get_env_var(
            "CONSERVATIVE_RISK_MANAGEMENT", "strict"
        )
        self.conservative_position_sizing = get_env_var(
            "CONSERVATIVE_POSITION_SIZING", "kelly_criterion"
        )

        self.aggressive_signal_confirmation_steps = get_env_var(
            "AGGRESSIVE_SIGNAL_CONFIRMATION_STEPS", 1, int
        )
        self.aggressive_risk_management = get_env_var(
            "AGGRESSIVE_RISK_MANAGEMENT", "moderate"
        )
        self.aggressive_position_sizing = get_env_var(
            "AGGRESSIVE_POSITION_SIZING", "fixed_percentage"
        )

        self.active_strategy = get_env_var("ACTIVE_STRATEGY", "conservative")

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

    def get_current_environment_config(self) -> Dict[str, Any]:
        """Get configuration for the current environment."""
        env = self.environment.lower()

        if env == "production":
            return {
                "data_source": self.prod_data_source,
                "cache_enabled": self.prod_cache_enabled,
                "debug_mode": self.prod_debug_mode,
            }
        elif env == "staging":
            return {
                "data_source": self.staging_data_source,
                "cache_enabled": self.staging_cache_enabled,
                "debug_mode": self.staging_debug_mode,
            }
        else:  # development (default)
            return {
                "data_source": self.dev_data_source,
                "cache_enabled": self.dev_cache_enabled,
                "debug_mode": self.dev_debug_mode,
            }

    def get_backtesting_scenario(self, scenario: str) -> Dict[str, Any]:
        """Get backtesting scenario configuration."""
        scenarios = {
            "bull_market": {
                "start": self.bull_market_start,
                "end": self.bull_market_end,
                "initial_capital": self.bull_market_capital,
                "description": self.bull_market_description,
            },
            "bear_market": {
                "start": self.bear_market_start,
                "end": self.bear_market_end,
                "initial_capital": self.bear_market_capital,
                "description": self.bear_market_description,
            },
            "sideways_market": {
                "start": self.sideways_market_start,
                "end": self.sideways_market_end,
                "initial_capital": self.sideways_market_capital,
                "description": self.sideways_market_description,
            },
        }

        if scenario not in scenarios:
            raise ValueError(
                f"Unknown backtesting scenario: {scenario}. Available: {list(scenarios.keys())}"
            )

        return scenarios[scenario]

    def get_strategy_config(self, strategy: str = None) -> Dict[str, Any]:
        """Get agent strategy configuration."""
        if strategy is None:
            strategy = self.active_strategy

        if strategy == "conservative":
            return {
                "signal_confirmation_steps": self.conservative_signal_confirmation_steps,
                "risk_management": self.conservative_risk_management,
                "position_sizing": self.conservative_position_sizing,
            }
        elif strategy == "aggressive":
            return {
                "signal_confirmation_steps": self.aggressive_signal_confirmation_steps,
                "risk_management": self.aggressive_risk_management,
                "position_sizing": self.aggressive_position_sizing,
            }
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Available: conservative, aggressive"
            )

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

    def setup_logging(self) -> None:
        """
        Setup logging configuration based on environment variables.
        
        This method configures:
        - Root logger level
        - Console and file handlers
        - Module-specific log levels
        - Log formatting and rotation
        """
        import logging
        import logging.handlers
        import sys
        from pathlib import Path
        
        # Create logs directory if it doesn't exist
        logs_dir = Path(self.logs_dir)
        logs_dir.mkdir(exist_ok=True, parents=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level_global.upper(), logging.INFO))
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(self.log_format)
        
        # Console handler
        if self.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(getattr(logging, self.log_level_global.upper(), logging.INFO))
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.log_to_file:
            log_file_path = logs_dir / Path(self.log_file).name
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=self.log_file_max_size,
                backupCount=self.log_file_backup_count
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(getattr(logging, self.log_level_global.upper(), logging.INFO))
            root_logger.addHandler(file_handler)
        
        # Configure module-specific loggers
        self._configure_module_loggers()
        
    def _configure_module_loggers(self) -> None:
        """Configure log levels for specific modules."""
        import logging
        
        # Map of module names to their configured log levels
        module_configs = {
            "trading_bot.backtesting": self.log_level_backtesting,
            "trading_bot.backtesting.engine": self.log_level_backtesting,
            "trading_bot.agents": self.log_level_agents,
            "trading_bot.agents.signal_detection_agent": self.log_level_agents,
            "trading_bot.agents.decision_maker_agent": self.log_level_agents,
            "trading_bot.data": self.log_level_data,
            "trading_bot.data.data_provider": self.log_level_data,
            "trading_bot.visualization": self.log_level_visualization,
            "trading_bot.visualization.charts": self.log_level_visualization,
            "trading_bot.cli": self.log_level_cli,
        }
        
        for module_name, level_str in module_configs.items():
            logger = logging.getLogger(module_name)
            level = getattr(logging, level_str.upper(), logging.INFO)
            logger.setLevel(level)
            
    def get_logging_status(self) -> Dict[str, Any]:
        """Get current logging configuration status."""
        import logging
        
        return {
            "global_level": self.log_level_global,
            "console_enabled": self.log_to_console,
            "file_enabled": self.log_to_file,
            "log_file": self.log_file,
            "module_levels": {
                "backtesting": self.log_level_backtesting,
                "agents": self.log_level_agents,
                "data": self.log_level_data,
                "visualization": self.log_level_visualization,
                "cli": self.log_level_cli,
            },
            "root_logger_level": logging.getLevelName(logging.getLogger().getEffectiveLevel()),
            "handlers_count": len(logging.getLogger().handlers),
        }


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
