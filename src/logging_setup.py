"""
Logging setup for Trading Bot.

This module provides environment-aware logging configuration.
Use env_config.setup_logging() for comprehensive logging setup.
"""

import logging
import sys
from .env_config import config

def setup_logging_from_env():
    """Setup logging using environment configuration."""
    config.setup_logging()
    return logging.getLogger("trading_bot")

# For backward compatibility - simple setup
def setup_basic_logging(level: str = "INFO"):
    """Basic logging setup - deprecated, use setup_logging_from_env() instead."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("trading_bot.log", mode="a"),
        ],
    )
    return logging.getLogger("trading_bot")

# Initialize with environment-aware configuration
try:
    logger = setup_logging_from_env()
    logger.info("Environment-aware logging framework initialized.")
except Exception as e:
    # Fallback to basic logging if env config fails
    logger = setup_basic_logging()
    logger.warning(f"Failed to setup env logging, using basic: {e}")
    logger.info("Basic logging framework initialized.")
