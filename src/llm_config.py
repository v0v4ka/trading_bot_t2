"""
LLM Model Configuration Utility for Trading Bot.

This module provides utilities to configure LLM models based on environment
variables, configuration files, and runtime parameters.
"""

import os
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Handle missing OpenAI dependency gracefully

from .cli.config import load_config


class LLMConfigurator:
    """Handles LLM model configuration for the trading bot."""

    # Valid OpenAI models
    VALID_MODELS = [
        "gpt-4o",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "gpt-4o-mini",
        "gpt-4.1-mini",  # Custom model name
    ]

    DEFAULT_MODEL = "gpt-4.1-mini"

    @staticmethod
    def get_model_from_env() -> str:
        """Get LLM model from environment variables."""
        model = os.getenv("OPENAI_MODEL", LLMConfigurator.DEFAULT_MODEL)
        if model not in LLMConfigurator.VALID_MODELS:
            print(
                f"Warning: Unknown model '{model}', using default '{LLMConfigurator.DEFAULT_MODEL}'"
            )
            return LLMConfigurator.DEFAULT_MODEL
        return model

    @staticmethod
    def get_model_from_config(config_path: Optional[str] = None) -> str:
        """Get LLM model from configuration file."""
        try:
            config = load_config(config_path)
            model = config.get("agents", {}).get(
                "llm_model", LLMConfigurator.DEFAULT_MODEL
            )
            if model not in LLMConfigurator.VALID_MODELS:
                print(f"Warning: Unknown model '{model}' in config, using default")
                return LLMConfigurator.DEFAULT_MODEL
            return model
        except Exception as e:
            print(f"Error loading config: {e}, using default model")
            return LLMConfigurator.DEFAULT_MODEL

    @staticmethod
    def is_test_mode() -> bool:
        """Check if test mode is enabled (disables LLM calls)."""
        return os.getenv("TBOT_TEST_MODE", "0") == "1"

    @staticmethod
    def create_openai_client() -> Optional[Any]:
        """Create OpenAI client with proper configuration."""
        if LLMConfigurator.is_test_mode():
            return None  # No client in test mode

        if OpenAI is None:
            print("Warning: OpenAI package not available, LLM features disabled")
            return None

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: No OPENAI_API_KEY found, LLM features disabled")
            return None

        return OpenAI(api_key=api_key) @ staticmethod

    def get_effective_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """Get effective LLM configuration from all sources."""
        import warnings

        # Priority: Environment > Config File > Defaults
        env_model = os.getenv("OPENAI_MODEL")
        config_model = None
        config_used = False

        if config_path:
            config_model = LLMConfigurator.get_model_from_config(config_path)
            config_used = True

        # Deprecation warning if YAML config is used when .env is available
        if config_used and env_model:
            warnings.warn(
                "Using both YAML config and .env settings. Environment variables take precedence. "
                "Consider migrating all settings to .env for consistency.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Get all configuration values with .env precedence
        effective_model = env_model or config_model or LLMConfigurator.DEFAULT_MODEL

        # Use environment variables first, fallback to config file
        signal_threshold = float(os.getenv("SIGNAL_CONFIDENCE_THRESHOLD", "0.6"))
        decision_threshold = float(os.getenv("DECISION_CONFIDENCE_THRESHOLD", "0.7"))
        use_llm_confirmation = (
            os.getenv("USE_LLM_CONFIRMATION", "true").lower() == "true"
        )

        # If config file is used, warn about deprecated overlapping settings
        if config_used:
            try:
                config = load_config(config_path)
                agents_config = config.get("agents", {})

                if "signal_confidence_threshold" in agents_config and os.getenv(
                    "SIGNAL_CONFIDENCE_THRESHOLD"
                ):
                    warnings.warn(
                        "signal_confidence_threshold found in both YAML and .env. Using .env value. "
                        "Consider removing from YAML config.",
                        DeprecationWarning,
                        stacklevel=2,
                    )

                if "decision_confidence_threshold" in agents_config and os.getenv(
                    "DECISION_CONFIDENCE_THRESHOLD"
                ):
                    warnings.warn(
                        "decision_confidence_threshold found in both YAML and .env. Using .env value. "
                        "Consider removing from YAML config.",
                        DeprecationWarning,
                        stacklevel=2,
                    )

                if "use_llm_confirmation" in agents_config and os.getenv(
                    "USE_LLM_CONFIRMATION"
                ):
                    warnings.warn(
                        "use_llm_confirmation found in both YAML and .env. Using .env value. "
                        "Consider removing from YAML config.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
            except Exception:
                pass  # If config loading fails, just use env values

        return {
            "model": effective_model,
            "test_mode": LLMConfigurator.is_test_mode(),
            "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "client": LLMConfigurator.create_openai_client(),
            "signal_confidence_threshold": signal_threshold,
            "decision_confidence_threshold": decision_threshold,
            "use_llm_confirmation": use_llm_confirmation,
            "source": "env" if env_model else ("config" if config_model else "default"),
        }


def configure_agent_llm(
    agent_class, config_path: Optional[str] = None, model_override: Optional[str] = None
):
    """
    Factory function to create an agent with proper LLM configuration.

    Args:
        agent_class: The agent class to instantiate
        config_path: Optional path to configuration file
        model_override: Optional model override

    Returns:
        Configured agent instance
    """
    llm_config = LLMConfigurator.get_effective_config(config_path)

    # Apply model override if provided
    if model_override and model_override in LLMConfigurator.VALID_MODELS:
        llm_config["model"] = model_override

    # Create agent with appropriate parameters
    if llm_config["test_mode"] or not llm_config["client"]:
        # Test mode or no API key - create agent without LLM
        return agent_class()
    else:
        # Production mode with LLM
        if hasattr(agent_class, "__init__"):
            # Check if agent accepts model and client parameters
            import inspect

            sig = inspect.signature(agent_class.__init__)
            kwargs = {}

            if "model" in sig.parameters:
                kwargs["model"] = llm_config["model"]
            if "client" in sig.parameters:
                kwargs["client"] = llm_config["client"]

            return agent_class(**kwargs)
        else:
            return agent_class()


# Usage examples:
def example_usage():
    """Examples of how to use LLM configuration."""
    import warnings

    # RECOMMENDED Method 1: Environment variables (.env file)
    os.environ["OPENAI_MODEL"] = "gpt-4.1-mini"
    os.environ["OPENAI_API_KEY"] = "your-key-here"
    os.environ["SIGNAL_CONFIDENCE_THRESHOLD"] = "0.6"
    os.environ["DECISION_CONFIDENCE_THRESHOLD"] = "0.7"
    os.environ["USE_LLM_CONFIRMATION"] = "true"

    # Get configuration - .env values take precedence
    config = LLMConfigurator.get_effective_config()
    print(f"Using model: {config['model']} (source: {config['source']})")

    # Method 2: Configuration file (DEPRECATED for overlapping settings)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = LLMConfigurator.get_effective_config("config/llm_config.yml")
        if w:
            print(f"Deprecation warnings: {len(w)} warnings about config overlap")

    # Method 3: Direct agent configuration (RECOMMENDED: .env only)
    from .agents.decision_maker_agent import DecisionMakerAgent

    # Configure with .env settings (recommended)
    agent = configure_agent_llm(DecisionMakerAgent)

    # Configure with model override
    agent_override = configure_agent_llm(
        DecisionMakerAgent,
        model_override="gpt-4.1-mini",
    )

    # Method 4: Test mode
    os.environ["TBOT_TEST_MODE"] = "1"
    test_agent = configure_agent_llm(DecisionMakerAgent)
