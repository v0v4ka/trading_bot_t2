"""
LLM Model Configuration Utility for Trading Bot.

This module provides utilities to configure LLM models based on environment
variables only. YAML configuration has been deprecated in favor of .env approach.
"""

import os
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Handle missing OpenAI dependency gracefully


class LLMConfigurator:
    """Handles LLM model configuration for the trading bot using environment variables only."""

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
            print(f"Warning: Unknown model '{model}', using default '{LLMConfigurator.DEFAULT_MODEL}'")
            return LLMConfigurator.DEFAULT_MODEL
        return model

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

        return OpenAI(api_key=api_key)

    @staticmethod
    def get_effective_config() -> Dict[str, Any]:
        """Get effective LLM configuration from environment variables only."""
        
        # Get all configuration values from environment
        model = LLMConfigurator.get_model_from_env()
        signal_threshold = float(os.getenv("SIGNAL_CONFIDENCE_THRESHOLD", "0.6"))
        decision_threshold = float(os.getenv("DECISION_CONFIDENCE_THRESHOLD", "0.7"))
        use_llm_confirmation = os.getenv("USE_LLM_CONFIRMATION", "true").lower() == "true"
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "500"))
        
        return {
            "model": model,
            "test_mode": LLMConfigurator.is_test_mode(),
            "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "client": LLMConfigurator.create_openai_client(),
            "signal_confidence_threshold": signal_threshold,
            "decision_confidence_threshold": decision_threshold,
            "use_llm_confirmation": use_llm_confirmation,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "source": "env"
        }


def configure_agent_llm(agent_class, model_override: Optional[str] = None):
    """
    Configure an agent class with LLM settings from environment variables.
    
    Args:
        agent_class: The agent class to configure
        model_override: Optional model name to override environment setting
        
    Returns:
        Configured agent instance
    """
    if LLMConfigurator.is_test_mode():
        # In test mode, create agent without LLM dependencies
        return agent_class()
    else:
        import inspect
        
        llm_config = LLMConfigurator.get_effective_config()
        
        if model_override:
            llm_config["model"] = model_override
        
        # Check what parameters the agent accepts
        sig = inspect.signature(agent_class.__init__)
        kwargs = {}
        
        if "model" in sig.parameters:
            kwargs["model"] = llm_config["model"]
        if "client" in sig.parameters:
            kwargs["client"] = llm_config["client"]
        if "temperature" in sig.parameters:
            kwargs["temperature"] = llm_config["temperature"]
        if "max_tokens" in sig.parameters:
            kwargs["max_tokens"] = llm_config["max_tokens"]
            
        return agent_class(**kwargs)


# Usage examples:
def example_usage():
    """Examples of how to use LLM configuration."""

    # RECOMMENDED: Environment variables (.env file)
    os.environ["OPENAI_MODEL"] = "gpt-4.1-mini"
    os.environ["OPENAI_API_KEY"] = "your-key-here"
    os.environ["SIGNAL_CONFIDENCE_THRESHOLD"] = "0.6"
    os.environ["DECISION_CONFIDENCE_THRESHOLD"] = "0.7"
    os.environ["USE_LLM_CONFIRMATION"] = "true"

    # Get configuration - all from .env
    config = LLMConfigurator.get_effective_config()
    print(f"Using model: {config['model']} (source: {config['source']})")

    # Direct agent configuration (recommended)
    from .agents.decision_maker_agent import DecisionMakerAgent

    # Configure with .env settings
    agent = configure_agent_llm(DecisionMakerAgent)
    
    # Configure with model override
    agent_override = configure_agent_llm(
        DecisionMakerAgent,
        model_override="gpt-4.1-mini",
    )

    # Test mode
    os.environ["TBOT_TEST_MODE"] = "1"
    test_agent = configure_agent_llm(DecisionMakerAgent)
