#!/usr/bin/env python3
"""
Integration example demonstrating Decision Maker Agent with Signal Detection Agent.
This script shows how the complete signal-to-decision workflow operates.
"""

import logging
from datetime import datetime
from pathlib import Path

from src.agents.decision_maker_agent import DecisionMakerAgent
from src.agents.signal_detection_agent import SignalDetectionAgent
from src.data.data_provider import DataProvider
from src.logging.decision_logger import DecisionLogger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate the complete signal detection to decision workflow."""
    try:
        # Initialize components
        data_provider = DataProvider()
        decision_logger = DecisionLogger(
            log_file=Path("logs/decision_integration_demo.log")
        )
        
        # Initialize agents
        signal_agent = SignalDetectionAgent()
        decision_agent = DecisionMakerAgent(decision_logger=decision_logger)
        
        # Fetch recent market data for AAPL
        logger.info("Fetching market data for AAPL...")
        market_data = data_provider.fetch_ohlcv("AAPL", period="5d", interval="1h")
        
        if market_data.empty:
            logger.error("No market data received")
            return
        
        # Get the latest data point
        latest_data = market_data.iloc[-1]
        logger.info(f"Latest AAPL data: Close=${latest_data.close:.2f}, Volume={latest_data.volume:,}")
        
        # Generate signals using the signal detection agent
        logger.info("Generating trading signals...")
        signals = signal_agent.detect_signals(market_data)
        
        logger.info(f"Generated {len(signals)} signals:")
        for signal in signals:
            logger.info(f"  - {signal.type} signal: {signal.confidence:.2f} confidence")
        
        # Make trading decision using the decision maker agent
        logger.info("Making trading decision...")
        decision = decision_agent.make_decision(signals, latest_data)
        
        # Display results
        logger.info("=== TRADING DECISION RESULTS ===")
        logger.info(f"Action: {decision.action}")
        logger.info(f"Confidence: {decision.confidence:.2f}")
        logger.info(f"Entry Price: ${decision.entry_price:.2f}" if decision.entry_price else "Entry Price: N/A")
        logger.info(f"Stop Loss: ${decision.stop_loss:.2f}" if decision.stop_loss else "Stop Loss: N/A")
        logger.info(f"Signals Used: {len(decision.signals_used)}")
        logger.info(f"Reasoning: {decision.reasoning}")
        
        # Example of how this would integrate into a larger trading system
        if decision.action in ["BUY", "SELL"]:
            logger.info(f"🔥 TRADING SIGNAL: {decision.action} AAPL at ${decision.entry_price:.2f}")
            logger.info(f"   Risk Management: Stop Loss at ${decision.stop_loss:.2f}")
            logger.info(f"   Confidence Level: {decision.confidence:.1%}")
        else:
            logger.info("💤 No trading action recommended - holding position")
        
        logger.info("Decision logged to: logs/decision_integration_demo.log")
        
    except Exception as e:
        logger.error(f"Integration demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
