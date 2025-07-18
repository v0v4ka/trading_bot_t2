#!/usr/bin/env python3
"""
Demo script for the Agent Decision Logging System.

This script demonstrates the key features of the decision logging system:
- Standard and verbose logging modes
- Different types of agent decisions
- Log analysis and filtering capabilities
- Export functionality
"""

import os
import sys
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.logging.decision_logger import (
    DecisionLogger,
    DecisionType,
    LogAnalyzer,
    LogLevel,
)


def demo_basic_logging():
    """Demonstrate basic logging functionality."""
    print("=== Demo: Basic Decision Logging ===")

    # Create a logger
    logger = DecisionLogger(
        log_file_path="logs/demo_decisions.log", log_level=LogLevel.STANDARD
    )

    print(f"Created logger with log file: {logger.get_log_file_path()}")

    # Log some decisions
    print("\nLogging decisions...")

    # Signal generation decision
    event1 = logger.log_decision(
        agent_name="SignalAgent_EURUSD",
        agent_type="TechnicalAnalysisAgent",
        decision_type=DecisionType.SIGNAL_GENERATION,
        action_taken="Generated BUY signal for EURUSD",
        confidence_score=0.85,
        reasoning_summary="Strong bullish signals: MA crossover, RSI oversold recovery, volume increase",
        context_data={
            "symbol": "EURUSD",
            "timeframe": "1h",
            "price": 1.0895,
            "indicators": {"ma_5": 1.0890, "ma_20": 1.0875, "rsi": 45.2},
        },
    )
    print(f"Logged: {event1.action_taken} (confidence: {event1.confidence_score})")

    # Risk assessment decision
    event2 = logger.log_decision(
        agent_name="RiskManager_Main",
        agent_type="RiskManagementAgent",
        decision_type=DecisionType.RISK_ASSESSMENT,
        action_taken="Approved trade with 2% position size",
        confidence_score=0.75,
        reasoning_summary="Risk within acceptable parameters, good R:R ratio",
        context_data={
            "position_size": 0.02,
            "stop_loss": 1.0850,
            "take_profit": 1.0950,
            "risk_reward_ratio": 2.2,
        },
    )
    print(f"Logged: {event2.action_taken} (confidence: {event2.confidence_score})")

    # Trade execution decision
    event3 = logger.log_decision(
        agent_name="ExecutionAgent_MT5",
        agent_type="TradeExecutionAgent",
        decision_type=DecisionType.TRADE_EXECUTION,
        action_taken="Executed BUY order: 0.02 lots EURUSD at 1.0895",
        confidence_score=0.95,
        reasoning_summary="Order filled at market price, slippage minimal",
        context_data={
            "order_id": "ORD_123456",
            "executed_price": 1.0895,
            "slippage": 0.0001,
            "execution_time_ms": 45,
        },
    )
    print(f"Logged: {event3.action_taken} (confidence: {event3.confidence_score})")


def demo_verbose_logging():
    """Demonstrate verbose logging with LLM interactions."""
    print("\n=== Demo: Verbose Logging with LLM Data ===")

    # Create a logger in verbose mode
    logger = DecisionLogger(
        log_file_path="logs/demo_decisions.log", log_level=LogLevel.VERBOSE
    )

    # Log a decision with full LLM prompt/response
    event = logger.log_decision(
        agent_name="AnalysisAgent_GPT4",
        agent_type="LLMAnalysisAgent",
        decision_type=DecisionType.DATA_ANALYSIS,
        action_taken="Market sentiment analysis: BULLISH",
        confidence_score=0.78,
        reasoning_summary="Positive news sentiment outweighs technical concerns",
        full_reasoning="""
        Comprehensive analysis performed on multiple data sources:
        1. Technical indicators show mixed signals (MA crossover bullish, RSI neutral)
        2. Fundamental analysis shows EUR strength due to ECB hawkish stance
        3. Sentiment analysis of news articles shows 70% positive sentiment
        4. Order book analysis shows institutional buying pressure
        5. Risk factors: US inflation data pending, geopolitical tensions
        
        Conclusion: Despite technical uncertainty, fundamental and sentiment 
        factors support bullish bias for short-term timeframe.
        """,
        llm_prompt="""
        You are a professional forex analyst. Analyze the current EURUSD market situation
        and provide a trading recommendation. Consider the following data:
        
        Technical indicators:
        - 5-period MA: 1.0890 (above price)
        - 20-period MA: 1.0875 (below price) 
        - RSI: 52.3 (neutral)
        - MACD: Bullish crossover just occurred
        
        Fundamental factors:
        - ECB meeting minutes showed hawkish tone
        - US unemployment data mixed
        - EU inflation trending down but still elevated
        
        News sentiment (last 24h):
        - 15 positive EUR articles
        - 8 negative EUR articles  
        - 12 neutral articles
        
        Provide your analysis and recommendation with confidence level.
        """,
        llm_response="""
        Based on the comprehensive analysis of technical, fundamental, and sentiment data:
        
        **Technical Analysis**: Mixed signals with recent bullish MA crossover being positive,
        but RSI in neutral territory suggests no strong momentum either direction.
        
        **Fundamental Analysis**: ECB hawkish stance is EUR supportive, while US mixed data
        creates uncertainty. Net positive for EUR.
        
        **Sentiment Analysis**: Clear positive bias in news flow (15 vs 8 positive to negative).
        
        **Recommendation**: BULLISH bias for EURUSD
        **Target**: 1.0950-1.0980 range
        **Stop**: Below 1.0850
        **Confidence**: 78% - Strong fundamental and sentiment support offset technical uncertainty
        **Timeframe**: 1-3 days
        """,
        context_data={
            "analysis_timestamp": datetime.now().isoformat(),
            "data_sources": ["TradingView", "ForexFactory", "Reuters", "Bloomberg"],
            "llm_model": "gpt-4-turbo",
            "processing_time_s": 12.4,
        },
    )

    print(f"Logged verbose decision: {event.action_taken}")
    print(f"Full reasoning length: {len(event.full_reasoning)} characters")
    print(f"LLM prompt length: {len(event.llm_prompt)} characters")
    print(f"LLM response length: {len(event.llm_response)} characters")


def demo_log_analysis():
    """Demonstrate log analysis capabilities."""
    print("\n=== Demo: Log Analysis ===")

    # Create analyzer
    analyzer = LogAnalyzer("logs/demo_decisions.log")

    # Read all decisions
    decisions = analyzer.read_all_decisions()
    print(f"Total decisions in log: {len(decisions)}")

    # Get summary statistics
    summary = analyzer.get_decision_summary(decisions)
    print("\nSummary Statistics:")
    print(f"  Total decisions: {summary['total_decisions']}")
    print(
        f"  Time range: {summary['time_range']['start']} to {summary['time_range']['end']}"
    )
    print(f"  Agents: {summary['agents']}")
    print(f"  Decision types: {summary['decision_types']}")
    print(
        f"  Confidence stats: mean={summary['confidence_stats']['mean']:.2f}, "
        f"min={summary['confidence_stats']['min']}, "
        f"max={summary['confidence_stats']['max']}"
    )

    # Filter examples
    print("\nFiltering Examples:")

    # Filter by agent type
    signal_decisions = analyzer.filter_decisions(
        decisions, agent_type="TechnicalAnalysisAgent"
    )
    print(f"  Technical analysis decisions: {len(signal_decisions)}")

    # Filter by decision type
    trade_decisions = analyzer.filter_decisions(
        decisions, decision_type=DecisionType.TRADE_EXECUTION
    )
    print(f"  Trade execution decisions: {len(trade_decisions)}")

    # Filter by confidence
    high_confidence = analyzer.filter_decisions(decisions, min_confidence=0.8)
    print(f"  High confidence decisions (>=0.8): {len(high_confidence)}")

    # Filter by time (last 5 minutes)
    recent_time = datetime.now() - timedelta(minutes=5)
    recent_decisions = analyzer.filter_decisions(decisions, start_time=recent_time)
    print(f"  Recent decisions (last 5 min): {len(recent_decisions)}")


def demo_export():
    """Demonstrate export functionality."""
    print("\n=== Demo: Export to JSON ===")

    analyzer = LogAnalyzer("logs/demo_decisions.log")

    # Export all decisions with summary
    output_file = "logs/demo_export.json"
    analyzer.export_to_json(output_file, include_summary=True)

    print(f"Exported decisions to: {output_file}")

    # Show file size
    import os

    if os.path.exists(output_file):
        size = os.path.getsize(output_file)
        print(f"Export file size: {size} bytes")


def show_example_log_entry():
    """Show what a log entry looks like in the file."""
    print("\n=== Demo: Raw Log Entry Format ===")

    log_file = "logs/demo_decisions.log"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            first_line = f.readline().strip()
            if first_line:
                import json

                entry = json.loads(first_line)
                print("Example log entry (formatted):")
                print(json.dumps(entry, indent=2))


def main():
    """Run all demonstrations."""
    print("Agent Decision Logging System Demo")
    print("=" * 50)

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Run demonstrations
    demo_basic_logging()
    demo_verbose_logging()
    demo_log_analysis()
    demo_export()
    show_example_log_entry()

    print("\n" + "=" * 50)
    print("Demo completed! Check the 'logs/' directory for generated files.")


if __name__ == "__main__":
    main()
