"""
Unit tests for the agent decision logging system.
"""

import json
import os

# Import our logging classes
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import mock_open, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.decision_logging.decision_logger import (
    DecisionEvent,
    DecisionLogger,
    DecisionType,
    LogAnalyzer,
    LogLevel,
)


class TestDecisionEvent(unittest.TestCase):
    """Test cases for DecisionEvent class."""

    def setUp(self):
        """Set up test data."""
        self.test_timestamp = datetime(2024, 1, 15, 10, 30, 45)
        self.test_event = DecisionEvent(
            agent_name="TestAgent",
            agent_type="SignalGenerator",
            timestamp=self.test_timestamp,
            decision_type=DecisionType.SIGNAL_GENERATION,
            action_taken="Generated BUY signal for EURUSD",
            confidence_score=0.85,
            reasoning_summary="Strong upward trend detected",
            full_reasoning="Detailed analysis of market conditions...",
            llm_prompt="Analyze the current market trend",
            llm_response="Based on indicators, trend is bullish",
            context_data={"symbol": "EURUSD", "timeframe": "1h"},
        )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = self.test_event.to_dict()

        self.assertEqual(result["agent_name"], "TestAgent")
        self.assertEqual(result["agent_type"], "SignalGenerator")
        self.assertEqual(result["timestamp"], "2024-01-15T10:30:45")
        self.assertEqual(result["decision_type"], "signal_generation")
        self.assertEqual(result["action_taken"], "Generated BUY signal for EURUSD")
        self.assertEqual(result["confidence_score"], 0.85)
        self.assertEqual(result["reasoning_summary"], "Strong upward trend detected")
        self.assertEqual(
            result["full_reasoning"], "Detailed analysis of market conditions..."
        )
        self.assertEqual(result["llm_prompt"], "Analyze the current market trend")
        self.assertEqual(
            result["llm_response"], "Based on indicators, trend is bullish"
        )
        self.assertEqual(
            result["context_data"], {"symbol": "EURUSD", "timeframe": "1h"}
        )

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = self.test_event.to_dict()
        reconstructed = DecisionEvent.from_dict(data)

        self.assertEqual(reconstructed.agent_name, self.test_event.agent_name)
        self.assertEqual(reconstructed.agent_type, self.test_event.agent_type)
        self.assertEqual(reconstructed.timestamp, self.test_event.timestamp)
        self.assertEqual(reconstructed.decision_type, self.test_event.decision_type)
        self.assertEqual(reconstructed.action_taken, self.test_event.action_taken)
        self.assertEqual(
            reconstructed.confidence_score, self.test_event.confidence_score
        )
        self.assertEqual(
            reconstructed.reasoning_summary, self.test_event.reasoning_summary
        )
        self.assertEqual(reconstructed.full_reasoning, self.test_event.full_reasoning)
        self.assertEqual(reconstructed.llm_prompt, self.test_event.llm_prompt)
        self.assertEqual(reconstructed.llm_response, self.test_event.llm_response)
        self.assertEqual(reconstructed.context_data, self.test_event.context_data)

    def test_round_trip_conversion(self):
        """Test that to_dict and from_dict are inverse operations."""
        original = self.test_event
        converted = DecisionEvent.from_dict(original.to_dict())

        # Should be equivalent (though not identical objects)
        self.assertEqual(original.agent_name, converted.agent_name)
        self.assertEqual(original.timestamp, converted.timestamp)
        self.assertEqual(original.decision_type, converted.decision_type)


class TestDecisionLogger(unittest.TestCase):
    """Test cases for DecisionLogger class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_decisions.log")
        self.logger = DecisionLogger(
            log_file_path=self.log_file, log_level=LogLevel.STANDARD
        )

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test logger initialization."""
        self.assertEqual(self.logger.log_file_path, Path(self.log_file))
        self.assertEqual(self.logger.log_level, LogLevel.STANDARD)
        self.assertTrue(self.logger.log_file_path.parent.exists())

    def test_log_decision_standard_mode(self):
        """Test logging a decision in standard mode."""
        event = self.logger.log_decision(
            agent_name="TestAgent",
            agent_type="SignalGenerator",
            decision_type=DecisionType.SIGNAL_GENERATION,
            action_taken="Generated BUY signal",
            confidence_score=0.85,
            reasoning_summary="Strong trend detected",
            full_reasoning="This should not be logged in standard mode",
            llm_prompt="This should not be logged in standard mode",
            llm_response="This should not be logged in standard mode",
        )

        # Check event properties
        self.assertEqual(event.agent_name, "TestAgent")
        self.assertEqual(event.agent_type, "SignalGenerator")
        self.assertEqual(event.decision_type, DecisionType.SIGNAL_GENERATION)
        self.assertEqual(event.action_taken, "Generated BUY signal")
        self.assertEqual(event.confidence_score, 0.85)
        self.assertEqual(event.reasoning_summary, "Strong trend detected")

        # These should be None in standard mode
        self.assertIsNone(event.full_reasoning)
        self.assertIsNone(event.llm_prompt)
        self.assertIsNone(event.llm_response)

        # Check that log file was created and contains the entry
        self.assertTrue(Path(self.log_file).exists())

        with open(self.log_file, "r") as f:
            content = f.read().strip()
            log_entry = json.loads(content)

            self.assertEqual(log_entry["log_level"], "standard")
            self.assertEqual(log_entry["event"]["agent_name"], "TestAgent")
            self.assertEqual(log_entry["event"]["decision_type"], "signal_generation")
            self.assertIsNone(log_entry["event"]["full_reasoning"])

    def test_log_decision_verbose_mode(self):
        """Test logging a decision in verbose mode."""
        # Set logger to verbose mode
        self.logger.set_log_level(LogLevel.VERBOSE)

        event = self.logger.log_decision(
            agent_name="TestAgent",
            agent_type="SignalGenerator",
            decision_type=DecisionType.SIGNAL_GENERATION,
            action_taken="Generated BUY signal",
            confidence_score=0.85,
            reasoning_summary="Strong trend detected",
            full_reasoning="Detailed analysis of market conditions",
            llm_prompt="Analyze the current market trend",
            llm_response="The trend is bullish",
        )

        # These should be present in verbose mode
        self.assertEqual(event.full_reasoning, "Detailed analysis of market conditions")
        self.assertEqual(event.llm_prompt, "Analyze the current market trend")
        self.assertEqual(event.llm_response, "The trend is bullish")

        # Check log file content
        with open(self.log_file, "r") as f:
            content = f.read().strip()
            log_entry = json.loads(content)

            self.assertEqual(log_entry["log_level"], "verbose")
            self.assertEqual(
                log_entry["event"]["full_reasoning"],
                "Detailed analysis of market conditions",
            )
            self.assertEqual(
                log_entry["event"]["llm_prompt"], "Analyze the current market trend"
            )

    def test_log_decision_with_context_data(self):
        """Test logging a decision with context data."""
        context = {"symbol": "EURUSD", "timeframe": "1h", "price": 1.2345}

        event = self.logger.log_decision(
            agent_name="TestAgent",
            agent_type="SignalGenerator",
            decision_type=DecisionType.SIGNAL_GENERATION,
            action_taken="Generated BUY signal",
            confidence_score=0.85,
            reasoning_summary="Strong trend detected",
            context_data=context,
        )

        self.assertEqual(event.context_data, context)

        # Check log file content
        with open(self.log_file, "r") as f:
            content = f.read().strip()
            log_entry = json.loads(content)

            self.assertEqual(log_entry["event"]["context_data"], context)

    def test_invalid_confidence_score(self):
        """Test that invalid confidence scores raise ValueError."""
        with self.assertRaises(ValueError):
            self.logger.log_decision(
                agent_name="TestAgent",
                agent_type="SignalGenerator",
                decision_type=DecisionType.SIGNAL_GENERATION,
                action_taken="Generated BUY signal",
                confidence_score=1.5,  # Invalid: > 1.0
                reasoning_summary="Strong trend detected",
            )

        with self.assertRaises(ValueError):
            self.logger.log_decision(
                agent_name="TestAgent",
                agent_type="SignalGenerator",
                decision_type=DecisionType.SIGNAL_GENERATION,
                action_taken="Generated BUY signal",
                confidence_score=-0.1,  # Invalid: < 0.0
                reasoning_summary="Strong trend detected",
            )

    def test_log_level_override(self):
        """Test overriding log level for individual entries."""
        # Logger is in standard mode
        self.assertEqual(self.logger.log_level, LogLevel.STANDARD)

        # Log with verbose override
        event = self.logger.log_decision(
            agent_name="TestAgent",
            agent_type="SignalGenerator",
            decision_type=DecisionType.SIGNAL_GENERATION,
            action_taken="Generated BUY signal",
            confidence_score=0.85,
            reasoning_summary="Strong trend detected",
            full_reasoning="This should be logged due to override",
            log_level=LogLevel.VERBOSE,
        )

        # Should include verbose data despite logger being in standard mode
        self.assertEqual(event.full_reasoning, "This should be logged due to override")

        # Check log file content
        with open(self.log_file, "r") as f:
            content = f.read().strip()
            log_entry = json.loads(content)

            self.assertEqual(log_entry["log_level"], "verbose")
            self.assertEqual(
                log_entry["event"]["full_reasoning"],
                "This should be logged due to override",
            )

    def test_multiple_decisions(self):
        """Test logging multiple decisions."""
        # Log three decisions
        for i in range(3):
            self.logger.log_decision(
                agent_name=f"Agent{i}",
                agent_type="TestType",
                decision_type=DecisionType.SIGNAL_GENERATION,
                action_taken=f"Action {i}",
                confidence_score=0.5 + i * 0.2,
                reasoning_summary=f"Reasoning {i}",
            )

        # Check that all three are in the log file
        with open(self.log_file, "r") as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 3)

        for i, line in enumerate(lines):
            log_entry = json.loads(line.strip())
            self.assertEqual(log_entry["event"]["agent_name"], f"Agent{i}")
            self.assertEqual(log_entry["event"]["action_taken"], f"Action {i}")


class TestLogAnalyzer(unittest.TestCase):
    """Test cases for LogAnalyzer class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_decisions.log")

        # Create test data
        self.test_decisions = [
            DecisionEvent(
                agent_name="Agent1",
                agent_type="SignalGenerator",
                timestamp=datetime(2024, 1, 15, 10, 0, 0),
                decision_type=DecisionType.SIGNAL_GENERATION,
                action_taken="BUY signal",
                confidence_score=0.8,
                reasoning_summary="Strong upward trend",
            ),
            DecisionEvent(
                agent_name="Agent2",
                agent_type="RiskManager",
                timestamp=datetime(2024, 1, 15, 11, 0, 0),
                decision_type=DecisionType.RISK_ASSESSMENT,
                action_taken="Risk level: LOW",
                confidence_score=0.9,
                reasoning_summary="Low volatility detected",
            ),
            DecisionEvent(
                agent_name="Agent1",
                agent_type="SignalGenerator",
                timestamp=datetime(2024, 1, 15, 12, 0, 0),
                decision_type=DecisionType.SIGNAL_GENERATION,
                action_taken="SELL signal",
                confidence_score=0.7,
                reasoning_summary="Reversal pattern detected",
            ),
        ]

        # Write test data to log file
        with open(self.log_file, "w") as f:
            for decision in self.test_decisions:
                log_entry = {"log_level": "standard", "event": decision.to_dict()}
                f.write(json.dumps(log_entry) + "\n")

        self.analyzer = LogAnalyzer(self.log_file)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_all_decisions(self):
        """Test reading all decisions from log file."""
        decisions = self.analyzer.read_all_decisions()

        self.assertEqual(len(decisions), 3)

        # Check first decision
        self.assertEqual(decisions[0].agent_name, "Agent1")
        self.assertEqual(decisions[0].decision_type, DecisionType.SIGNAL_GENERATION)
        self.assertEqual(decisions[0].action_taken, "BUY signal")

        # Check timestamp parsing
        self.assertEqual(decisions[0].timestamp, datetime(2024, 1, 15, 10, 0, 0))

    def test_read_empty_file(self):
        """Test reading from non-existent file."""
        empty_analyzer = LogAnalyzer("nonexistent_file.log")
        decisions = empty_analyzer.read_all_decisions()
        self.assertEqual(len(decisions), 0)

    def test_filter_by_agent_name(self):
        """Test filtering decisions by agent name."""
        decisions = self.analyzer.read_all_decisions()
        filtered = self.analyzer.filter_decisions(decisions, agent_name="Agent1")

        self.assertEqual(len(filtered), 2)
        for decision in filtered:
            self.assertEqual(decision.agent_name, "Agent1")

    def test_filter_by_decision_type(self):
        """Test filtering decisions by decision type."""
        decisions = self.analyzer.read_all_decisions()
        filtered = self.analyzer.filter_decisions(
            decisions, decision_type=DecisionType.SIGNAL_GENERATION
        )

        self.assertEqual(len(filtered), 2)
        for decision in filtered:
            self.assertEqual(decision.decision_type, DecisionType.SIGNAL_GENERATION)

    def test_filter_by_time_range(self):
        """Test filtering decisions by time range."""
        decisions = self.analyzer.read_all_decisions()

        # Filter for decisions after 10:30
        start_time = datetime(2024, 1, 15, 10, 30, 0)
        filtered = self.analyzer.filter_decisions(decisions, start_time=start_time)

        self.assertEqual(len(filtered), 2)
        for decision in filtered:
            self.assertGreaterEqual(decision.timestamp, start_time)

    def test_filter_by_confidence_range(self):
        """Test filtering decisions by confidence range."""
        decisions = self.analyzer.read_all_decisions()

        # Filter for high confidence decisions (>= 0.8)
        filtered = self.analyzer.filter_decisions(decisions, min_confidence=0.8)

        self.assertEqual(len(filtered), 2)
        for decision in filtered:
            self.assertGreaterEqual(decision.confidence_score, 0.8)

    def test_get_decision_summary(self):
        """Test getting decision summary statistics."""
        decisions = self.analyzer.read_all_decisions()
        summary = self.analyzer.get_decision_summary(decisions)

        self.assertEqual(summary["total_decisions"], 3)

        # Check agent statistics
        expected_agents = {"Agent1 (SignalGenerator)": 2, "Agent2 (RiskManager)": 1}
        self.assertEqual(summary["agents"], expected_agents)

        # Check decision type statistics
        expected_types = {"signal_generation": 2, "risk_assessment": 1}
        self.assertEqual(summary["decision_types"], expected_types)

        # Check confidence statistics
        confidence_stats = summary["confidence_stats"]
        self.assertEqual(confidence_stats["min"], 0.7)
        self.assertEqual(confidence_stats["max"], 0.9)
        self.assertAlmostEqual(confidence_stats["mean"], 0.8, places=2)

    def test_export_to_json(self):
        """Test exporting decisions to JSON file."""
        output_file = os.path.join(self.temp_dir, "export.json")
        decisions = self.analyzer.read_all_decisions()

        self.analyzer.export_to_json(output_file, decisions, include_summary=True)

        # Read and verify exported data
        with open(output_file, "r") as f:
            exported_data = json.load(f)

        self.assertIn("decisions", exported_data)
        self.assertIn("summary", exported_data)
        self.assertEqual(len(exported_data["decisions"]), 3)
        self.assertEqual(exported_data["summary"]["total_decisions"], 3)

    def test_malformed_log_entries(self):
        """Test handling of malformed log entries."""
        # Create log file with some malformed entries
        malformed_log = os.path.join(self.temp_dir, "malformed.log")
        with open(malformed_log, "w") as f:
            # Valid entry
            valid_entry = {
                "log_level": "standard",
                "event": self.test_decisions[0].to_dict(),
            }
            f.write(json.dumps(valid_entry) + "\n")

            # Malformed JSON
            f.write('{"incomplete": json\n')

            # Missing event key
            f.write('{"log_level": "standard"}\n')

            # Another valid entry
            valid_entry2 = {
                "log_level": "standard",
                "event": self.test_decisions[1].to_dict(),
            }
            f.write(json.dumps(valid_entry2) + "\n")

        analyzer = LogAnalyzer(malformed_log)
        decisions = analyzer.read_all_decisions()

        # Should only read the 2 valid entries, skipping malformed ones
        self.assertEqual(len(decisions), 2)


if __name__ == "__main__":
    unittest.main()
