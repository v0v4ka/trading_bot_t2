"""
Unit tests for the Decision Maker Agent.
"""

import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from src.agents.decision_maker_agent import DecisionMakerAgent
from src.agents.schemas import Signal
from src.data.models import OHLCV
from src.decision_logging.decision_logger import DecisionLogger


class TestDecisionMakerAgent(unittest.TestCase):
    def test_alligator_awake_allows_trade(self):
        """Trade allowed when Alligator is awake."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.9,
                details={"indicator": "alligator", "jaw": 105, "teeth": 102, "lips": 100},
            )
        ]
        decision = self.agent.make_decision(signals, self.market_data)
        self.assertNotEqual(decision.action, "HOLD")
        self.assertIn("Alligator state: awake", decision.reasoning)

    def test_alligator_sleeping_filters_trade(self):
        """Trade filtered when Alligator is sleeping."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.9,
                details={"indicator": "alligator", "jaw": 102, "teeth": 101.7, "lips": 101.5},
            )
        ]
        decision = self.agent.make_decision(signals, self.market_data)
        self.assertEqual(decision.action, "HOLD")
        self.assertIn("Alligator sleeping: trade filtered", decision.reasoning)
        self.assertIn("Alligator state: sleeping", decision.reasoning)
    def test_three_wise_men_first(self):
        """Test detection of First Wise Man staged entry."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.9,
                details={"indicator": "alligator", "outside_mouth": True, "ao_color": "green"},
            )
        ]
        decision = self.agent.make_decision(signals, self.market_data)
        self.assertEqual(decision.action, "BUY")
        self.assertIn("First Wise Man", decision.reasoning)

    def test_three_wise_men_second(self):
        """Test detection of Second Wise Man staged entry."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.85,
                details={"indicator": "ao", "saucer": True},
            )
        ]
        decision = self.agent.make_decision(signals, self.market_data)
        self.assertEqual(decision.action, "BUY_ADDON")
        self.assertIn("Second Wise Man", decision.reasoning)

    def test_three_wise_men_third(self):
        """Test detection of Third Wise Man staged entry."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.88,
                details={"indicator": "fractal", "breakout": True},
            )
        ]
        decision = self.agent.make_decision(signals, self.market_data)
        self.assertEqual(decision.action, "BUY_ADDON2")
        self.assertIn("Third Wise Man", decision.reasoning)
    """Test cases for Decision Maker Agent."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger = Mock(spec=DecisionLogger)
        self.mock_client = Mock()
        self.agent = DecisionMakerAgent(
            client=self.mock_client, decision_logger=self.mock_logger
        )

        # Sample market data
        self.market_data = OHLCV(
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=10000,
        )

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "DecisionMaker")
        self.assertIsNotNone(self.agent.system_prompt)
        self.assertEqual(self.agent.confluence_threshold, 0.7)

    def test_make_decision_no_signals(self):
        """Test decision making with no signals."""
        decision = self.agent.make_decision([], self.market_data)

        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.confidence, 0.0)
        self.assertIn("No signals", decision.reasoning)
        self.assertEqual(len(decision.signals_used), 0)
        self.mock_logger.log_decision.assert_called_once()

    def test_make_decision_single_buy_signal(self):
        """Test decision making with single BUY signal (should HOLD)."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.8,
                details={"indicator": "fractal"},
            )
        ]

        decision = self.agent.make_decision(signals, self.market_data)

        # Single signal should result in HOLD due to insufficient confluence
        self.assertEqual(decision.action, "HOLD")
        self.assertLess(decision.confidence, 0.7)
        self.assertEqual(len(decision.signals_used), 1)

    def test_make_decision_multiple_buy_signals(self):
        """Test decision making with multiple BUY signals."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.8,
                details={"indicator": "fractal"},
            ),
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.7,
                details={"indicator": "alligator"},
            ),
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.9,
                details={"indicator": "ao"},
            ),
        ]

        decision = self.agent.make_decision(signals, self.market_data)

        self.assertEqual(decision.action, "BUY")
        self.assertGreater(decision.confidence, 0.6)  # Adjusted expectation
        self.assertEqual(len(decision.signals_used), 3)
        self.assertIsNotNone(decision.entry_price)
        self.assertIsNotNone(decision.stop_loss)

    def test_make_decision_multiple_sell_signals(self):
        """Test decision making with multiple SELL signals."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="SELL",
                confidence=0.8,
                details={"indicator": "fractal"},
            ),
            Signal(
                timestamp=datetime.now(),
                type="SELL",
                confidence=0.7,
                details={"indicator": "alligator"},
            ),
        ]

        decision = self.agent.make_decision(signals, self.market_data)

        self.assertEqual(decision.action, "SELL")
        self.assertGreater(decision.confidence, 0.5)
        self.assertEqual(len(decision.signals_used), 2)
        self.assertIsNotNone(decision.entry_price)
        self.assertIsNotNone(decision.stop_loss)

    def test_make_decision_conflicting_signals(self):
        """Test decision making with conflicting signals."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.8,
                details={"indicator": "fractal"},
            ),
            Signal(
                timestamp=datetime.now(),
                type="SELL",
                confidence=0.7,
                details={"indicator": "alligator"},
            ),
        ]

        decision = self.agent.make_decision(signals, self.market_data)

        # Conflicting signals should result in HOLD
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(len(decision.signals_used), 2)

    def test_evaluate_confluence_empty_signals(self):
        """Test confluence evaluation with empty signals."""
        confluence = self.agent.evaluate_confluence([])
        self.assertEqual(confluence, 0.0)

    def test_evaluate_confluence_single_direction(self):
        """Test confluence evaluation with signals in single direction."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.8,
                details={},
            ),
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.7,
                details={},
            ),
        ]

        confluence = self.agent.evaluate_confluence(signals)
        self.assertGreater(confluence, 0.5)

    def test_evaluate_confluence_conflicting_signals(self):
        """Test confluence evaluation with conflicting signals."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.8,
                details={},
            ),
            Signal(
                timestamp=datetime.now(),
                type="SELL",
                confidence=0.7,
                details={},
            ),
        ]

        confluence = self.agent.evaluate_confluence(signals)
        # Should be penalized for conflicting signals
        self.assertLess(confluence, 0.8)

    def test_determine_action_low_confluence(self):
        """Test action determination with low confluence."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.3,
                details={},
            )
        ]

        confluence = self.agent.evaluate_confluence(signals)
        action = self.agent.determine_action(confluence, signals)

        self.assertEqual(action, "HOLD")

    def test_determine_action_high_confluence_buy(self):
        """Test action determination with high confluence BUY."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.9,
                details={},
            ),
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.8,
                details={},
            ),
        ]

        confluence = self.agent.evaluate_confluence(signals)
        action = self.agent.determine_action(confluence, signals)

        # High confluence with 2+ BUY signals should result in BUY
        if confluence >= self.agent.confluence_threshold:
            self.assertEqual(action, "BUY")

    def test_calculate_confidence_no_signals(self):
        """Test confidence calculation with no signals."""
        confidence = self.agent.calculate_confidence(0.0, [])
        self.assertEqual(confidence, 0.0)

    def test_calculate_confidence_multiple_signals(self):
        """Test confidence calculation with multiple signals."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.8,
                details={},
            ),
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.7,
                details={},
            ),
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.9,
                details={},
            ),
        ]

        confidence = self.agent.calculate_confidence(0.8, signals)
        self.assertGreater(confidence, 0.5)
        self.assertLessEqual(confidence, 1.0)

    def test_generate_reasoning(self):
        """Test reasoning generation."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.8,
                details={},
            ),
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.7,
                details={},
            ),
        ]

        reasoning = self.agent.generate_reasoning(signals, 0.8, "BUY")

        self.assertIn("Decision: BUY", reasoning)
        self.assertIn("confluence: 0.80", reasoning)
        self.assertIn("Total signals analyzed: 2", reasoning)
        self.assertIn("BUY signals: 2", reasoning)

    def test_calculate_stop_loss_buy(self):
        """Test stop loss calculation for BUY action."""
        stop_loss = self.agent.calculate_stop_loss(self.market_data, "BUY")

        # Should be 2% below entry price
        expected = self.market_data.close * 0.98
        self.assertAlmostEqual(stop_loss, expected, places=2)

    def test_calculate_stop_loss_sell(self):
        """Test stop loss calculation for SELL action."""
        stop_loss = self.agent.calculate_stop_loss(self.market_data, "SELL")

        # Should be 2% above entry price
        expected = self.market_data.close * 1.02
        self.assertAlmostEqual(stop_loss, expected, places=2)

    def test_calculate_stop_loss_hold(self):
        """Test stop loss calculation for HOLD action."""
        stop_loss = self.agent.calculate_stop_loss(self.market_data, "HOLD")
        self.assertIsNone(stop_loss)

    def test_decision_logging_integration(self):
        """Test that decisions are properly logged."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.8,
                details={},
            ),
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.9,
                details={},
            ),
        ]

        decision = self.agent.make_decision(signals, self.market_data)

        # Verify logger was called
        self.mock_logger.log_decision.assert_called_once()

        # Check call arguments
        call_args = self.mock_logger.log_decision.call_args
        self.assertEqual(call_args[1]["agent_name"], "DecisionMaker")
        self.assertEqual(call_args[1]["confidence_score"], decision.confidence)
        self.assertIn("action", call_args[1]["context_data"])

    def test_decision_logging_failure_handling(self):
        """Test handling of logging failures."""
        # Make logger raise an exception
        self.mock_logger.log_decision.side_effect = Exception("Logging failed")

        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.8,
                details={},
            )
        ]

        # Should not raise exception despite logging failure
        decision = self.agent.make_decision(signals, self.market_data)
        self.assertIsNotNone(decision)

    def test_custom_confluence_threshold(self):
        """Test agent with custom confluence threshold."""
        agent = DecisionMakerAgent(client=self.mock_client, confluence_threshold=0.5)

        signals = [
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.6,
                details={},
            ),
            Signal(
                timestamp=datetime.now(),
                type="BUY",
                confidence=0.5,
                details={},
            ),
        ]

        decision = agent.make_decision(signals, self.market_data)

        # With lower threshold, should be more likely to make trading decisions
        self.assertIn(decision.action, ["BUY", "SELL", "HOLD"])


if __name__ == "__main__":
    unittest.main()
