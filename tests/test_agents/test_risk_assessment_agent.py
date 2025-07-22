import unittest
from src.agents.risk_assessment_agent import RiskAssessmentAgent, RiskAssessmentResult

class TestRiskAssessmentAgent(unittest.TestCase):
    def setUp(self):
        self.agent = RiskAssessmentAgent(max_risk_per_trade=0.02, account_balance=10000.0)

    def test_basic_position_sizing_buy(self):
        decision = {
            'action': 'BUY',
            'entry_price': 100.0,
            'stop_loss': 98.0,
            'position_size': 1.0
        }
        result = self.agent.assess(decision)
        self.assertGreater(result.position_size, 0.0)
        self.assertEqual(result.stop_loss, 98.0)
        self.assertEqual(result.risk_score, 1.0)

    def test_basic_position_sizing_sell(self):
        decision = {
            'action': 'SELL',
            'entry_price': 100.0,
            'stop_loss': 102.0,
            'position_size': 1.0
        }
        result = self.agent.assess(decision)
        self.assertGreater(result.position_size, 0.0)
        self.assertEqual(result.stop_loss, 102.0)
        self.assertEqual(result.risk_score, 1.0)

    def test_no_stop_loss(self):
        decision = {
            'action': 'BUY',
            'entry_price': 100.0,
            'stop_loss': None,
            'position_size': 1.0
        }
        result = self.agent.assess(decision)
        self.assertEqual(result.position_size, 0.0)
        self.assertEqual(result.risk_score, 0.0)

    def test_low_win_rate_adjustment(self):
        decision = {
            'action': 'BUY',
            'entry_price': 100.0,
            'stop_loss': 98.0,
            'position_size': 1.0
        }
        backtest_metrics = {'win_rate': 0.35}
        result = self.agent.assess(decision, backtest_metrics)
        self.assertLess(result.risk_score, 1.0)
        self.assertEqual(result.notes, 'Low historical win rate; risk reduced.')

if __name__ == "__main__":
    unittest.main()
