import unittest
from src.backtesting.config import BacktestConfig
from src.backtesting.engine import BacktestEngine
from src.visualization.charts import ChartVisualizer
from src.visualization.config import ChartConfig, ChartTheme
from src.data.data_provider import DataProvider
from src.agents.risk_assessment_agent import RiskAssessmentAgent

class TestEndToEndWorkflow(unittest.TestCase):
    def test_full_system_integration(self):
        # Data fetch
        provider = DataProvider(symbol="AAPL", interval="1d", start="2024-01-01", end="2024-01-31")
        series = provider.fetch()
        df = series.to_dataframe()
        self.assertGreater(len(df), 0)

        # Backtesting
        config = BacktestConfig(
            symbol="AAPL",
            start_date=df.index[0],
            end_date=df.index[-1],
            initial_capital=10000.0,
            signal_confidence_threshold=0.7,
            decision_confidence_threshold=0.6,
        )
        engine = BacktestEngine(config)
        results = engine.run_backtest()
        self.assertIsNotNone(results)
        self.assertGreaterEqual(len(results.trades), 0)

        # Risk assessment
        risk_agent = RiskAssessmentAgent(account_balance=10000.0)
        for trade in results.trades[:3]:  # Check first 3 trades
            decision = {
                'action': trade.trade_type,
                'entry_price': trade.entry_price,
                'stop_loss': trade.stop_loss,
                'position_size': trade.position_size
            }
            risk_result = risk_agent.assess(decision)
            self.assertIsNotNone(risk_result)

        # Visualization
        chart_config = ChartConfig(figure_size=(10, 6))
        visualizer = ChartVisualizer(config=chart_config)
        fig = visualizer.create_backtesting_summary_chart(
            data=df,
            backtest_results=results,
            title="Integration Test Chart"
        )
        self.assertIsNotNone(fig)

if __name__ == "__main__":
    unittest.main()
