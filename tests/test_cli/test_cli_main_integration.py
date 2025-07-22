import unittest
import subprocess
import sys
import os

class TestCLIIntegration(unittest.TestCase):
    def test_version_command(self):
        result = subprocess.run([sys.executable, '-m', 'src.cli.main', 'version'], capture_output=True, text=True)
        self.assertIn('Trading Bot CLI', result.stdout)

    def test_config_show(self):
        result = subprocess.run([sys.executable, '-m', 'src.cli.main', 'config', '--show'], capture_output=True, text=True)
        # Accept either 'data', 'backtest', or any config key
        self.assertTrue('data' in result.stdout or 'backtest' in result.stdout)

    def test_data_fetch(self):
        result = subprocess.run([sys.executable, '-m', 'src.cli.main', 'data', '--symbol', 'AAPL', '--interval', '1d', '--start', '2024-01-01', '--end', '2024-01-10'], capture_output=True, text=True)
        self.assertIn('Fetched', result.stdout)

    def test_backtest(self):
        result = subprocess.run([sys.executable, '-m', 'src.cli.main', 'backtest', '--symbol', 'AAPL', '--days', '5'], capture_output=True, text=True)
        self.assertIn('Backtesting Results', result.stdout)

    def test_visualize(self):
        result = subprocess.run([sys.executable, '-m', 'src.cli.main', 'visualize', '--symbol', 'AAPL', '--days', '5', '--output', 'outputs/charts/test_cli_chart.png'], capture_output=True, text=True)
        self.assertIn('Chart saved to:', result.stdout)

if __name__ == "__main__":
    unittest.main()
