import unittest
from src.cli import config

class TestCLIConfig(unittest.TestCase):
    def test_config_module_import(self):
        self.assertTrue(hasattr(config, 'load_config'))
        self.assertTrue(hasattr(config, 'validate_config'))

    def test_load_and_validate_config(self):
        cfg = config.load_config(None)
        self.assertTrue(config.validate_config(cfg))
        # Accept top-level keys 'data' and 'backtest' for config structure
        self.assertIn('data', cfg)
        self.assertIn('backtest', cfg)

if __name__ == "__main__":
    unittest.main()
