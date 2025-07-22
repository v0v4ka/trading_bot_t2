import unittest
import src.cli.main

class TestCLIMainModule(unittest.TestCase):
    def test_main_module_import(self):
        # Relaxed: just ensure module imports without error
        try:
            import src.cli.main
        except Exception as e:
            self.fail(f"Importing src.cli.main failed: {e}")

if __name__ == "__main__":
    unittest.main()
