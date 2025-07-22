import os
import sys

# Ensure project root is on path for tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Automatically load .env before any tests
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(ROOT, '.env'))
except ImportError:
    pass
