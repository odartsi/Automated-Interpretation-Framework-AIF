"""
Pytest configuration and shared fixtures for unit tests.
Adds src to Python path so tests can import application modules.
"""
import sys
from pathlib import Path

# Add src to path when running tests from project root
_root = Path(__file__).resolve().parent.parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
