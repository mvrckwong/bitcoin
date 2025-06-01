"""
Bitcoin Trading Model Tests

This package contains all tests for the Bitcoin trading model project.

Test Structure:
- unit/: Unit tests for individual components
- integration/: Integration tests for module interactions
- model_validation/: Model validation and performance tests

Usage:
    python -m pytest tests/
    python -m pytest tests/unit/
    python -m pytest tests/integration/
    python -m pytest tests/model_validation/
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports during testing
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path)) 