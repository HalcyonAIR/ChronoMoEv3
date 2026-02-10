"""Run ChronoMoE integration validation tests."""

import sys
from pathlib import Path

# Add swiss-ai-MoE to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from chronomoe_integration.tests.test_integration import run_all_tests

if __name__ == "__main__":
    run_all_tests()
