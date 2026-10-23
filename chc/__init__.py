"""CHC: controlled intervention experiments, not a certified causal graph system.

The package separates observable policy state from benchmark reference labels.
See docs/METHODOLOGY.md before interpreting an intervention as a causal effect.
"""

import sys
from pathlib import Path

# Missing optional packages can be installed privately without replacing the
# user's global Python environment. Ordinary pip installs remain supported.
_local_dependencies = Path(__file__).resolve().parents[1] / '.chc_deps'
if _local_dependencies.exists() and str(_local_dependencies) not in sys.path:
    sys.path.insert(0, str(_local_dependencies))

__version__ = '0.2.0'
