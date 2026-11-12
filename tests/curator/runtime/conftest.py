"""Makes the merged repository importable for the runtime test suite."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

