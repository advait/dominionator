import sys
from pathlib import Path

# Ensure that the parent directory of this file exists on Python path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import dominionator_rust  # noqa: F401

print(dominionator_rust.sum_as_string(1, 2))
