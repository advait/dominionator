import logging
import sys
from pathlib import Path

import maturin_import_hook
from maturin_import_hook.settings import MaturinSettings

logging.basicConfig(format="%(name)s [%(levelname)s] %(message)s", level=logging.DEBUG)
maturin_import_hook.reset_logger()
maturin_import_hook.install(
    enable_reloading=True,
    settings=MaturinSettings(
        uv=True,
    ),
    enable_automatic_installation=True,
)

from dominionator._rust import *  # noqa: E402, F403

# Ensure that the parent directory of this file exists on Python path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

print(sum_as_string(1, 2))  # type: ignore
