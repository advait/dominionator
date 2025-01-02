import logging

import maturin_import_hook
from maturin_import_hook.settings import MaturinSettings

logging.basicConfig(format="%(name)s [%(levelname)s] %(message)s", level=logging.DEBUG)
maturin_import_hook.reset_logger()
maturin_import_hook.install(
    settings=MaturinSettings(
        uv=True,
    ),
    enable_automatic_installation=True,
)

import dominionator._rust as rust  # noqa: F403

print(rust.N_EMBEDDINGS)  # type: ignore
