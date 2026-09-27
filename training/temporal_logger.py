import logging
import sys

from training.temporal_config import TEMPORAL_LOG_DIR


TEMPORAL_LOGGER = logging.getLogger("training.temporal")
TEMPORAL_LOGGER.setLevel(logging.INFO)
TEMPORAL_LOGGER.handlers.clear()
TEMPORAL_LOGGER.propagate = False

_FORMATTER = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

_FILE_HANDLER = logging.FileHandler(
    TEMPORAL_LOG_DIR / "training.log",
    encoding="utf-8",
)
_FILE_HANDLER.setFormatter(_FORMATTER)

_CONSOLE_HANDLER = logging.StreamHandler(sys.stdout)
_CONSOLE_HANDLER.setFormatter(_FORMATTER)

TEMPORAL_LOGGER.addHandler(_FILE_HANDLER)
TEMPORAL_LOGGER.addHandler(_CONSOLE_HANDLER)
