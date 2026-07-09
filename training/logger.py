import logging

from training.config import LOG_DIR


LOGGER = logging.getLogger("training")

LOGGER.setLevel(logging.INFO)

LOGGER.handlers.clear()


formatter = logging.Formatter(

    "%(asctime)s | %(levelname)s | %(message)s"

)


file_handler = logging.FileHandler(

    LOG_DIR / "training.log",

    encoding="utf-8"

)

file_handler.setFormatter(formatter)


console_handler = logging.StreamHandler()

console_handler.setFormatter(formatter)


LOGGER.addHandler(file_handler)

LOGGER.addHandler(console_handler)