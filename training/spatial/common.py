import hashlib
import json
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def logger_for(run):
    logger = logging.getLogger("spatial_v2")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(Path(run) / "training.log", encoding="utf-8"),
                    logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def event(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
