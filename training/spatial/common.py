import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1


def replace_file(source, destination):
    """Keep the old file intact while transient Windows readers release it."""
    for attempt in range(7):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(min(.01 * 2**attempt, .16))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name+".", suffix=".tmp", delete=False) as stream:
            tmp = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        replace_file(tmp, path)
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                logging.getLogger("spatial_v2").warning("Temporary JSON cleanup deferred: %s", tmp)


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
