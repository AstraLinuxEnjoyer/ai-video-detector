"""960px frame preparation in an isolated cache; no source videos are rewritten."""
import io
import json
import random
from concurrent.futures import as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from training.spatial.common import atomic_json, event, fingerprint
from training.spatial.media import decoding_pool, decode, probe, SourceChangedError
from training.spatial.sequences import sequence_unchanged


def prepare(records, cache, split, count, seed, logger, run, workers=2):
    def one(record):
        path = Path(record["path"])
        sequence = record.get("kind") == "image_sequence"
        if sequence:
            valid = sequence_unchanged(record)
        else:
            stat = path.stat()
            valid = (stat.st_size, stat.st_mtime_ns) == (record["size"], record["mtime"])
        if not valid:
            raise SourceChangedError(f"Source changed since audit: {path}")
        directory = cache.root / split / record["id"]
        signature = fingerprint(["patch960-v1", record["id"], record["mtime"], split, seed, count])
        index = directory / "frames.json"
        if index.exists():
            saved = json.loads(index.read_text())
            if saved["signature"] == signature and all(Path(p).exists() for p in saved["frames"]):
                return dict(record, **saved)
        meta = None if sequence else probe(path)
        end = len(record["frame_files"]) if sequence else max(0, meta["duration"]-max(.1, 2/max(meta["fps"], 1)))
        rng = random.Random(f"{seed}:{record['id']}:{split}")
        positions = [end*(i + (rng.random() if split.startswith("train_") else .5))/count for i in range(count)]
        frames = []
        for i, position in enumerate(positions):
            output = directory / f"{i:02}.png"
            if sequence:
                with Image.open(path / record["frame_files"][min(int(position), len(record["frame_files"])-1)]) as src:
                    image = src.convert("RGB")
                    image.thumbnail((960, 960), Image.Resampling.BICUBIC)
                    buffer = io.BytesIO()
                    image.save(buffer, "PNG")
                    content = buffer.getvalue()
            else:
                content = decode(path, position, meta, size=960)
            cache.put(output, content)
            frames.append(str(output))
        saved = dict(signature=signature, frames=frames, metadata=meta, positions=positions)
        cache.put(index, json.dumps(saved).encode())
        return dict(record, **saved)

    good, errors = [], []
    with decoding_pool(workers) as pool:
        futures = {pool.submit(one, r): r for r in records}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"PATCH DECODE {split}"):
            record = futures[future]
            try:
                good.append(future.result())
            except Exception as exc:
                detail = dict(id=record["id"], path=record["path"], split=split, error=str(exc))
                errors.append(detail)
                event(run / "data_errors.jsonl", detail)
                logger.error("Decode failed: %s", detail)
                if isinstance(exc, SourceChangedError) or isinstance(exc, OSError) and (
                        "BUDGET" in str(exc) or "RESERVE" in str(exc)):
                    for other in futures:
                        other.cancel()
                    raise
    atomic_json(run / f"{split}_decode.json", dict(requested=len(records), usable=len(good), errors=errors))
    if errors:
        raise RuntimeError(f"Incomplete {split}: {len(errors)} files failed. No sample replacement; see data_errors.jsonl")
    return sorted(good, key=lambda r: r["id"])
