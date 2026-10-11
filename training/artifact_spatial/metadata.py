"""Restartable header audit for matching. Never decodes/re-encodes originals."""
import json
from concurrent.futures import as_completed
from pathlib import Path

from tqdm import tqdm

try:
    import av
except ImportError:
    av = None

from training.spatial.common import atomic_json, event
from training.spatial.media import decoding_pool, probe, SourceChangedError


def attach_dimensions(records, index):
    result = []
    for record in records:
        if record.get('kind') == 'image_sequence' or (record.get('metadata') or {}).get('width'):
            result.append(record)
        else:
            saved = index[record['id']]
            if (saved['size'], saved['mtime']) != (record['size'], record['mtime']):
                raise SourceChangedError(f"Metadata snapshot mismatch: {record['path']}")
            if saved.get('error'):
                continue
            result.append(dict(record, metadata=dict(width=saved['width'], height=saved['height'])))
    return result


def audit_dimensions(records, run, logger, workers=4):
    journal = run / 'dimensions_progress.jsonl'
    index = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                saved = json.loads(line)
                index[saved['id']] = saved
            except json.JSONDecodeError:
                # A killed process can leave only its last journal line incomplete.
                continue
    pending = [r for r in records if r.get('kind') != 'image_sequence'
               and not (r.get('metadata') or {}).get('width') and r['id'] not in index]
    logger.info('DIMENSIONS audit remaining=%d cached=%d (headers only; originals unchanged)', len(pending), len(index))

    def one(record):
        path = Path(record['path'])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (record['size'], record['mtime']):
            raise SourceChangedError(f"Source changed: {path}")
        width, height, backend = 0, 0, 'pyav_header'
        if av is not None:
            try:
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    width, height = stream.codec_context.width, stream.codec_context.height
            except (OSError, ValueError, IndexError):
                pass
        if width <= 0 or height <= 0:
            meta = probe(path)
            width, height, backend = meta['width'], meta['height'], 'ffprobe_fallback'
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {path}")
        return dict(id=record['id'], size=record['size'], mtime=record['mtime'],
                    width=width, height=height, backend=backend)

    errors = []
    with decoding_pool(workers) as pool:
        futures = {pool.submit(one, row): row for row in pending}
        for step, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc='AUDIT DIMENSIONS'), 1):
            record = futures[future]
            try:
                saved = future.result()
                index[record['id']] = saved
                event(journal, saved)
            except SourceChangedError:
                for other in futures:
                    other.cancel()
                raise
            except Exception as exc:
                detail = dict(id=record['id'], path=record['path'], error=str(exc))
                errors.append(detail)
                rejected = dict(detail, size=record['size'], mtime=record['mtime'], backend='rejected')
                index[record['id']] = rejected
                event(journal, rejected)
                event(run/'metadata_errors.jsonl', detail)
                logger.error('Header audit failed: %s', detail)
            if step % 1000 == 0:
                logger.info('DIMENSIONS %d/%d errors=%d', step, len(pending), len(errors))
    rejected = [dict(index[r['id']], source=r['source'], label=r['label']) for r in records
                if r['id'] in index and index[r['id']].get('error')]
    atomic_json(run/'quarantine.json', dict(records=rejected, count=len(rejected),
                policy='Training only; originals and partitions untouched. More than 1% blocks preparation.'))
    if len(rejected) > .01*len(records):
        raise RuntimeError('More than 1% failed headers. Inspect quarantine.json before training.')
    if rejected:
        logger.warning('QUARANTINED %d train files; see quarantine.json, no replacement labels or modified originals', len(rejected))
    attach_dimensions(records, index)
    atomic_json(run/'dimensions.json', index)
    logger.info('DIMENSIONS complete files=%d ffprobe_fallback=%d', len(index),
                sum(r['backend'] == 'ffprobe_fallback' for r in index.values()))
    return index
