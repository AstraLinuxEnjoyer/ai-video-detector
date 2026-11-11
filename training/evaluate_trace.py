"""Compare the deployed TRACE weights on the same independent videos."""

import argparse
import csv
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from demo.analysis import clip_positions, extract_frames, ffmpeg, filters, probe
from demo.assessment import FRAME_STEP, POLICY, THRESHOLD, assess, segments
from demo.inference import ROOT, Detector, sha256
from training.spatial.common import atomic_json
from training.spatial.metrics import grouped_report

INPUT = ROOT / 'test_videos' / 'benchmark'
EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
METHODS = ('spatial', 'temporal', 'fused_70_30', 'demo_alert', 'policy_alert')
MEDIA_ERRORS = (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired)


def inventory(root):
    rows = []
    for label, name in enumerate(('real', 'fake')):
        for path in sorted((root / name).rglob('*')):
            if path.suffix.lower() in EXTENSIONS and path.is_file():
                parts = path.relative_to(root).parts
                rows.append(dict(file=path.relative_to(root).as_posix(), label=label,
                                 source=parts[1] if len(parts) > 2 else f'unspecified/{name}',
                                 sha256=sha256(path)))
    if not rows:
        raise ValueError(f'No videos. Add clips to {root / "real"} and {root / "fake"}.')
    return rows


def known_files(info, requested):
    known, audit, snapshots = {}, {}, {}
    for kind, path in (('spatial', 'artifact_spatial/runs/hybrid_v2'),
                       ('temporal', 'temporal_v2/runs/motion_flow_v1')):
        run = ROOT / 'cache' / path
        if not all((run / name).is_file() for name in ('best.pt', 'best.json', 'splits.json')):
            audit[kind] = 'missing_local_training_snapshot'
            continue
        if sha256(run / 'best.pt') != info[kind]['sha256']:
            audit[kind] = 'deployed_weights_do_not_match_local_best'
            continue
        best = json.loads((run / 'best.json').read_text(encoding='utf-8'))
        snapshot = run / 'splits.json'
        digest = sha256(snapshot)
        if digest not in snapshots:
            data = json.loads(snapshot.read_text(encoding='utf-8'))
            matches = {row['id']: split for split, rows in data['splits'].items()
                       for row in rows if row['id'] in requested}
            snapshots[digest] = (data['fingerprint'], matches)
        split_hash, matches = snapshots[digest]
        if split_hash != best.get('split_hash'):
            audit[kind] = 'local_best_and_snapshot_disagree'
            continue
        audit[kind] = 'checked_exact_sha256'
        for key, split in matches.items():
            known.setdefault(key, []).append(f'{kind}/{split}')
    return known, audit


def eligible(rows, known):
    seen, rejected, selected = {}, [], []
    for row in rows:
        digest = row['sha256']
        if digest in seen and row['label'] != seen[digest]['label']:
            raise ValueError(f'Conflicting labels for identical videos: {row["file"]} and {seen[digest]["file"]}')
        if digest in known:
            rejected.append(dict(row, reason='already_in_model_dataset', detail=known[digest]))
        elif digest in seen:
            rejected.append(dict(row, reason='exact_duplicate', detail=seen[digest]['file']))
        else:
            selected.append(row)
        seen.setdefault(digest, row)
    return selected, rejected


def read_annotations(root, rows):
    path = root / 'segments.json'
    if not path.is_file():
        return {}
    truth = json.loads(path.read_text(encoding='utf-8-sig'))
    fake = {row['file'] for row in rows if row['label'] == 1}
    if not isinstance(truth, dict) or set(truth) - fake:
        raise ValueError('segments.json must map existing fake/ paths to changed time ranges.')
    for file, intervals in truth.items():
        if not isinstance(intervals, list) or not intervals:
            raise ValueError(f'{file}: expected a nonempty list of [start, end] seconds')
        previous = 0.
        for pair in intervals:
            if (not isinstance(pair, list) or len(pair) != 2 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in pair) or
                    not previous <= pair[0] < pair[1]):
                raise ValueError(f'{file}: ranges must be finite, positive in length, sorted and non-overlapping')
            previous = pair[1]
    return truth


class EvaluationJob:
    def check(self):
        pass  # Ctrl+C unwinds the shared decoder and terminates FFmpeg.


def score_video(path, detector, directory):
    started = time.monotonic()
    directory.mkdir()
    meta = probe(path)
    frames = extract_frames(path, meta, EvaluationJob(), directory)
    for frame in frames:
        frame.update(detector.spatial_score(directory / frame.pop('image')))
    positions = clip_positions(meta, detector.args)
    clips, errors = [], []
    seconds = detector.args.clip_length / detector.args.fps
    for start in positions:
        try:
            score = detector.motion_score(path, start, meta)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            errors.append(dict(start=start, error=str(exc)))
        else:
            clips.append(dict(start=start, end=start + seconds, temporal=score))
    duration = meta['duration']
    assessment = assess(frames, clips, duration, len(positions))
    full_motion = assessment['temporal'] if assessment['coverage']['motion'] == 'complete' else None
    combined = assessment['combined'] if full_motion is not None else None
    regions = dict(spatial=segments(frames, [], duration), temporal=segments([], clips, duration),
                   demo_alert=assessment['candidate_segments'], policy_alert=assessment['segments'])
    return dict(meta=meta, frames=frames, clips=clips, clip_errors=errors, expected_clips=len(positions),
                assessment=assessment,
                temporal_status=('complete' if full_motion is not None else 'clip_decode_failure' if errors
                                 else 'short_or_low_fps'), regions=regions,
                scores=dict(spatial=assessment['spatial'], temporal=full_motion, fused_70_30=combined,
                            demo_alert=float(bool(regions['demo_alert'])),
                            policy_alert=float(assessment['verdict'] == 'suspected')),
                elapsed_seconds=time.monotonic() - started)


def transcode(path, destination, meta, directory):
    directory.mkdir()
    if filters(meta) or meta['width'] % 2 or meta['height'] % 2:
        raise ValueError('Compression control requires SDR and even dimensions; clean scores are unaffected')
    ffmpeg(['-i', str(path), '-map', '0:v:0', '-an', '-sn', '-dn', '-c:v', 'libx264',
            '-preset', 'veryfast', '-crf', '30', '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough',
            '-threads', '1', str(destination)], EvaluationJob(), directory)


def timeline_measure(row, key):
    data = row['result']
    truth = row['intervals']
    if truth is None or (key == 'temporal' and data['temporal_status'] != 'complete'):
        return None
    spans = [(p['start'], p['end']) for p in data['regions'][key]]
    predicted = sum(end - start for start, end in spans)
    duration = data['meta']['duration']
    covered = [sum(max(0., min(end, b) - max(start, a)) for a, b in spans) for start, end in truth]
    return dict(predicted_seconds=predicted, truth_seconds=sum(b - a for a, b in truth),
                overlap_seconds=sum(covered), real_seconds=duration if row['label'] == 0 else 0.,
                real_highlight_seconds=predicted if row['label'] == 0 else 0.,
                real_videos=int(row['label'] == 0), real_flagged=int(row['label'] == 0 and predicted > 0),
                intervals=len(truth), intervals_half_covered=sum(c >= (b-a)/2 for c, (a, b) in zip(covered, truth)))


def summarize_timeline(rows):
    reports = {}
    for key in ('spatial', 'temporal', 'demo_alert', 'policy_alert'):
        measures = [m for r in rows if 'result' in r and (m := timeline_measure(r, key)) is not None]
        sums = {name: sum(m[name] for m in measures) for name in measures[0]} if measures else {}
        reports[key] = dict(videos=len(measures), **sums)
    return reports


def available(row, method):
    return row.get('result', {}).get('scores', {}).get(method) is not None


def method_reports(rows):
    reports = {}
    for method in METHODS:
        eligible_rows = [dict(label=r['label'], source=r['source'], score=r['result']['scores'][method])
                         for r in rows if available(r, method)]
        report = grouped_report(eligible_rows, THRESHOLD) if eligible_rows else dict(n=0, by_source={})
        if method in ('demo_alert', 'policy_alert'):
            for group in [report, *report['by_source'].values()]:
                for name in ('auroc', 'average_precision', 'brier', 'log_loss', 'score_quantiles'):
                    group.pop(name, None)
        report['requested_by_class'] = {str(label): sum(r['label'] == label for r in rows) for label in (0, 1)}
        report['scored_by_class'] = {str(label): sum(r['label'] == label for r in eligible_rows) for label in (0, 1)}
        reports[method] = report
    return reports


def summary(rows):
    shared = [r for r in rows if available(r, 'temporal')]
    verdicts = {str(label): {key: sum(r['label'] == label and
                 r.get('result', {}).get('assessment', {}).get('verdict') == key for r in rows)
                 for key in ('suspected', 'uncertain', 'no_signals')} for label in (0, 1)}
    return dict(all=method_reports(rows), same_videos=method_reports(shared), timeline=summarize_timeline(rows),
                failures=sum('result' not in r for r in rows), full_motion=len(shared), policy_verdicts=verdicts)


def compression_comparison(clean, compressed):
    pairs = list(zip(clean, compressed, strict=True))
    reports = {}
    for method in METHODS:
        matched = [(a, b) for a, b in pairs if available(a, method) and available(b, method)]
        before, after = [a for a, _ in matched], [b for _, b in matched]
        flips = sum((a['result']['scores'][method] >= THRESHOLD) !=
                    (b['result']['scores'][method] >= THRESHOLD) for a, b in matched)
        reports[method] = dict(n=len(matched), changed_decisions=flips, clean=method_reports(before)[method],
                               compressed=method_reports(after)[method])
    return reports


def write_csv(path, rows):
    keys = ('file', 'label', 'source', 'treatment', *METHODS, 'temporal_status', 'frames', 'clips',
            'expected_clips', 'highlighted_seconds', 'policy_verdict', 'policy_highlighted_seconds',
            'elapsed_seconds', 'error')
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            result = row.get('result', {})
            out = {k: row[k] for k in ('file', 'label', 'source', 'treatment')}
            out.update(result.get('scores', {}), temporal_status=result.get('temporal_status'),
                       frames=len(result.get('frames', [])), clips=len(result.get('clips', [])),
                       expected_clips=result.get('expected_clips'), elapsed_seconds=result.get('elapsed_seconds'),
                       highlighted_seconds=sum(s['end']-s['start'] for s in result.get('regions', {}).get('demo_alert', [])),
                       policy_verdict=result.get('assessment', {}).get('verdict'),
                       policy_highlighted_seconds=result.get('assessment', {}).get('suspicious_seconds'),
                       error=row.get('error'))
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--no-reencode', action='store_true')
    args = parser.parse_args()
    root = args.input.resolve()
    rows = inventory(root)
    truth = read_annotations(root, rows)
    os.environ['TRACE_DEVICE'] = args.device
    detector = Detector()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    known, audit = known_files(detector.info, {r['sha256'] for r in rows})
    selected, rejected = eligible(rows, known)
    if not selected:
        raise ValueError('Every input matched the existing model data; no independent clips remain.')
    output = ROOT / 'reports' / f'trace_benchmark_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', handlers=[
        logging.FileHandler(output / 'evaluation.log', encoding='utf-8'), logging.StreamHandler()])
    log = logging.getLogger('trace_benchmark')
    log.info('Output: %s | input=%s | candidates=%d selected=%d excluded=%d | audit=%s',
             output, root, len(rows), len(selected), len(rejected), audit)
    clean, compressed = [], []
    for row in tqdm(selected, desc='TRACE benchmark', unit='video'):
        path = root / row['file']
        intervals = [] if row['label'] == 0 else truth.get(row['file'])
        with tempfile.TemporaryDirectory(prefix='work_', dir=output) as tmp:
            directory = Path(tmp)
            for treatment in (('clean',) if args.no_reencode else ('clean', 'h264_crf30')):
                entry = dict(row, treatment=treatment, intervals=intervals)
                try:
                    if treatment == 'clean':
                        data = score_video(path, detector, directory / 'clean')
                    else:
                        original = clean[-1].get('result')
                        if original is None:
                            raise ValueError('Clean evaluation failed; no compression pair')
                        target = directory / 'compressed.mp4'
                        transcode(path, target, original['meta'], directory / 'encode')
                        data = score_video(target, detector, directory / 'compressed')
                        if abs(data['meta']['duration'] - original['meta']['duration']) > .1:
                            raise ValueError('Compression changed the playable duration by more than 0.1s')
                    if intervals and intervals[-1][1] > data['meta']['duration'] + .001:
                        raise ValueError('Annotated interval extends beyond the evaluated video')
                except MEDIA_ERRORS as exc:
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        raise
                    entry['error'] = str(exc)
                    log.warning('%s %s: %s', treatment, row['file'], exc)
                else:
                    entry['result'] = data
                    log.info('%s %s label=%s scores=%s motion=%s (%d/%d) seconds=%.2f',
                             treatment, row['file'], row['label'], data['scores'], data['temporal_status'],
                             len(data['clips']), data['expected_clips'], data['elapsed_seconds'])
                (clean if treatment == 'clean' else compressed).append(entry)
    report = dict(models=detector.describe(), policy=POLICY, threshold=THRESHOLD, frame_step=FRAME_STEP,
                  clip_seconds=detector.args.clip_length / detector.args.fps, audit=audit,
                  input=str(root), excluded=rejected, clean=summary(clean),
                  caveat='Exact matching cannot detect derivatives or shared scenes. No thresholds tuned on this set. '
                         'demo_alert is legacy OR. policy_alert is current suspected flag; zero also includes uncertain, '
                         'not a real label. See policy_verdicts for the separate outcomes.')
    if compressed:
        report.update(h264_crf30=summary(compressed), compression_pairs=compression_comparison(clean, compressed))
    atomic_json(output / 'report.json', report)
    atomic_json(output / 'predictions.json', clean + compressed)
    write_csv(output / 'predictions.csv', clean + compressed)
    for treatment in ('clean', 'h264_crf30') if compressed else ('clean',):
        for method, values in report[treatment]['same_videos'].items():
            log.info('%s matched %s | n=%d TP=%s TN=%s FP=%s FN=%s AUROC=%s',
                     treatment, method, values['n'], *(values.get(k) for k in ('tp', 'tn', 'fp', 'fn', 'auroc')))
    log.info('Report: %s', output / 'report.json')


if __name__ == '__main__':
    main()
