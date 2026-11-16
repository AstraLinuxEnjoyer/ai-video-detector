"""Frozen, source-stratified evaluation of the shipped TRACE ensemble."""

import argparse
import csv
import logging
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch

from demo.analysis import clip_positions, extract_frames, probe
from demo.assessment import assess_ensemble, ENSEMBLE_POLICY
from demo.inference import ROOT, Detector, sha256
from demo.spectral import SpectralDetector
from training.evaluate_trace import EvaluationJob, EXTENSIONS, MEDIA_ERRORS, inventory
from training.final_cohort import MOTION_FINAL, SPATIAL_PARENT, SPECTRAL_SNAPSHOT
from training.final_metrics import read_json
from training.spatial.common import atomic_json, fingerprint

SEED = 20261116
QUOTAS = {0: {'UCF-101': 28, 'VideoUFO': 24, 'KoNViD_1k_videos': 22, 'DeepAction:Pexels': 2},
          1: {'GenVidBench:vc2': 15, 'GenVidBench:ms': 15, 'GenVidBench:t2vz': 15,
              'GenVidBench:pika': 15, 'keling': 4, 'DeepAction:Veo': 4,
              'DeepAction:VideoPoet': 4, 'DeepAction:RunwayML': 4}}


def select_rows(snapshots):
    conflict_ids, conflict_groups, all_ids = set(), set(), set()
    for snapshot in snapshots:
        for split, rows in snapshot.items():
            all_ids.update(r['id'] for r in rows)
            if split in ('train', 'val'):
                conflict_ids.update(r['id'] for r in rows)
                conflict_groups.update(r['group'] for r in rows)
    used_ids, used_groups = set(), set()

    def allowed(r):
        return (r.get('kind') != 'image_sequence' and Path(r['path']).suffix.lower() in EXTENSIONS and
                r['id'] not in conflict_ids and r['group'] not in conflict_groups)

    source = snapshots[1]  # Broad frozen local holdout, not the smaller spectral sample.
    def rank(row):
        return fingerprint([SEED, row['id']])
    selected = []

    def add(row, cohort, allow_pair=False):
        if row['id'] in used_ids or (row['group'] in used_groups and not allow_pair):
            return False
        selected.append(dict(row, cohort=cohort))
        used_ids.add(row['id'])
        used_groups.add(row['group'])
        return True

    for label, sources in QUOTAS.items():
        for name, quota in sources.items():
            rows = [r for r in source['test'] if r['source'] == name and r['label'] == label and allowed(r)]
            count = 0
            for row in sorted(rows, key=rank):
                count += add(row, 'holdout')
                if count == quota:
                    break
    pairs = defaultdict(list)
    for row in source['test']:
        if row['source'] == 'DFDC' and allowed(row):
            pairs[row['group']].append(row)
    for group in sorted(pairs):
        pair = sorted(pairs[group], key=lambda r: r['label'])
        if [r['label'] for r in pair] == [0, 1]:
            for row in pair:
                add(row, 'holdout', allow_pair=True)
    by_source = defaultdict(list)
    for row in source['ood']:
        if allowed(row):
            by_source[row['source']].append(row)
    for name in sorted(by_source):
        count = 0
        for row in sorted(by_source[name], key=rank):
            count += add(row, 'ood')
            if count == 6:
                break
    personal_root = ROOT/'test_videos'
    for row in inventory(personal_root):
        if row['sha256'] in all_ids or row['sha256'] in used_ids:
            continue
        add(dict(id=row['sha256'], path=str(personal_root/row['file']), relative=row['file'],
                 source='personal/' + ('fake' if row['label'] else 'real'),
                 label=row['label'], group='personal/'+row['sha256']), 'personal')
    return selected


def score_row(row, detector, spectral, directory):
    path = Path(row['path'])
    if sha256(path) != row['id']:
        raise ValueError('Video no longer matches frozen SHA-256')
    job = EvaluationJob()
    meta = probe(path)
    frames = extract_frames(path, meta, job, directory)
    for frame in frames:
        frame.update(detector.spatial_score(directory/frame['image']))
    clips = []
    seconds = detector.args.clip_length/detector.args.fps
    positions = clip_positions(meta, detector.args)
    for start in positions:
        try:
            clips.append(dict(start=start, end=start+seconds, temporal=detector.motion_score(path, start, meta)))
        except MEDIA_ERRORS as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            logging.warning('Motion clip %.2f: %s', start, exc)
    try:
        spectrum = spectral.analyze(path, job, directory)
    except (*MEDIA_ERRORS, cv2.error) as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            raise
        spectrum = None
        logging.warning('Spectrum: %s', exc)
    a = assess_ensemble(frames, clips, meta['duration'], len(positions), spectrum)
    return dict(verdict=a['verdict'], reasons=a['reasons'], spatial=a['spatial'], temporal=a['temporal'],
                spectral=a['spectral'], combined=a['combined'], motion_status=a['coverage']['motion'],
                spectral_status=a['coverage']['spectral'], duration=meta['duration'],
                suspicious_seconds=a['suspicious_seconds'])


def summarize(rows):
    labels = {}
    for label, key in enumerate(('real', 'fake')):
        counts = Counter(r.get('verdict', 'error') for r in rows if r['label'] == label)
        labels[key] = dict(total=sum(counts.values()), **{k: counts[k] for k in
                           ('suspected', 'uncertain', 'no_signals', 'error')})
    return dict(n=len(rows), **labels, motion=dict(Counter(r.get('motion_status', 'error') for r in rows)),
                spectrum=dict(Counter(r.get('spectral_status', 'error') for r in rows)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    cli.add_argument('--output', type=Path, default=ROOT/'reports/trace_thesis_final')
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'report.json').exists():
        cli.error('Report already exists; choose a new --output for another experiment')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output/'evaluation.log', encoding='utf8')])
    paths = [SPATIAL_PARENT/'splits.json', MOTION_FINAL/'splits.json', SPECTRAL_SNAPSHOT]
    snapshots = []
    for path in paths:
        saved = read_json(path)
        saved = saved.get('cohort', saved)
        if fingerprint(saved['splits']) != saved['fingerprint']:
            raise ValueError(f'Frozen split changed: {path}')
        snapshots.append(saved['splits'])
    selected = select_rows(snapshots)
    os.environ['TRACE_DEVICE'] = args.device
    cv2.setNumThreads(1)
    detector, spectral = Detector(), SpectralDetector()
    for tag, run in (('spatial', SPATIAL_PARENT), ('temporal', MOTION_FINAL)):
        if detector.info[tag]['sha256'] != sha256(run/'best.pt'):
            raise ValueError(f'The deployed {tag} weights do not match this benchmark snapshot')
    if spectral.info['sha256'] != sha256(SPECTRAL_SNAPSHOT.with_name('model.json')):
        raise ValueError('The deployed spectral weights do not match this benchmark snapshot')
    models = dict(detector.describe(), spectral=spectral.info)
    fields = ('id','cohort','source','relative','group','label')
    atomic_json(args.output/'selection.json', dict(seed=SEED, models=models,
                input_snapshots=[p.relative_to(ROOT).as_posix() for p in paths],
                rows=[{k:r[k] for k in fields} for r in selected]))
    records = []
    for i, row in enumerate(selected, 1):
        start = time.monotonic()
        item = {k: row[k] for k in fields}
        with tempfile.TemporaryDirectory(dir=args.output) as temp:
            try:
                item.update(score_row(row, detector, spectral, Path(temp)))
            except (*MEDIA_ERRORS, cv2.error) as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
                item['error'] = str(exc)
        item['seconds'] = round(time.monotonic()-start, 3)
        records.append(item)
        logging.info('%d/%d %s %s %s %.2fs', i, len(selected), row['cohort'], row['source'],
                     item.get('verdict', item.get('error')), item['seconds'])
    r = dict(policy=ENSEMBLE_POLICY, models=models, requested=len(selected),
             cohorts={c: summarize([r for r in records if r['cohort'] == c])
                      for c in ('holdout', 'ood', 'personal')},
             by_source={f'{c}/{s}': summarize([r for r in records if (r['cohort'],r['source']) == (c,s)])
                        for c,s in sorted({(r['cohort'],r['source']) for r in records})})
    with (args.output/'predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in records for k in row}))
        writer.writeheader()
        writer.writerows(records)
    atomic_json(args.output/'report.json', r)
    logging.info('Report: %s', args.output/'report.json')


if __name__ == '__main__':
    main()
