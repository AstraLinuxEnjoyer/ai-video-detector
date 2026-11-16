"""Reuse the audited local DFDC split without rebuilding or downloading media."""
import json
from collections import Counter
from pathlib import Path

from training.spatial.common import ROOT, atomic_json, fingerprint
from training.spatial.data import assert_disjoint

SPECTRAL_SNAPSHOT = ROOT / 'cache/spectral/runs/spectral_face_v3/spectral_run.json'
SPATIAL_PARENT = ROOT / 'cache/artifact_spatial/runs/hybrid_v2'
MOTION_PARENT = ROOT / 'cache/temporal_v2/runs/motion_flow_v1'
SPATIAL_FINAL = SPATIAL_PARENT.with_name('final_hybrid_v1')
MOTION_FINAL = MOTION_PARENT.with_name('final_motion_v1')
SUSPECT = 'figure status update - openai speech-to-speech reasoning.mp4'


def merge_splits(parent, cohort, failed_ids):
    if cohort['parent_fingerprint'] != fingerprint(parent) or fingerprint(cohort['splits']) != cohort['fingerprint']:
        raise ValueError('DFDC snapshot does not match the parent split or has changed')
    splits = {}
    for split, rows in parent.items():
        base = [row for row in rows if row['id'] not in failed_ids and
                not (row['label'] == 1 and Path(row['path']).name.casefold() == SUSPECT)]
        dfdc = [row for row in cohort['splits'][split] if row['source'] == 'DFDC']
        if any(row.get('partition') != split for row in dfdc):
            raise ValueError('DFDC family assigned to an unexpected split')
        splits[split] = base + dfdc
        if len({r['id'] for r in splits[split]}) != len(splits[split]):
            raise ValueError(f'Duplicate video ID in {split}')
    assert_disjoint(splits)
    return splits


def final_splits(parent, run):
    saved = json.loads(SPECTRAL_SNAPSHOT.read_text(encoding='utf-8'))
    if saved['config']['recipe'] != 'spectral-face-v3' or saved['config']['counts']['train'] != 0:
        raise ValueError('Expected the completed spectral-face-v3 full pool')
    coverage = json.loads(SPECTRAL_SNAPSHOT.with_name('train_coverage.json').read_text(encoding='utf-8'))
    if not SPECTRAL_SNAPSHOT.with_name('final_report.json').is_file():
        raise ValueError('The audited spectral run must have finished before using its DFDC split')
    failed = {row['id'] for row in coverage['failures']}
    if not failed <= {row['id'] for row in parent['train']}:
        raise ValueError('Unexpected failed media outside the old train pool; review before proceeding')
    splits = merge_splits(parent, saved['cohort'], failed)
    atomic_json(run / 'final_cohort.json', dict(source=str(SPECTRAL_SNAPSHOT),
                parent_fingerprint=saved['cohort']['parent_fingerprint'],
                excluded_train_decode_ids=sorted(failed), excluded_label_filename=SUSPECT,
                sources={split: dict(Counter(f"{r['label']}/{r['source']}" for r in rows))
                         for split, rows in splits.items()}))
    return splits


def heldout_selection(rows, count, seed):
    from training.spatial.data import fixed_selection

    # Reserve a bounded set of complete DFDC pairs, not a generator-heavy random draw.
    pairs = [row for row in rows if row['source'] == 'DFDC']
    paired = {}
    for row in pairs:
        paired.setdefault(row['group'], []).append(row)
    groups = [group for group in paired.values() if sorted(row['label'] for row in group) == [0, 1]]
    groups.sort(key=lambda group: fingerprint([seed, group[0]['group']]))
    selected = [row for group in groups[:count//8] for row in group]
    return selected + fixed_selection([r for r in rows if r['source'] != 'DFDC'], count-len(selected), seed)
