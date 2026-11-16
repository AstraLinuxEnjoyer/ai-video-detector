"""Validation-only operating points and paired holdout comparisons."""
import json

from training.spatial.common import atomic_json
from training.spatial.metrics import grouped_report, select_threshold

TARGET_FPR = .05


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def operating_point(clean, stress):
    threshold = select_threshold([r['label'] for r in clean], [r['score'] for r in clean], TARGET_FPR)
    return dict(threshold=threshold, target_validation_fpr=TARGET_FPR,
                clean=grouped_report(clean, threshold), stress=grouped_report(stress, threshold))


def low_fpr_selection(selection, clean, stress):
    point = operating_point(clean, stress)
    utility = sum(point[mode]['recall_fake'] - point[mode]['false_positive_rate']
                  for mode in ('clean', 'stress')) / 2
    # Retain the source-balanced objective while preferring useful low-FPR operation.
    return dict(selection, score=(selection['score'] + (1+utility)/2)/2,
                source_balanced_score=selection['score'], operating_point=point)


def comparison(run, split, predictions, thresholds, requested):
    result = dict(requested=len(requested), thresholds=thresholds, treatments={},
                  caveat='Known held-out cohorts, not a fresh independent real-world test. '
                  'Thresholds selected on clean validation only; stress is not H.264 recompression.')
    groups, pairs = {}, []
    for row in requested:
        if row['source'] == 'DFDC':
            groups.setdefault(row['group'], []).append(row)
    for group in groups.values():
        if sorted(r['label'] for r in group) == [0, 1]:
            real, fake = sorted(group, key=lambda r: r['label'])
            pairs.append((real['id'], fake['id']))
    truth = {row['id']: row['label'] for row in requested}
    for mode, (baseline, candidate) in predictions.items():
        by_id = [{row['id']: row for row in rows} for rows in (baseline, candidate)]
        if any(len(index) != len(rows) or any(truth.get(key) != row['label'] for key, row in index.items())
               for rows, index in zip((baseline, candidate), by_id)):
            raise ValueError('Comparison received duplicate IDs, unexpected videos or inconsistent labels')
        common = sorted(by_id[0].keys() & by_id[1].keys())
        common_ids = set(common)
        complete_pairs = [(real, fake) for real, fake in pairs if real in common_ids and fake in common_ids]
        result['treatments'][mode] = dict(common=len(common), ids=common,
            coverage=dict(baseline=len(baseline), candidate=len(candidate)),
            baseline_only=sorted(by_id[0].keys() - by_id[1].keys()),
            candidate_only=sorted(by_id[1].keys() - by_id[0].keys()),
            **{name: grouped_report([index[item] for item in common], thresholds[name]) if common else None
               for name, index in zip(('baseline', 'candidate'), by_id)})
        result['treatments'][mode]['dfdc_pairs'] = dict(requested=len(pairs), common=len(complete_pairs),
            **{name: sum(index[fake]['score'] > index[real]['score'] for real, fake in complete_pairs)
               for name, index in zip(('baseline', 'candidate'), by_id)})
    atomic_json(run / f'{split}_comparison.json', result)
    return result
