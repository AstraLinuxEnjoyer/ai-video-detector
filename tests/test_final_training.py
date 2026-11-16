import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from training.final_cohort import heldout_selection, merge_splits
from training.final_metrics import comparison, low_fpr_selection
from training.artifact_spatial.sampling import MatchedRotation
from training.spatial.common import fingerprint
from training.spatial.lineage import file_sha
from training.temporal_v2.flow import tensor_from_flow
from training.temporal_v2.paired_batch import build_plans, verify_plans
from training.temporal_v2.run import verify_frozen_artifacts


def row(key, label, source='legacy', group=None, **extras):
    return dict(id=key, label=label, group=group or key, source=source, path=key+'.mp4', **extras)


def cohort(parent, dfdc):
    return dict(parent_fingerprint=fingerprint(parent), splits=dfdc, fingerprint=fingerprint(dfdc))


class FinalTrainingTests(unittest.TestCase):
    def test_frozen_paired_artifact_reopens_with_saved_checksum_key(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            names = {'initial.pt': 'initial.pt_sha256', 'dimensions.json': 'dimensions.json_sha256',
                     'previous_attempts.json': 'previous_attempts_sha256'}
            config = dict(paired_batch=True)
            for name, key in names.items():
                (run/name).write_text(name)
                config[key] = file_sha(run/name)
            verify_frozen_artifacts(run, config)
            (run/'previous_attempts.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Frozen artifact changed: previous_attempts'):
                verify_frozen_artifacts(run, config)

    def test_paired_experiment_uses_new_dfdc_variants_and_keeps_replay_balanced(self):
        meta = dict(width=1280, height=720, fps=25)
        data = [row(f'{group}-{kind}', int(kind != 'real'), 'DFDC', group, metadata=meta)
                for group in ('a','b','c','d') for kind in ('real','used','new')]
        data += [row(f'mix-{label}-{i}', label, f'mix{i%2}', metadata=meta)
                 for label in (0,1) for i in range(6)]
        args = SimpleNamespace(videos_per_epoch=8, epochs=2, max_fake_source_fraction=.6, seed=42)
        tried = {r['id'] for r in data if r['id'].endswith('-used')}
        result = build_plans(data, tried, args)
        self.assertEqual(result, build_plans(data, tried, args))
        ids = {r['id']: r for r in data}
        verify_plans(result, result['fingerprint'], ids.keys(), 2, 8)
        paired_groups = set()
        for p in result['plans']:
            self.assertEqual(Counter(ids[i]['label'] for i in p['ids']), {0: 4, 1: 4})
            self.assertEqual(p['sources']['DFDC'], 4)
            for real, fake in p['pairs']:
                self.assertEqual(ids[real]['group'], ids[fake]['group'])
                self.assertNotIn(fake, tried)
                self.assertNotIn(ids[real]['group'], paired_groups)
                paired_groups.add(ids[real]['group'])
        result['plans'][0]['ids'][0] = 'changed'
        with self.assertRaisesRegex(ValueError, 'plan changed'):
            verify_plans(result, result['fingerprint'], ids.keys(), 2, 8)

    def test_coverage_priority_across_sources_and_cells_survives_resume(self):
        records = [row(f'{cell}_{label}_{i}', label, f's{i%3}', cell=cell)
                   for cell in ('fresh', 'mixed', 'old') for label in (0, 1) for i in range(6)]
        inherited = {r['id'] for r in records if r['cell'] == 'old' or
                     (r['cell'] == 'mixed' and r['label'] == 0)}
        kwargs = dict(bucket=lambda r: r['cell'], previously_seen=inherited, prefer_unseen=True,
                      max_source_fraction_by_class={1: .75})
        rotation = MatchedRotation(records, **kwargs)
        first = rotation.take(8)
        self.assertTrue(all(r['cell'] == 'fresh' for r in first))
        resumed = MatchedRotation(records, state=rotation.state, **kwargs)
        second = rotation.take(8)
        self.assertEqual(second, resumed.take(8))
        self.assertEqual(len({r['id'] for r in second}), 8)
        self.assertEqual(Counter(r['cell'] for r in second if r['label'] == 0),
                         Counter(r['cell'] for r in second if r['label'] == 1))
        self.assertEqual(sum(r['id'] not in inherited | {p['id'] for p in first} for r in second), 6)

    def test_verified_dfdc_adds_to_parent_without_replacing_validation_or_sequences(self):
        parent = dict(train=[row('bad', 1), row('sequence', 0, kind='image_sequence')],
                      val=[row('old-real', 0)], test=[row('old-fake', 1)], ood=[])
        dfdc = {split: [row(split+str(label), label, 'DFDC', split, partition=split) for label in (0, 1)]
                for split in ('train', 'val', 'test')}
        dfdc['ood'] = []
        saved = cohort(parent, dfdc)
        result = merge_splits(parent, saved, {'bad'})
        self.assertEqual([r['id'] for r in result['train']], ['sequence', 'train0', 'train1'])
        self.assertEqual(result['val'][0]['id'], 'old-real')
        dfdc['val'][0]['group'] = 'train'
        with self.assertRaisesRegex(ValueError, 'Leakage'):
            merge_splits(parent, cohort(parent, dfdc), set())
        with self.assertRaisesRegex(ValueError, 'changed'):
            merge_splits(parent, saved, set())

    def test_holdout_reserves_a_complete_pair(self):
        rows = [row(str(i), i%2) for i in range(10)]
        rows += [row('p'+str(i), i%2, 'DFDC', f'pair{i//2}') for i in range(4)]
        selected = heldout_selection(rows, 8, 42)
        self.assertEqual(len(selected), 8)
        pair = [r for r in selected if r['source']=='DFDC']
        self.assertEqual({r['label'] for r in pair}, {0, 1})
        self.assertEqual(len({r['group'] for r in pair}), 1)

    def test_operating_point_not_fit_on_stress_and_comparison_uses_common_ids(self):
        clean = [row(str(i), 0, score=i/100) for i in range(20)] + [row('fake', 1, score=.7)]
        stress = [dict(r, score=.9) for r in clean]
        point = low_fpr_selection(dict(score=.7), clean, stress)['operating_point']
        self.assertAlmostEqual(point['clean']['false_positive_rate'], .05)
        self.assertEqual(point['stress']['false_positive_rate'], 1)
        with tempfile.TemporaryDirectory() as root:
            report = comparison(Path(root), 'test', {'clean': (clean, clean[1:])},
                                dict(baseline=point['threshold'], candidate=point['threshold']), clean)
        self.assertEqual(report['treatments']['clean']['common'], 20)
        self.assertEqual(report['treatments']['clean']['baseline_only'], ['0'])

    def test_motion_perturbation_is_deterministic_and_never_used_in_validation(self):
        field = np.random.default_rng(42).uniform(-.4, .4, (8, 40, 40, 2)).astype(np.float32)
        make = lambda seed, robust: tensor_from_flow(field, 32, .5, seed=seed, full_frame=True, robust_train=robust)
        torch.testing.assert_close(make(None, True), make(None, False))
        torch.testing.assert_close(make('fixed', True), make('fixed', True))
        self.assertTrue(any(not torch.equal(make(i, True), make(i, False)) for i in range(12)))
