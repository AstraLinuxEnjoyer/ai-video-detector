import argparse
import copy
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tests.test_artifact_spatial import Tiny, fixture
from training.artifact_spatial import engine
from training.artifact_spatial.objective import paired_loss
from training.artifact_spatial.patches import PatchFrames, aggregate_views
from training.artifact_spatial.run import DEFAULTS, initialize
from training.artifact_spatial.sampling import MatchedRotation, distribution, spatial_bucket
from training.artifact_spatial import metadata
from training.spatial.common import atomic_json, fingerprint


def records():
    rows = []
    for label in (0, 1):
        for source in (["UCF-101", "phone", "nature"] if label == 0 else ["generatorA", "generatorB"]):
            for cell, (w, h) in enumerate(((320, 240), (1280, 720))):
                for i in range(15):
                    rows.append(dict(id=f"{source}-{cell}-{i}", source=source, label=label,
                                     metadata=dict(width=w, height=h)))
    return rows


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sampler_balance_caps_and_resume(self):
        rows = records()
        sampler = MatchedRotation(rows)
        a = sampler.take(40)
        self.assertEqual(len({r['id'] for r in a}), 40)
        self.assertLessEqual(sum(r['source'] == 'UCF-101' for r in a), 4)
        for value in distribution(a).values():
            self.assertEqual(value['0'], value['1'])
        restored = MatchedRotation(rows, state=json.loads(json.dumps(sampler.state)))
        b = sampler.take(40)
        self.assertEqual(b, restored.take(40))
        self.assertFalse({r['id'] for r in a} & {r['id'] for r in b})

    def test_unmatched_not_silently_used_and_no_short_epochs(self):
        rows = records() + [dict(id='portrait', label=1, source='other', metadata=dict(width=720, height=1280))]
        sampler = MatchedRotation(rows)
        self.assertEqual(sampler.audit['unmatched_ids'], ['portrait'])
        self.assertNotIn('portrait', [r['id'] for r in sampler.take(40)])
        with self.assertRaisesRegex(ValueError, 'only'):
            MatchedRotation(rows).take(1000)
        with self.assertRaises(ValueError):
            MatchedRotation(rows).take(3)

    def test_sequence_dimensions_and_missing_metadata(self):
        self.assertEqual(spatial_bucket(dict(id='x', sequence_width=1920, sequence_height=1080)), 'high/landscape')
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            spatial_bucket(dict(id='x'))

    def test_header_audit_restart_quarantine_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'video.mp4'
            path.write_bytes(b'not a valid video')
            stat = path.stat()
            bad = dict(id='broken', path=str(path), size=stat.st_size, mtime=stat.st_mtime_ns, label=1, source='broken')
            valid = [dict(id=f'known{i}', metadata=dict(width=320, height=240)) for i in range(100)]
            with patch.object(metadata, 'av', None), patch.object(metadata, 'probe', side_effect=ValueError('broken header')):
                index = metadata.audit_dimensions(valid+[bad], root, logging.getLogger('metadata-test'))
            self.assertIn('error', index['broken'])
            self.assertEqual(len(metadata.attach_dimensions(valid+[bad], index)), 100)
            with patch.object(metadata, 'probe', side_effect=AssertionError('must reuse completed audit')):
                self.assertEqual(metadata.audit_dimensions(valid+[bad], root, logging.getLogger('metadata-test')), index)
            with self.assertRaisesRegex(RuntimeError, '1%'):
                metadata.audit_dimensions([bad], root, logging.getLogger('metadata-test'))

    def test_header_change_is_fatal_not_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'changed.mp4'
            path.write_bytes(b'changed')
            row = dict(id='x', path=str(path), size=100, mtime=1)
            with self.assertRaises(metadata.SourceChangedError):
                metadata.audit_dimensions([row], root, logging.getLogger('metadata-test'))

    def test_global_view_label_independent_and_inference_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = fixture(Path(tmp))['splits']['train']
            changed = copy.deepcopy(rows)
            for row in changed:
                row['label'] = 1-row['label']
                row['source'] = 'other'
            a = PatchFrames(rows, 32, train=True, hybrid=True)
            b = PatchFrames(changed, 32, train=True, hybrid=True)
            self.assertEqual(a[0][0].shape, (2, 3, 3, 32, 32))
            torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
            self.assertEqual(PatchFrames(rows, 32, hybrid=True)[0][0].shape, (6, 3, 32, 32))
            self.assertAlmostEqual(float(aggregate_views(torch.tensor([1., 0., 0., 0., 0., 0.]), True)), .5)

    def test_loss_does_not_dilute_global_branch_with_patch_count(self):
        labels, weights = torch.tensor([1.]), torch.ones(1)
        logits = torch.tensor([[[2., -1., -1.], [2., -1., -1.]]], requires_grad=True)
        expanded = torch.cat([logits[..., :1], logits[..., 1:].repeat(1, 1, 3)], -1)
        a = paired_loss(logits, labels, weights, hybrid=True)[0]
        b = paired_loss(expanded, labels, weights, hybrid=True)[0]
        torch.testing.assert_close(a, b)
        a.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_end_to_end_resume_reports_and_parent_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = fixture(root)
            for rows in payload['splits'].values():
                for row in rows:
                    row['metadata'] = dict(width=80, height=48)
            payload['fingerprint'] = fingerprint(payload['splits'])
            parent = root/'parent'
            parent.mkdir()
            torch.save(Tiny().state_dict(), parent/'best.pt')
            atomic_json(parent/'best.json', dict(epoch=0, split_hash=payload['fingerprint']))
            atomic_json(parent/'splits.json', payload)
            common = dict(recipe='hybrid-v2', epochs=2, videos_per_epoch=8, val_videos=8,
                          batch_size=2, image_size=32, workers=0, device='cpu', save_every=1, min_source_count=1)
            runs = [root/'whole', root/'resumed']
            for run in runs:
                run.mkdir()
                initialize(run, parent, common)
            args = argparse.Namespace(**{**DEFAULTS, **common})
            log = logging.getLogger('hybrid-test')
            with patch.object(engine, 'build_model', lambda *a: Tiny()), patch.object(engine, 'prepare', lambda r,*a: sorted(r,key=lambda x:x['id'])):
                engine.train(runs[0], payload, args, log)
                save = engine.save_state
                def stop(*a):
                    save(*a)
                    if a[-1]['epoch'] == 1 and a[-1]['offset'] == 4:
                        raise KeyboardInterrupt()
                with patch.object(engine, 'save_state', stop):
                    with self.assertRaises(KeyboardInterrupt):
                        engine.train(runs[1], payload, args, log)
                engine.train(runs[1], payload, args, log)
            whole, resumed = [torch.load(r/'checkpoint.pt', weights_only=False) for r in runs]
            for field in ('rotation', 'seen', 'hard', 'best'):
                self.assertEqual(whole[field], resumed[field])
            for key in whole['model']:
                torch.testing.assert_close(whole['model'][key], resumed['model'][key], rtol=0, atol=0)
            report = json.loads((runs[0]/'epoch_002.json').read_text())
            self.assertEqual(report['clean']['frame_metrics']['n'], 8)
            self.assertIn('global_only', report['clean'])
            self.assertIn('by_spatial_bucket', report['clean'])
            self.assertIn('global_score', (runs[0]/'epoch_002_clean.csv').read_text().splitlines()[0])
            self.assertEqual(json.loads((runs[0]/'best.json').read_text())['recipe'], 'hybrid-v2')
            self.assertEqual(json.loads((runs[0]/'splits.json').read_text()), payload)
            with self.assertRaisesRegex(ValueError, 'frozen'):
                initialize(runs[0], parent, dict(recipe='patch-v1'))


if __name__ == '__main__':
    unittest.main()
