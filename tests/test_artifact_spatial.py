import argparse
import copy
import json
import logging
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from training.artifact_spatial import engine
from training.artifact_spatial.objective import paired_loss, hard_weights, update_history, selection_score
from training.artifact_spatial.patches import PatchFrames, degrade, patch_tensor
from training.artifact_spatial.run import DEFAULTS, initialize
from training.spatial.common import fingerprint, atomic_json
from training.spatial.metrics import grouped_report


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4),
                                            torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1))
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(4, 1))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def fixture(root):
    splits = {name: [] for name in ("train", "val", "test", "ood")}
    for split in ("train", "val", "test"):
        for i in range(8):
            path = root / f"{split}_{i}.png"
            Image.fromarray(np.random.default_rng(i).integers(0, 256, (48, 80, 3), dtype=np.uint8)).save(path)
            splits[split].append(dict(id=f"{split}_{i}", group=f"{split}_{i}", label=i%2,
                                     source=f"source{i%2}", path=str(path), frames=[str(path)]))
    return dict(schema=1, splits=splits, fingerprint=fingerprint(splits))


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_patches_repeatable_label_independent_and_fixed_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = fixture(Path(tmp))["splits"]["train"]
            a = PatchFrames(records, 32, train=True)
            changed = copy.deepcopy(records)
            for row in changed:
                row["label"] = 1-row["label"]
                row["source"] = "different-source"
            b = PatchFrames(changed, 32, train=True)
            self.assertEqual(a[0][0].shape, (2, 2, 3, 32, 32))
            torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
            self.assertFalse(torch.equal(a[0][0][0], a[0][0][1]))
            self.assertFalse(torch.equal(a[0][0], PatchFrames(records, 32, train=True, seed=43)[0][0]))
            self.assertEqual(PatchFrames(records, 32)[0][0].shape, (5, 3, 32, 32))

    def test_degradation_deterministic_and_preserves_dimensions(self):
        image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (60, 90, 3), dtype=np.uint8))
        self.assertEqual(degrade(image).size, image.size)
        np.testing.assert_array_equal(degrade(image, random.Random(42)), degrade(image, random.Random(42)))
        self.assertEqual(patch_tensor(Image.new("RGB", (10, 50)), 32).shape, (5, 3, 32, 32))

    def test_consistency_does_not_replace_supervised_loss(self):
        labels = torch.tensor([0., 1.])
        weights = torch.ones(2)
        good = torch.tensor([[[-3., -3.], [-3., -3.]], [[3., 3.], [3., 3.]]], requires_grad=True)
        loss, _, agreement = paired_loss(good, labels, weights)
        self.assertEqual(float(agreement.sum()), 0.)
        bad, _, _ = paired_loss(-good, labels, weights)
        collapsed, _, _ = paired_loss(torch.zeros_like(good), labels, weights)
        self.assertLess(float(loss.detach()), float(collapsed.detach()))
        self.assertLess(float(collapsed.detach()), float(bad.detach()))
        loss.backward()
        self.assertTrue(torch.isfinite(good.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_amp_overflow_replays_once_with_frozen_bn(self):
        model = Tiny().cuda().train()
        model.features[1].eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        scaler = torch.amp.GradScaler('cuda', init_scale=128.)
        calls = 0
        def overflow(gradient):
            nonlocal calls
            calls += 1
            return torch.full_like(gradient, float('inf')) if calls <= 2 else gradient
        model.classifier[1].weight.register_hook(overflow)
        with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
            result = engine.train_step(model, torch.randn(2, 2, 2, 3, 32, 32, device='cuda'),
                     torch.tensor([0., 1.], device='cuda'), torch.ones(2, device='cuda'), optimizer, scaler,
                     torch.device('cuda'))
        self.assertEqual(step.call_count, 1)
        self.assertEqual(len(result['retries']), 2)
        self.assertEqual(int(model.features[1].num_batches_tracked), 0)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_hard_weights_capped_source_relative_and_history_replayable(self):
        history = {}
        update_history(history, ['a', 'b', 'c'], ['real', 'real', 'fake'], [.1, 10., .5])
        weights = hard_weights(['a', 'b', 'c', 'new'], ['real', 'real', 'fake', 'fake'], history)
        self.assertTrue(all(1 <= w <= 2 for w in weights))
        self.assertGreater(weights[1], weights[0])
        self.assertEqual(weights[-1], 1.)
        self.assertEqual(weights, hard_weights(['a', 'b', 'c', 'new'], ['real', 'real', 'fake', 'fake'],
                                            json.loads(json.dumps(history))))

    def test_source_selection_cannot_hide_large_source_bias(self):
        rows = [dict(label=0, score=.1, source='easy') for _ in range(100)]
        rows += [dict(label=0, score=.9, source='hard') for _ in range(10)]
        rows += [dict(label=1, score=.9, source='fake') for _ in range(100)]
        rows += [dict(label=1, score=.9, source='tiny')]
        report = grouped_report(rows)
        result = selection_score(report, report)
        self.assertAlmostEqual(result['score'], .75)
        self.assertGreater(report['accuracy'], .95)
        self.assertEqual(result['clean']['excluded_small_groups'][0]['source'], 'tiny')

    def test_resume_and_hard_history_match_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = fixture(root)
            args = argparse.Namespace(**{**DEFAULTS, 'epochs': 2, 'videos_per_epoch': 8, 'val_videos': 8,
                    'batch_size': 2, 'image_size': 32, 'workers': 0, 'device': 'cpu', 'save_every': 1,
                    'min_source_count': 1})
            initial = Tiny().state_dict()
            whole, resumed = root/'whole', root/'resumed'
            for directory in (whole, resumed):
                directory.mkdir()
                torch.save(initial, directory/'initial.pt')
                atomic_json(directory/'lineage.json', dict(seen_ids=[]))
            log = logging.getLogger('artifact-test')
            def prepared(rows, *a):
                return sorted(rows, key=lambda r:r['id'])
            with patch.object(engine, 'build_model', lambda *a: Tiny()), patch.object(engine, 'prepare', prepared):
                engine.train(whole, payload, args, log)
                original_save = engine.save_state
                interrupted = False
                def save_then_interrupt(*a):
                    nonlocal interrupted
                    original_save(*a)
                    if a[-1]['epoch']==1 and a[-1]['offset']==4 and not interrupted:
                        interrupted = True
                        raise KeyboardInterrupt()
                with patch.object(engine, 'save_state', save_then_interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        engine.train(resumed, payload, args, log)
                engine.train(resumed, payload, args, log)
            a = torch.load(whole/'checkpoint.pt', map_location='cpu', weights_only=False)
            b = torch.load(resumed/'checkpoint.pt', map_location='cpu', weights_only=False)
            for field in ('hard', 'seen', 'rotation', 'epoch', 'offset', 'stale'):
                self.assertEqual(a[field], b[field])
            for name in a['model']:
                torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
            self.assertEqual(int(a['model']['features.1.num_batches_tracked']), 0)
            self.assertTrue(a['complete'])
            self.assertTrue(set(a['hard']) <= {r['id'] for r in payload['splits']['train']})

    def test_initialize_preserves_partitions_and_copies_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            parent, run = root/'parent', root/'new'
            parent.mkdir()
            run.mkdir()
            payload=fixture(root)
            atomic_json(parent/'splits.json', payload)
            atomic_json(parent/'best.json', dict(epoch=1, split_hash=payload['fingerprint']))
            atomic_json(parent/'epoch_001_plan.json', dict(ids=[r['id'] for r in payload['splits']['train']]))
            torch.save(Tiny().state_dict(), parent/'best.pt')
            # Metadata must be written after the weights, matching atomic best publication.
            atomic_json(parent/'best.json', dict(epoch=1, split_hash=payload['fingerprint']))
            config, cloned=initialize(run, parent, {})
            self.assertEqual(cloned['splits'], payload['splits'])
            self.assertTrue((run/'initial.pt').exists())
            with self.assertRaisesRegex(ValueError, 'frozen'):
                initialize(run, parent, {'batch_size': 1})
            with (run/'initial.pt').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                initialize(run, parent, {})


if __name__=='__main__':
    unittest.main()
