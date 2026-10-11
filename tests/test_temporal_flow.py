import argparse
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from training.spatial.media import FrameCache
from training.temporal_model import build_temporal_model
from training.temporal_v2 import flow, media
from training.temporal_v2.run import DEFAULTS, MOTION_DEFAULTS, choose_validation, initialize
from training.temporal_v2 import engine
from training.spatial.common import fingerprint
from training.spatial.engine import atomic_torch


class TinyFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Sequential(torch.nn.Conv3d(2, 2, 1), torch.nn.BatchNorm3d(2), torch.nn.ReLU())
        self.layer4 = torch.nn.Conv3d(2, 2, 1)
        self.fc = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(self.layer4(self.stem(x)).mean((2, 3, 4)))


class Monitor:
    def publish(self, **values):
        pass

    record = publish


class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fixed_velocity_units_and_no_rgb_input(self):
        rng = np.random.default_rng(42)
        texture = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        frames = []
        for offset in (0, 2, 4):
            image = np.zeros((96, 96, 3), np.uint8)
            image[30:62, 20+offset:52+offset] = texture[..., None]
            frames.append(image)
        fields = flow.estimate_flow(np.stack(frames), 4)
        self.assertEqual(fields.shape, (2, 96, 96, 2))
        self.assertGreater(float(np.median(fields[:, 38:54, 32:44, 0])), .05)
        x = flow.tensor_from_flow(fields, 64, .5)
        self.assertEqual(x.shape, (2, 2, 64, 64))
        self.assertTrue(torch.isfinite(x).all())
        np.testing.assert_array_equal(flow.flow_colors(torch.zeros_like(x)), 0)
        colored = np.stack(frames).copy()
        # Identical luminance with arbitrary RGB channels produces the same flow.
        gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in colored])
        np.testing.assert_allclose(fields, flow.estimate_flow(np.repeat(gray[..., None], 3, -1), 4), atol=.001)

    def test_camera_translation_is_removed_not_local_motion(self):
        base = np.random.default_rng(7).integers(0, 256, (128, 128, 3), dtype=np.uint8)
        pixels = np.stack([base, np.roll(base, 2, axis=1)])
        value = flow.estimate_flow(pixels, 4).astype(np.float32)
        self.assertLess(float(np.abs(value[:, 24:-24, 24:-24]).mean()), .01)

    def test_vector_augmentation_is_consistent_and_fixed_scale(self):
        fields = np.zeros((8, 128, 128, 2), np.float16)
        fields[..., 0] = .125
        fields[..., 1] = .0625
        for seed in range(10):
            a = flow.tensor_from_flow(fields, 112, .5, seed)
            b = flow.tensor_from_flow(fields, 112, .5, seed)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(a[:, 0], a[:, -1], rtol=0, atol=0)
            self.assertTrue(torch.all(a[1] > 0))
            self.assertAlmostEqual(float(a[0, 0, 0, 0].abs()/a[1, 0, 0, 0]), 2.)
        self.assertTrue(any(flow.tensor_from_flow(fields, 112, .5, seed)[0, 0, 0, 0] < 0 for seed in range(10)))
        doubled = flow.tensor_from_flow(fields*2, 128, .5)
        torch.testing.assert_close(doubled, flow.tensor_from_flow(fields, 128, .5)*2)

    def test_flow_dataset_never_opens_rgb_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'flow.npy'
            np.save(path, np.ones((8, 32, 32, 2), np.float16)*.1)
            records = [dict(id='one', label=1, clips=['missing_rgb.npy'], flow_clips=[str(path)])]
            dataset = media.Clips(records, 32, input_mode='flow')
            x, y, index = dataset[0]
            self.assertEqual(x.shape, (2, 8, 32, 32))
            self.assertEqual((y, index), (1., 0))

    def test_model_has_two_channels_and_zero_rgb_running_statistics(self):
        model = build_temporal_model(False, 'flow')
        self.assertEqual(model.stem[0].in_channels, 2)
        self.assertEqual(model.fc.out_features, 1)
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 2, 8, 32, 32))
        self.assertEqual(output.shape, (1, 1))
        self.assertTrue(torch.isfinite(output).all())
        del model

    def test_fps_matched_rotation_and_validation_are_deterministic(self):
        args = argparse.Namespace(**{**DEFAULTS, **MOTION_DEFAULTS, 'val_videos': 4})
        rows = []
        for label in (0, 1):
            for fps in (8, 24):
                for index in range(3):
                    rows.append(dict(id=f'{label}_{fps}_{index}', label=label, source=f'source{label}_{index%2}',
                                     metadata=dict(fps=fps, width=640, height=480)))
        a = choose_validation(rows, args)
        self.assertEqual(a, choose_validation(rows, args))
        sampler = flow.rotation_for(rows, args)
        chosen = sampler.take(8)
        counts = {}
        for r in chosen:
            cell = flow.motion_bucket(r)
            counts.setdefault(cell, [0, 0])[r['label']] += 1
        self.assertTrue(all(real == fake for real, fake in counts.values()))
        b = flow.rotation_for(rows, args, state=sampler.state).take(8)
        self.assertNotEqual([r['id'] for r in chosen], [r['id'] for r in b])

    def test_low_fps_is_abstention_not_fake_label(self):
        args = argparse.Namespace(**{**DEFAULTS, **MOTION_DEFAULTS})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'video.mp4'
            path.write_bytes(b'video')
            row = dict(id='one', path=str(path), size=5, mtime=path.stat().st_mtime_ns,
                       group='one', label=0, source='real')
            with patch.object(media, 'probe', return_value=dict(duration=5., fps=2)), patch.object(media, 'decode_clip') as decoder:
                prepared = media.prepare([row], FrameCache(root, 1, 0), 'external', 1, args,
                                         logging.getLogger('test'), root, allow_short=True)
                self.assertEqual(prepared, [])
                decoder.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, 'no replacement'):
                    media.prepare([row], FrameCache(root, 1, 0), 'val', 1, args, logging.getLogger('test'), root)
            report = json.loads((root/'external_decode.json').read_text())
            self.assertEqual(report['unsupported'][0]['reason'], 'native_fps_below_target')

    def test_controls_remove_time_information_without_mutating_input(self):
        x = torch.randn(2, 2, 8, 16, 16)
        before = x.clone()
        zero = flow.motion_variant(x, 'zero')
        self.assertFalse(zero.any())
        mean = flow.motion_variant(x, 'mean_field')
        torch.testing.assert_close(mean[:, :, 0], mean[:, :, -1])
        shuffled = flow.motion_variant(x, 'shuffle')
        torch.testing.assert_close(shuffled.sum(2), x.sum(2))
        torch.testing.assert_close(x, before, rtol=0, atol=0)

    def test_rgb_run_cannot_be_converted_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config.json').write_text(json.dumps(dict(DEFAULTS, recipe='temporal-v2')))
            with self.assertRaisesRegex(ValueError, 'NEW --run-dir'):
                initialize(root, None, dict(input_mode='flow'), logging.getLogger('test'))

    def test_robot_demo_is_quarantined_not_relabelled_or_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'Figure Status Update - OpenAI Speech-to-Speech Reasoning.mp4'
            path.write_bytes(b'original')
            suspect = dict(id='robot', path=str(path), label=1, source='OpenAI_Sora')
            good = dict(id='generated', path=str(root/'Tokyo walk.mp4'), label=1, source='OpenAI_Sora')
            rows = flow.quarantine_labels(dict(train=[suspect, good], val=[], test=[], ood=[]), root,
                                          logging.getLogger('test'))
            self.assertEqual(rows['train'], [good])
            self.assertEqual(suspect['label'], 1)
            self.assertEqual(path.read_bytes(), b'original')
            report = json.loads((root/'label_quarantine.json').read_text())
            self.assertEqual(report['records'][0]['id'], 'robot')

    def test_header_audit_reuses_cache_and_freezes_eligibility(self):
        args = argparse.Namespace(**{**DEFAULTS, **MOTION_DEFAULTS})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = dict(train=[], val=[], test=[], ood=[])
            for split in ('train', 'val'):
                for label in (0, 1):
                    path = root/f'{split}_{label}.mp4'
                    path.write_bytes(b'video')
                    splits[split].append(dict(id=path.stem, path=str(path), label=label,
                                              size=5, mtime=path.stat().st_mtime_ns, source=f'source{label}'))
            meta = dict(width=640, height=480, fps=24., duration=6.)
            with patch.object(flow.av, 'open', side_effect=ValueError('fallback')), \
                    patch.object(flow, 'probe', return_value=meta) as probe:
                first = flow.audit_headers(splits, root, args, logging.getLogger('test'))
                self.assertEqual(probe.call_count, 4)
                second = flow.audit_headers(splits, root, args, logging.getLogger('test'))
                self.assertEqual(probe.call_count, 4)
                self.assertEqual(first, second)
            journal = root/'motion_headers.jsonl'
            entries = [json.loads(line) for line in journal.read_text().splitlines()]
            entries[0]['metadata']['fps'] = 25.
            journal.write_text('\n'.join(json.dumps(row) for row in entries)+'\n')
            with self.assertRaisesRegex(ValueError, 'eligibility/matching changed'):
                flow.audit_headers(splits, root, args, logging.getLogger('test'))

    def test_motion_checkpoint_replays_exactly_including_batchnorm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(**{**DEFAULTS, **MOTION_DEFAULTS, 'device': 'cpu', 'epochs': 1,
                       'videos_per_epoch': 4, 'batch_size': 2, 'workers': 0, 'image_size': 32,
                       'min_source_count': 1, 'preview_count': 2, 'save_every': 1, 'log_every': 50,
                       'reserve_gb': 0., 'max_fake_source_fraction': 1.})
            splits = {}
            for split, count in (('train', 2), ('val', 1)):
                rows = []
                for label in (0, 1):
                    for index in range(count):
                        name = f'{split}_{label}_{index}'
                        raw, field = root/f'{name}.npy', root/f'{name}_flow.npy'
                        np.save(raw, np.zeros((9, 32, 32, 3), np.uint8))
                        np.save(field, np.full((8, 32, 32, 2), .03+label*.1+index*.01, np.float16))
                        rows.append(dict(id=name, path=str(raw), group=name, label=label,
                                         source=f'source{label}', clips=[str(raw)], flow_clips=[str(field)],
                                         starts=[0.], metadata=dict(width=640, height=480, fps=24)))
                splits[split] = rows
            payload = dict(splits=splits, fingerprint=fingerprint(splits))
            torch.manual_seed(123)
            initial = TinyFlow().state_dict()
            a, b = root/'a', root/'b'
            for run in (a, b):
                run.mkdir()
                atomic_torch(run/'initial.pt', initial)
            original, calls = engine.optimizer_batch, 0
            def interrupt(*values, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt()
                return original(*values, **kwargs)
            with patch.object(engine, 'build_temporal_model', lambda *_: TinyFlow()), \
                    patch.object(engine, 'prepare', lambda rows, *_, **__: rows):
                engine.train(a, payload, splits['train'], splits['val'], args, logging.getLogger('test'), Monitor())
                with patch.object(engine, 'optimizer_batch', interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        engine.train(b, payload, splits['train'], splits['val'], args, logging.getLogger('test'), Monitor())
                saved = torch.load(b/'checkpoint.pt', weights_only=False)
                self.assertEqual(saved['offset'], 2)
                self.assertEqual(saved['schema'], 'temporal-motion-v1')
                engine.train(b, payload, splits['train'], splits['val'], args, logging.getLogger('test'), Monitor())
            first = torch.load(a/'checkpoint.pt', weights_only=False)
            resumed = torch.load(b/'checkpoint.pt', weights_only=False)
            for name, value in first['model'].items():
                torch.testing.assert_close(value, resumed['model'][name], rtol=0, atol=0)
            self.assertFalse(torch.equal(initial['stem.1.running_mean'], resumed['model']['stem.1.running_mean']))
            report = json.loads((b/'epoch_001.json').read_text())
            self.assertEqual(set(report['motion_controls']['variants']), {'zero', 'mean_field', 'shuffle'})


if __name__ == '__main__':
    unittest.main()
