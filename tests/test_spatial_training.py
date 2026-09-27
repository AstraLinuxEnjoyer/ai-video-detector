import argparse
import copy
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from training.spatial.common import fingerprint
from training.spatial.engine import Frames, optimizer_batch, train
from training.spatial.run import DEFAULTS


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
                                             torch.nn.AdaptiveAvgPool2d(1))
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(4, 1))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def fixture(root):
    splits = {}
    for split in ("train", "val", "test"):
        records = []
        for index in range(8):
            path = root / f"{split}_{index}.png"
            rng = np.random.default_rng(index)
            Image.fromarray(rng.integers(0, 255, (40, 64, 3), dtype=np.uint8)).save(path)
            records.append(dict(id=f"{split}_{index}", group=f"{split}_{index}", path=str(path),
                                label=index % 2, source=f"source{index%2}", frames=[str(path)],
                                size=path.stat().st_size, metadata=dict(width=64, height=40, fps=24, duration=4)))
        splits[split] = records
    splits["ood"] = []
    return dict(splits=splits, fingerprint=fingerprint(splits))


def prepared(records, *_args, **_kwargs):
    return sorted(records, key=lambda r: r["id"]), []


class TrainingTests(unittest.TestCase):
    def numerical_fixture(self, device):
        model = TinyModel().to(device)
        model.features.insert(1, torch.nn.BatchNorm2d(4).to(device))
        return model, torch.randn(2, 3, 16, 16, device=device), torch.tensor([[0.], [1.]], device=device)

    @unittest.skipUnless(torch.cuda.is_available(), "AMP retry requires CUDA")
    def test_amp_overflow_retries_same_batch_and_restores_batchnorm(self):
        for failing_attempts in (1, 2):
            with self.subTest(failing_attempts=failing_attempts), tempfile.TemporaryDirectory() as tmp:
                model, images, labels = self.numerical_fixture("cuda")
                optimizer = torch.optim.SGD(model.parameters(), lr=.01)
                scaler = torch.amp.GradScaler("cuda", init_scale=128.)
                calls = 0

                def overflow(gradient):
                    nonlocal calls
                    calls += 1
                    return torch.full_like(gradient, float("inf")) if calls <= failing_attempts else gradient

                model.classifier[1].weight.register_hook(overflow)
                with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    loss, logits, norm, retries = optimizer_batch(model, images, labels, optimizer,
                        scaler, logging.getLogger("test"), Path(tmp), dict(step=1))
                self.assertEqual(step.call_count, 1)
                self.assertEqual(retries, failing_attempts)
                self.assertEqual(int(model.features[1].num_batches_tracked), 1)
                self.assertEqual(scaler.get_scale(), 128. / 2**failing_attempts)
                self.assertTrue(torch.isfinite(loss) and torch.isfinite(norm) and torch.isfinite(logits).all())
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_fp32_nonfinite_gradients_fail_without_updating_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, images, labels = self.numerical_fixture("cpu")
            before = copy.deepcopy(model.state_dict())
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            model.classifier[1].weight.register_hook(lambda g: torch.full_like(g, float("nan")))
            with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                with self.assertRaisesRegex(RuntimeError, "persist in FP32"):
                    optimizer_batch(model, images, labels, optimizer, torch.amp.GradScaler("cuda", enabled=False),
                                    logging.getLogger("test"), Path(tmp), dict(step=1))
            self.assertEqual(step.call_count, 0)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name])

    def test_sample_augmentation_stable_across_workers_and_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = fixture(Path(tmp))["splits"]["train"]
            a = Frames(rows, 32, train=True, seed=42)
            b = Frames(rows, 32, train=True, seed=42)
            self.assertTrue(torch.equal(a[0][0], b[0][0]))
            c = Frames(rows, 32, train=True, seed=43)
            self.assertFalse(torch.equal(a[0][0], c[0][0]))

    def test_resume_matches_uninterrupted_optimizer_steps(self):
        from training.spatial import engine
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = fixture(root)
            args = argparse.Namespace(**{**DEFAULTS, "epochs": 2, "videos_per_epoch": 8,
                "val_videos": 8, "image_size": 32, "batch_size": 2, "val_batch_size": 4,
                "workers": 0, "allow_cpu": True, "no_pretrained": True, "save_every": 1})
            whole, resumed = root / "whole", root / "resumed"
            whole.mkdir()
            resumed.mkdir()
            log = logging.getLogger("training_test")
            with patch.object(engine, "build_model", lambda **kw: TinyModel()), \
                 patch.object(engine, "prepare_frames", prepared):
                train(whole, payload, args, log)
                real_save = engine.save_state
                interrupted = False

                def save_then_interrupt(*a):
                    nonlocal interrupted
                    real_save(*a)
                    if a[-1]["offset"] == 4 and not interrupted:
                        interrupted = True
                        raise KeyboardInterrupt()

                with patch.object(engine, "save_state", save_then_interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        train(resumed, payload, args, log)
                train(resumed, payload, args, log)
            a = torch.load(whole / "checkpoint.pt", weights_only=False, map_location="cpu")
            b = torch.load(resumed / "checkpoint.pt", weights_only=False, map_location="cpu")
            self.assertEqual(a["seen"], b["seen"])
            self.assertEqual(a["rotation"], b["rotation"])
            self.assertEqual(a["epoch"], 2)
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], rtol=1e-5, atol=1e-6)

    def test_finetune_keeps_parent_when_no_validation_improvement(self):
        from training.spatial import engine
        from training.spatial.lineage import file_sha
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = fixture(root)
            initial = TinyModel().state_dict()
            parent = root / "parent.pt"
            torch.save(initial, parent)
            (root / "lineage.json").write_text(json.dumps(dict(weights_sha256=file_sha(parent), seen_ids=[])))
            args = argparse.Namespace(**{**DEFAULTS, "epochs": 1, "videos_per_epoch": 8,
                "val_videos": 8, "image_size": 32, "batch_size": 2, "val_batch_size": 4,
                "workers": 0, "allow_cpu": True, "init_model": str(parent), "parent_run": str(root)})
            original_evaluate = engine.evaluate

            def controlled_evaluate(*a):
                rows, report = original_evaluate(*a)
                report["balanced_accuracy"] = 1. if a[-1] == "INITIAL PARENT" else .5
                return rows, report

            with patch.object(engine, "build_model", lambda **kw: TinyModel()), \
                 patch.object(engine, "prepare_frames", prepared), \
                 patch.object(engine, "evaluate", controlled_evaluate):
                train(root, payload, args, logging.getLogger("test"))
            self.assertEqual(json.loads((root / "best.json").read_text())["epoch"], 0)
            best = torch.load(root / "best.pt", weights_only=True, map_location="cpu")
            for name in initial:
                torch.testing.assert_close(best[name], initial[name])


if __name__ == "__main__":
    unittest.main()
