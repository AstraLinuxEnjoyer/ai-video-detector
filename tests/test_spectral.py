import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.data.dfdc import build_index, family_split
from training.spatial.common import fingerprint
from training.spatial.lineage import file_sha
from training.spectral.cohort import build_cohort, paired_selection
from training.spectral.faces import native_box
from training.spectral.features import DIM, PATCH, frame_patches, jpeg_stress, patch_features, pool_features
from training.spectral.model import VIEWS, balanced_weights, fit, score, scores_for
from training.spectral.run import report_pairs


class DfdcTests(unittest.TestCase):
    def fixture(self, root):
        part = root / "dfdc_train_part_00"
        part.mkdir()
        rows = {}
        for split in ("train", "val", "test"):
            original = next(f"{split}-{i}.mp4" for i in range(1000)
                            if family_split(f"dfdc/{part.name}/{split}-{i}.mp4") == split)
            rows[original] = {"label": "REAL"}
            for variant in (1, 2):
                rows[f"{split}-fake{variant}.mp4"] = {"label": "FAKE", "original": original}
        for name in rows:
            (part / name).write_bytes(name.encode())
        (part / "metadata.json").write_text(json.dumps(rows), encoding="utf-8")
        return part, rows

    def test_index_pairs_group_and_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            index = build_index(root)
            self.assertEqual(len(index["records"]), 9)
            for split in ("train", "val", "test"):
                records = [r for r in index["records"] if r["partition"] == split]
                self.assertEqual(len({r["group"] for r in records}), 1)
                self.assertEqual([r["label"] for r in paired_selection(records, 1, 42)], [0, 1])
            self.assertEqual(build_index(root)["fingerprint"], index["fingerprint"])

    def test_index_rejects_missing_original_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            part, entries = self.fixture(root)
            entries["train-fake1.mp4"]["original"] = "../missing.mp4"
            (part / "metadata.json").write_text(json.dumps(entries), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "without a labeled original"):
                build_index(root)

    def test_index_rejects_unlabeled_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            part, _ = self.fixture(root)
            (part / "unknown.mp4").touch()
            with self.assertRaisesRegex(ValueError, "1 unlabeled"):
                build_index(root)

    def test_mixed_cohort_preserves_base_splits_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            dfdc = build_index(root)
            old = {split: [dict(id=f"{split}-{i}", group=f"{split}-{i}", label=i % 2, source="Old")
                           for i in range(6)] for split in ("train", "val", "test")}
            old["ood"] = []
            path = root / "base.json"

            def save():
                path.write_text(json.dumps(dict(splits=old, fingerprint=fingerprint(old))), encoding="utf-8")

            save()
            counts = dict(train=8, val=8, test=8, ood=0)
            result = build_cohort(path, dfdc, counts, .25, 42)
            for split in ("train", "val", "test"):
                selected = result["splits"][split]
                self.assertEqual({r["id"] for r in selected if r["source"] == "Old"},
                                 {r["id"] for r in old[split]})
                self.assertEqual(sum(r["source"] == "DFDC" for r in selected), 2)
            dfdc_test = next(r for r in result["splits"]["test"] if r["source"] == "DFDC")
            old["train"][0]["id"] = file_sha(Path(dfdc_test["path"]))
            save()
            with self.assertRaisesRegex(ValueError, "DFDC duplicate"):
                build_cohort(path, dfdc, counts, .25, 42)

    def test_all_train_uses_every_variant_and_keeps_eval_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            dfdc = build_index(root)
            old = {split: [dict(id=f"{split}-{i}", group=f"{split}-{i}", label=i % 2, source="Old",
                               path=str(root/f"{split}-{i}.mp4"))
                           for i in range(6)] for split in ("train", "val", "test")}
            old["ood"] = []
            path = root / "base.json"
            path.write_text(json.dumps(dict(splits=old, fingerprint=fingerprint(old))), encoding="utf-8")
            counts = dict(train=8, val=8, test=8, ood=0)
            base = build_cohort(path, dfdc, counts, .25, 42)["splits"]
            full = build_cohort(path, dfdc, dict(counts, train=0), .25, 42,
                                hash_database=root/"hashes.sqlite")["splits"]
            self.assertEqual(len(full["train"]), len(old["train"])+3)
            self.assertEqual(full["val"], base["val"])
            self.assertEqual(full["test"], base["test"])


class SpectralTests(unittest.TestCase):
    def test_fft_distinguishes_spatial_frequencies(self):
        x = np.arange(PATCH)
        pattern = lambda k: np.tile(np.rint(128 + 100*np.sin(2*np.pi*k*x/PATCH)).astype(np.uint8), (PATCH, 1))
        low, low_noise = patch_features(pattern(4))
        high, high_noise = patch_features(pattern(48))
        self.assertLess(np.argmax(low), np.argmax(high))
        self.assertLess(low_noise[1], high_noise[1])

    def test_finite_pooled_features_and_native_size_guard(self):
        gray = np.full((256, 256), 128, dtype=np.uint8)
        for view in (gray, jpeg_stress(gray)):
            patches = frame_patches(view)
            self.assertEqual(len(patches), 5)
            self.assertEqual(len(pool_features(patches)), DIM)
            self.assertTrue(np.isfinite(pool_features(patches)).all())
        with self.assertRaisesRegex(ValueError, "not upsampling"):
            frame_patches(gray[:64])

    def test_model_round_trip_and_three_ablations(self):
        x = np.vstack((np.zeros((6, DIM)), np.ones((6, DIM))))
        y = [0] * 6 + [1] * 6
        for mode in VIEWS:
            model = json.loads(json.dumps(fit(x, y, mode)))
            predictions = score(model, x)
            self.assertLess(max(predictions[:6]), .5)
            self.assertGreater(min(predictions[6:]), .5)

    def test_pair_metrics_report_unscored_pairs(self):
        rows = [dict(source="DFDC", group="a", label=0, score=.2),
                dict(source="DFDC", group="a", label=1, score=.7),
                dict(source="DFDC", group="b", label=1, score=.8)]
        report = report_pairs(rows, rows)
        self.assertEqual((report["requested"], report["complete"], report["fake_ranked_higher"]), (2, 1, 1))

    def test_native_face_box_ignores_detector_padding(self):
        detections = np.array([[10, 20, 30, 40, .9], [15, 300, 150, 150, .99]])
        box = native_box(detections, 640, 240, 320, 120)
        self.assertEqual(box, (14, 32, 86, 128))

    def test_balancing_does_not_reward_more_versions_of_one_original(self):
        rows = [dict(label=0, source="A", group="real"),
                dict(label=1, source="A", group="first"),
                dict(label=1, source="A", group="first"),
                dict(label=1, source="A", group="second"),
                dict(label=1, source="B", group="only")]
        np.testing.assert_allclose(balanced_weights(rows), [.5, .0625, .0625, .125, .25])

    def test_fusion_uses_full_score_when_no_face_is_available(self):
        x = np.vstack((np.zeros((4, DIM)), np.ones((4, DIM))))
        labels = [0]*4 + [1]*4
        fit_full = fit(x, labels, "combined")
        fit_face = fit(x, labels[::-1], "combined")
        rows = [dict(vectors={"clean": x[-1]}, face_vectors={"clean": x[-1]}),
                dict(vectors={"clean": x[-1]})]
        models = dict(combined=fit_full, face=fit_face, full_face=dict(face_weight=.5))
        face = scores_for(models, "face", rows)
        self.assertTrue(np.isnan(face[1]))
        whole = scores_for(models, "combined", rows)
        fused = scores_for(models, "full_face", rows)
        self.assertAlmostEqual(fused[0], (whole[0]+face[0])/2)
        self.assertEqual(fused[1], whole[1])


if __name__ == "__main__":
    unittest.main()
