import copy
import logging
import tempfile
import unittest
from pathlib import Path

from training.spatial.data import Rotation, assert_disjoint, deduplicate, identify, make_splits
from training.spatial.metrics import metrics, select_threshold


def record(i, label=0, source="camera", group=None):
    return dict(id=str(i), label=label, source=source, group=group or str(i),
                aliases=[dict(source=source)], path=str(i))


class DataTests(unittest.TestCase):
    def test_known_source_in_wrong_label_folder_is_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "a.mp4"
            path.write_bytes(b"video" * 400)
            rows = [dict(path=str(path), label=1, source="DeepAction:Pexels", origin="a")]
            result = deduplicate(rows, root / "hashes.db", logging.getLogger("test"), root / "errors.jsonl")
            self.assertEqual(result, [])
            self.assertIn("Source label mismatch", (root / "errors.jsonl").read_text())

    def test_known_origin_groups(self):
        self.assertEqual(identify("VideoUFO/abc.1.mp4")[1], identify("VideoUFO/abc.9.mp4")[1])
        self.assertEqual(identify("ms/ms-shared-uuid.mp4")[1], identify("vc2/vc-shared-uuid.mp4")[1])
        self.assertEqual(identify("UCF-101/Run/v_Run_g01_c01.avi")[1], identify("UCF-101/Run/v_Run_g01_c02.avi")[1])
        self.assertEqual(identify("Pexels/12/a.mp4")[1], identify("VideoPoet/12/b.mp4")[1])

    def test_dedup_unions_groups_and_conflicting_labels_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / "a.mp4", root / "b.mp4"
            a.write_bytes(b"same-content" * 200)
            b.write_bytes(a.read_bytes())
            rows = [dict(path=str(a), label=0, source="A", origin="first"),
                    dict(path=str(b), label=0, source="B", origin="second")]
            unique = deduplicate(rows, root / "hashes.db", logging.getLogger("test"), root / "errors.jsonl")
            self.assertEqual(len(unique), 1)
            self.assertEqual(len(unique[0]["aliases"]), 2)
            rows[1]["label"] = 1
            with self.assertRaisesRegex(ValueError, "conflicting"):
                deduplicate(rows, root / "hashes.db", logging.getLogger("test"), root / "errors.jsonl")

    def test_splits_preserve_entire_pool_and_holdout_connected_groups(self):
        rows = [record(i, i % 2) for i in range(200)]
        rows.extend([record("h", 1, "unseen", "shared"), record("other", 0, "camera", "shared")])
        splits = make_splits(rows, holdout="unseen")
        self.assertEqual(sum(map(len, splits.values())), len(rows))
        self.assertEqual({r["id"] for r in splits["ood"]}, {"h", "other"})
        assert_disjoint(splits)
        splits["val"].append(splits["train"][0])
        with self.assertRaisesRegex(ValueError, "Leakage"):
            assert_disjoint(splits)

    def test_rotation_coverage_and_exact_resume(self):
        rows = [record(f"{label}-{i}", label, f"s{i % 2}") for label in (0, 1) for i in range(20)]
        rotation = Rotation(rows)
        first = rotation.take(8)
        saved = copy.deepcopy(rotation.state)
        second = rotation.take(8)
        self.assertEqual(second, Rotation(rows, state=saved).take(8))
        self.assertFalse({r["id"] for r in first} & {r["id"] for r in second})
        seen = {r["id"] for r in first + second}
        for _ in range(3):
            seen.update(r["id"] for r in rotation.take(8))
        self.assertEqual(len(seen), 40)

    def test_rotation_small_source_no_duplicate_within_epoch(self):
        rows = [record(f"{label}-{i}", label, "tiny" if i == 0 else "large")
                for label in (0, 1) for i in range(15)]
        rotation = Rotation(rows)
        for _ in range(3):
            selected = rotation.take(20)
            self.assertEqual(len(selected), len({r["id"] for r in selected}))
            self.assertEqual(sum(r["label"] for r in selected), 10)


class MetricsTests(unittest.TestCase):
    def test_constant_and_perfect_predictions(self):
        constant = metrics([0, 0, 1, 1], [.7]*4)
        self.assertEqual(constant["auroc"], .5)
        self.assertEqual(constant["balanced_accuracy"], .5)
        self.assertEqual(constant["average_precision"], .5)
        self.assertEqual(metrics([0, 0, 1, 1], [.1, .2, .8, .9])["auroc"], 1)

    def test_one_class_does_not_invent_balanced_accuracy(self):
        result = metrics([1, 1], [.8, .9])
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["balanced_accuracy"])

    def test_calibration_ties_respect_false_positive_budget(self):
        threshold = select_threshold([0]*20 + [1]*4, [.5]*20 + [.8]*4)
        result = metrics([0]*20 + [1]*4, [.5]*20 + [.8]*4, threshold)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["tp"], 4)


if __name__ == "__main__":
    unittest.main()
