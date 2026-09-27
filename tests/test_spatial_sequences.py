import copy
import logging
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from training.spatial.data import collect_paths, deduplicate, Rotation
from training.spatial.lineage import pin_parent_splits
from training.spatial.media import FrameCache, prepare_frames, SourceChangedError


class SequenceTests(unittest.TestCase):
    def make_scene(self, root, resolution="Full-Resolution", scene="walking"):
        path = root / "data/real/DAVIS/JPEGImages" / resolution / scene
        path.mkdir(parents=True)
        for index in range(12):
            Image.new("RGB", (80, 48), (index * 20, 10, 30)).save(path / f"{index:05}.jpg")
        return path

    def audit(self, root):
        return deduplicate(collect_paths(root), root / "hashes.db", logging.getLogger("test"), root / "errors.jsonl")

    def test_only_rgb_scenes_not_masks_or_lower_resolution_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_scene(root)
            self.make_scene(root, "480p")
            masks = root / "data/real/DAVIS/Annotations/Full-Resolution/walking"
            masks.mkdir(parents=True)
            Image.new("L", (80, 48)).save(masks / "00000.png")
            records = self.audit(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["source"], "DAVIS")
            self.assertEqual(records[0]["label"], 0)
            self.assertEqual(records[0]["group"], "davis/walking")
            self.assertEqual(len(records[0]["sequence_frames"]), 12)

    def test_sequence_sampling_repeatability_and_changed_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = self.make_scene(root)
            records = self.audit(root)
            cache = FrameCache(root, 1, .001)
            log = logging.getLogger("test")
            val, errors = prepare_frames(records, cache, "val", 4, 42, log, root, 1)
            self.assertFalse(errors)
            self.assertEqual(val[0]["source_frame_indices"], [1, 4, 7, 10])
            self.assertIsNone(val[0]["metadata"]["fps"])
            again, _ = prepare_frames(records, cache, "val", 4, 42, log, root, 1)
            self.assertEqual(val, again)
            a, _ = prepare_frames(records, cache, "train_0", 4, 42, log, root, 1)
            b, _ = prepare_frames(records, cache, "train_1", 4, 43, log, root, 1)
            self.assertNotEqual(a[0]["source_frame_indices"], b[0]["source_frame_indices"])
            Image.new("RGB", (81, 49)).save(scene / "00000.jpg")
            with self.assertRaises(SourceChangedError):
                prepare_frames(records, cache, "val", 4, 42, log, root, 1)

    def test_copied_sequences_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_scene(root, scene="a")
            self.make_scene(root, scene="b")
            records = self.audit(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(len(records[0]["aliases"]), 2)

    def test_parent_pinning_and_conflicting_groups(self):
        def row(i, group):
            return dict(id=i, label=0, group=group, origin=group, aliases=[dict(path=i)])
        a, b = row("a", "old-a"), row("b", "old-b")
        parent = dict(train=[a], val=[b], test=[], ood=[])
        candidate = dict(train=[b], val=[a], test=[row("new", "new")], ood=[])
        pinned = pin_parent_splits(candidate, parent)
        self.assertEqual(pinned["train"], [a])
        self.assertEqual(pinned["val"], [b])
        conflict = copy.deepcopy(candidate)
        conflict["train"][0]["group"] = "old-a"
        with self.assertRaisesRegex(ValueError, "different parent splits"):
            pin_parent_splits(conflict, parent)

    def test_parent_seen_order_and_resume(self):
        rows = [dict(id=f"{label}-{i}", label=label, source="s") for label in (0, 1) for i in range(8)]
        old = {f"{label}-{i}" for label in (0, 1) for i in range(4)}
        rotation = Rotation(rows, previously_seen=old)
        chosen = rotation.take(8)
        self.assertFalse(old & {r["id"] for r in chosen})
        state = copy.deepcopy(rotation.state)
        self.assertEqual(rotation.take(8), Rotation(rows, state=state, previously_seen=old).take(8))


if __name__ == "__main__":
    unittest.main()
