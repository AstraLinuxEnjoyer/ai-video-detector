import argparse
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.spatial import data, media
from training.spatial.evaluate import external_evaluation


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.log = logging.getLogger("spatial_media_tests")

    def test_overwrite_counts_actual_cache_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = media.FrameCache(Path(tmp), 1, .001)
            target = cache.root / "test.bin"
            cache.put(target, b"123456")
            cache.put(target, b"12")
            self.assertEqual(cache.used, 2)

    def test_changed_signature_regenerates_cached_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "video.mp4"
            source.write_bytes(b"video")
            stat = source.stat()
            record = dict(path=str(source), id="sha", source="test", size=stat.st_size, mtime=stat.st_mtime_ns)
            cache = media.FrameCache(root, 1, .001)
            meta = dict(duration=2, fps=24)
            with patch.object(media, "probe", return_value=meta), \
                 patch.object(media, "decode", side_effect=lambda *a: str(a[1]).encode()) as decoder:
                first, _ = media.prepare_frames([record], cache, "train_0", 2, 42, self.log, root, 1)
                old_bytes = Path(first[0]["frames"][0]).read_bytes()
                media.prepare_frames([record], cache, "train_0", 2, 42, self.log, root, 1)
                self.assertEqual(decoder.call_count, 2)
                changed, _ = media.prepare_frames([record], cache, "train_0", 2, 43, self.log, root, 1)
                self.assertEqual(decoder.call_count, 4)
                self.assertNotEqual(old_bytes, Path(changed[0]["frames"][0]).read_bytes())
            self.assertEqual(cache.used, sum(p.stat().st_size for p in cache.root.rglob("*") if p.is_file()))

    def test_changed_original_is_fatal_not_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "video.mp4"
            source.write_bytes(b"changed")
            record = dict(path=str(source), id="sha", source="test", size=1, mtime=0)
            cache = media.FrameCache(root, 1, .001)
            with self.assertRaises(media.SourceChangedError):
                media.prepare_frames([record], cache, "val", 1, 42, self.log, root, 1)

    def test_empty_optional_external_set_does_not_fail_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = external_evaluation(None, root, root, argparse.Namespace(), None, self.log, set(), .5)
            self.assertEqual(result["status"], "no_data")
            self.assertEqual(json.loads((root / "external_report.json").read_text()), result)

    def test_offline_remote_cache_without_legacy_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached = root / "cache/spatial/videos/faridlab__deepaction_v1/Veo/0/a.mp4"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"video")
            with patch.object(data, "ROOT", root):
                records = data.remote_paths([], root, self.log, offline=True)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["label"], 1)
            self.assertEqual(records[0]["source"], "DeepAction:Veo")

    def test_image_with_mp4_extension_is_rejected_before_splitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "actually_jpeg.mp4"
            path.write_bytes(b"\xff\xd8\xff" + b"0" * 2048)
            records = [dict(path=str(path), label=0, source="unknown", origin="unknown")]
            result = data.deduplicate(records, root / "hashes.sqlite", self.log, root / "errors.jsonl")
            self.assertEqual(result, [])
            self.assertIn("Image content disguised", (root / "errors.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
