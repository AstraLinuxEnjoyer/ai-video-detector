import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.dataset_paths import archive_source_path
from training import import_videos


class DatasetPathTests(unittest.TestCase):
    def test_named_archive(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "GenVideo-100K/extracted/train_OpenSora"
            folder.mkdir(parents=True)
            self.assertEqual(archive_source_path("GenVideo_OpenSora", root), folder.resolve())

    def test_missing_and_unknown_archive(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                archive_source_path("GenVideo_SVD", root)
            with self.assertRaises(ValueError):
                archive_source_path("../../real", root)

    def test_cross_volume_does_not_replace_existing(self):
        with tempfile.TemporaryDirectory() as root:
            source, output = Path(root) / "source.mp4", Path(root) / "output.mp4"
            source.write_bytes(b"new")
            output.write_bytes(b"original")
            with patch.object(import_videos, "same_volume", return_value=False):
                with self.assertRaisesRegex(OSError, "different volumes"):
                    import_videos.import_video(source, output, "hardlink", True, False)
            self.assertEqual(output.read_bytes(), b"original")

    def test_existing_file_is_not_reimported(self):
        with tempfile.TemporaryDirectory() as root:
            source, output = Path(root) / "source.mp4", Path(root) / "output.mp4"
            source.write_bytes(b"new")
            output.write_bytes(b"original")
            with patch.object(import_videos, "same_volume", side_effect=AssertionError("not needed")):
                self.assertEqual(import_videos.import_video(source, output, "hardlink", False, False), "exists")

    def test_copy_is_explicit_and_same_volume_is_detected(self):
        with tempfile.TemporaryDirectory() as root:
            source, output = Path(root) / "source.mp4", Path(root) / "new/output.mp4"
            source.write_bytes(b"video")
            self.assertTrue(import_videos.same_volume(source, output))
            self.assertEqual(import_videos.import_video(source, output, "copy", False, False), "copied")
            self.assertEqual(output.read_bytes(), b"video")

    def test_dry_run_has_no_writes(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root) / "archives"
            source = archive / "GenVideo-100K/extracted/train_OpenSora"
            source.mkdir(parents=True)
            (source / "clip.mp4").write_bytes(b"x" * 2048)
            output = Path(root) / "working/fake"
            with patch.object(import_videos, "LOCAL_FAKE_DIR", output), contextlib.redirect_stdout(io.StringIO()):
                import_videos.main(["--dataset", "GenVideo_OpenSora", "--dataset-root", str(archive),
                                    "--label", "fake", "--source", "GenVideo_OpenSora", "--dry-run"])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
