import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image, ImageOps

from training.spatial import predict_images as photos


class ConstantModel(torch.nn.Module):
    def forward(self, tensor):
        assert not torch.is_grad_enabled()
        return torch.full((len(tensor), 1), 2., device=tensor.device)


class PhotoTests(unittest.TestCase):
    def test_supported_images_and_no_folder_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in ("real", "fake"):
                (root / label).mkdir()
                Image.new("RGB", (60, 40), "red").save(root / label / "sample.PNG")
            (root / "ignored.txt").write_text("not a photo")
            paths = photos.collect_images(root)
            self.assertEqual(len(paths), 2)
            results = [photos.classify(ConstantModel().eval(), p, 32, torch.device("cpu")) for p in paths]
            self.assertEqual([r["prediction"] for r in results], ["FAKE", "FAKE"])
            self.assertEqual(results[0]["fake_score"], results[1]["fake_score"])

    def test_exif_orientation_matches_already_rotated_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = Image.new("RGB", (40, 60), "blue")
            original.paste("red", (0, 0, 20, 30))
            exif = original.getexif()
            exif[274] = 6
            original.save(root / "exif.png", exif=exif)
            with Image.open(root / "exif.png") as image:
                ImageOps.exif_transpose(image).save(root / "rotated.png")
            torch.testing.assert_close(photos.image_tensor(root / "exif.png", 32),
                                       photos.image_tensor(root / "rotated.png", 32))

    def test_transparency_composites_white(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGBA", (30, 40), (0, 0, 0, 0)).save(root / "alpha.png")
            Image.new("RGB", (30, 40), "white").save(root / "white.png")
            torch.testing.assert_close(photos.image_tensor(root / "alpha.png", 32),
                                       photos.image_tensor(root / "white.png", 32))

    def test_best_snapshot_loads_epoch_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save({}, root / "best.pt")
            (root / "best.json").write_text(json.dumps(dict(epoch=7)))
            weights, best, sha = photos.best_snapshot(root)
            self.assertEqual(weights, {})
            self.assertEqual(best["epoch"], 7)
            self.assertEqual(len(sha), 64)

    def test_empty_folder_does_not_load_weights(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(photos, "best_snapshot") as read:
            with contextlib.redirect_stdout(io.StringIO()):
                photos.main(["--input", tmp])
            read.assert_not_called()

    def test_corrupt_photo_does_not_hide_other_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "photos"
            inputs.mkdir()
            Image.new("RGB", (40, 60), "red").save(inputs / "good.jpg")
            (inputs / "bad.jpg").write_bytes(b"broken")
            meta = dict(epoch=7, image_size=32, threshold=.342)
            with patch.object(photos, "ROOT", root), \
                 patch.object(photos, "best_snapshot", return_value=({}, meta, "test-hash")), \
                 patch.object(photos, "build_model", return_value=ConstantModel()), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as failed:
                    photos.main(["--input", str(inputs), "--run-dir", str(root)])
            self.assertEqual(failed.exception.code, 1)
            report = json.loads(next((root / "reports").glob("photo_check_*.json")).read_text())
            self.assertEqual(len(report["predictions"]), 1)
            self.assertEqual(len(report["errors"]), 1)
            self.assertEqual(report["threshold"], .5)
            self.assertEqual(report["video_validation_threshold"], .342)


if __name__ == "__main__":
    unittest.main()
