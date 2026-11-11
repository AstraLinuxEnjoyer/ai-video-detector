import argparse
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from training.artifact_spatial.patches import view_tensor
from training.spatial.media import probe
from training.spectral.features import DIM, full_frame_features, patch_features, pool_features
from training.temporal_v2.flow import tensor_from_flow
from training.temporal_v2.media import decode_clip


class FullFrameTests(unittest.TestCase):
    def test_spatial_view_keeps_both_edges(self):
        pixels = np.zeros((64, 128, 3), dtype=np.uint8)
        pixels[:, :16, 0] = 255
        pixels[:, -16:, 2] = 255
        result = view_tensor(Image.fromarray(pixels), 32, full_frame=True)
        self.assertEqual(result.shape, (1, 3, 32, 32))
        self.assertGreater(float(result[0, 0, 16, 0]), 1)
        self.assertGreater(float(result[0, 2, 16, -1]), 1)

    def test_temporal_flow_does_not_recrop_training_views(self):
        field = np.zeros((2, 128, 128, 2), dtype=np.float16)
        field[:, :, :12, 1] = .2
        field[:, :, -12:, 1] = .4
        for seed in range(5):
            result = tensor_from_flow(field, 112, .5, seed=seed, full_frame=True)
            edges = sorted(float(result[1, 0, 0, i]) for i in (0, -1))
            self.assertAlmostEqual(edges[0], .4, places=3)
            self.assertAlmostEqual(edges[1], .8, places=3)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools unavailable")
    def test_temporal_decoder_retains_borders_that_crop_removes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "borders.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=black:s=256x96:r=8:d=2,drawbox=x=0:y=0:w=24:h=96:color=red:t=fill,"
                            "drawbox=x=232:y=0:w=24:h=96:color=blue:t=fill",
                            "-c:v", "libx264", "-threads", "1", str(path)], capture_output=True, check=True, timeout=20)
            meta = probe(path)
            args = argparse.Namespace(decode_size=96, fps=4, clip_length=4, frame_view="full")
            whole = decode_clip(path, 0, meta, args)
            args.frame_view = "crop"
            cropped = decode_clip(path, 0, meta, args)
            self.assertGreater(int(whole[0, 48, 0, 0]), 150)
            self.assertGreater(int(whole[0, 48, -1, 2]), 150)
            self.assertLess(int(cropped.max()), 20)

    def test_real_fft_matches_legacy_square_band_power(self):
        gray = np.random.default_rng(15).integers(0, 256, (128, 128), dtype=np.uint8)
        full, stats = full_frame_features(gray)
        legacy, old_stats = patch_features(gray)
        np.testing.assert_allclose(full, legacy, atol=1e-5)
        np.testing.assert_allclose(stats, old_stats, atol=1e-5)

    def test_native_rectangular_frames_use_all_widths_and_heights(self):
        gray = np.random.default_rng(16).integers(0, 256, (16, 577), dtype=np.uint8)
        blank = np.zeros_like(gray)
        partial = blank.copy()
        partial[:, -32:] = gray[:, -32:]
        vector = pool_features([full_frame_features(partial)])
        self.assertEqual(len(vector), DIM)
        self.assertTrue(np.isfinite(vector).all())
        self.assertNotEqual(vector, pool_features([full_frame_features(blank)]))
