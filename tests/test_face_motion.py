import unittest
from unittest.mock import patch

import numpy as np

from training.temporal_v2.face_region import FaceUnavailable, complete_pairs, crop_face_clip


class FaceMotionTests(unittest.TestCase):
    def test_fixed_native_crop_does_not_manufacture_motion_from_box_jitter(self):
        frame = np.random.default_rng(42).integers(0, 255, (256, 360, 3), dtype=np.uint8)
        pixels = np.repeat(frame[None], 9, axis=0)
        with patch('training.temporal_v2.face_region.locate',
                   side_effect=[(99, 80, 180, 161), (100, 80, 181, 161), (101, 81, 181, 162)]):
            cropped = crop_face_clip(pixels, 128, 'unused')
        self.assertEqual(cropped.shape, (9, 128, 128, 3))
        np.testing.assert_array_equal(cropped, np.repeat(cropped[:1], 9, axis=0))

    def test_lost_or_inconsistent_face_abstains(self):
        pixels = np.zeros((9, 256, 360, 3), dtype=np.uint8)
        box = (99, 80, 180, 161)
        for detections in ([None, box, None], [(230, 80, 311, 161), box, box]):
            with patch('training.temporal_v2.face_region.locate', side_effect=detections):
                with self.assertRaises(FaceUnavailable):
                    crop_face_clip(pixels, 128, 'unused')

    def test_abstention_never_leaves_unpaired_label(self):
        incomplete = dict(group='a', label=1)
        pair = [dict(group='b', label=label) for label in (0, 1)]
        self.assertEqual(complete_pairs([incomplete, *pair]), pair)
