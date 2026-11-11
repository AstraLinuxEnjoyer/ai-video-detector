import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from training.evaluate_trace import (compression_comparison, eligible, inventory, read_annotations,
                                     score_video, summary, timeline_measure)


def prediction(label, spatial, motion):
    fused = None if motion is None else .7 * spatial + .3 * motion
    return dict(label=label, source='example', result=dict(scores=dict(
        spatial=spatial, temporal=motion, fused_70_30=fused, demo_alert=float(spatial >= .65))))


class TraceBenchmarkTests(unittest.TestCase):
    def test_inventory_provenance_and_conflicting_exact_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'real/phone').mkdir(parents=True)
            (root / 'fake').mkdir()
            (root / 'real/phone/one.mp4').write_bytes(b'one')
            (root / 'real/phone/two.mp4').write_bytes(b'one')
            rows = inventory(root)
            self.assertEqual(rows[0]['source'], 'phone')
            self.assertEqual(len(eligible(rows, {})[0]), 1)
            selected, excluded = eligible(rows, {rows[0]['sha256']: ['spatial/train']})
            self.assertFalse(selected)
            self.assertEqual(excluded[0]['reason'], 'already_in_model_dataset')
            (root / 'fake/copy.mp4').write_bytes(b'one')
            with self.assertRaisesRegex(ValueError, 'Conflicting labels'):
                eligible(inventory(root), {})

    def test_annotation_validates_paths_and_time_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / 'segments.json'
            rows = [dict(file='fake/clip.mp4', label=1)]
            for data in ({'real/clip.mp4': [[1, 2]]}, {'fake/clip.mp4': [[1, 3], [2, 4]]}):
                annotation.write_text(json.dumps(data), encoding='utf-8')
                with self.assertRaises(ValueError):
                    read_annotations(root, rows)
            annotation.write_text('{"fake/clip.mp4": [[1, 3], [3, 4]]}', encoding='utf-8')
            self.assertEqual(read_annotations(root, rows)['fake/clip.mp4'], [[1, 3], [3, 4]])

    def test_partial_motion_can_alert_but_not_score_in_matched_comparison(self):
        detector = Mock(args=SimpleNamespace(clip_length=9, fps=4))
        detector.spatial_score.return_value = dict(spatial=.1)
        detector.motion_score.side_effect = [.9, RuntimeError('bad clip'), .2]
        with tempfile.TemporaryDirectory() as directory, patch(
                'training.evaluate_trace.probe', return_value=dict(duration=5., fps=25.)), patch(
                'training.evaluate_trace.extract_frames', return_value=[dict(time=0., image='frame.jpg')]):
            result = score_video(Path('test.mp4'), detector, Path(directory) / 'work')
        self.assertEqual(result['temporal_status'], 'clip_decode_failure')
        self.assertIsNone(result['scores']['temporal'])
        self.assertIsNone(result['scores']['fused_70_30'])
        self.assertEqual(result['scores']['demo_alert'], 1.)
        self.assertEqual(result['scores']['policy_alert'], 0.)
        self.assertEqual(result['assessment']['verdict'], 'uncertain')
        self.assertEqual(result['expected_clips'], 3)

    def test_timeline_measures_overlap_and_false_highlights(self):
        result = dict(meta=dict(duration=10), regions=dict(spatial=[dict(start=3, end=6)]))
        truth = timeline_measure(dict(result=result, intervals=[[2, 4], [5, 8]], label=1), 'spatial')
        self.assertEqual(truth['truth_seconds'], 5)
        self.assertEqual(truth['overlap_seconds'], 2)
        self.assertEqual(truth['intervals_half_covered'], 1)
        real = timeline_measure(dict(result=result, intervals=[], label=0), 'spatial')
        self.assertEqual(real['real_highlight_seconds'], 3)
        self.assertEqual(real['real_flagged'], 1)

    def test_same_videos_and_recompression_use_matching_denominators(self):
        rows = [prediction(0, .2, None), prediction(1, .9, .8), prediction(0, .8, .2)]
        report = summary([dict(r, intervals=None) for r in rows])
        self.assertEqual(report['all']['spatial']['n'], 3)
        self.assertEqual(report['same_videos']['spatial']['n'], 2)
        self.assertEqual(report['same_videos']['spatial']['fp'], 1)
        self.assertEqual(report['same_videos']['fused_70_30']['fp'], 0)
        self.assertNotIn('auroc', report['all']['demo_alert'])
        compressed = [prediction(0, .9, .8), prediction(1, .1, .2), dict(label=0, error='decode')]
        pairs = compression_comparison(rows, compressed)
        self.assertEqual(pairs['spatial']['n'], 2)
        self.assertEqual(pairs['spatial']['changed_decisions'], 2)
        self.assertEqual(pairs['temporal']['n'], 1)
