import unittest

from demo.assessment import assess


def frame(t, score):
    return dict(time=t, spatial=score)


def clip(t, score):
    return dict(start=t, end=t + 2.25, temporal=score)


class AssessmentTests(unittest.TestCase):
    def test_one_spike_is_uncertain_not_a_positive_interval(self):
        result = assess([frame(0, .1), frame(2, .1), frame(4, .1)],
                        [clip(0, .1), clip(1.5, .7), clip(3, .1)], 6, 3)
        self.assertEqual(result['verdict'], 'uncertain')
        self.assertIn('isolated_high_scores', result['reasons'])
        self.assertEqual(result['segments'], [])
        self.assertEqual(len(result['candidate_segments']), 1)
        self.assertAlmostEqual(result['combined'], .16)

    def test_sustained_signals_require_review_and_cross_branch_ones_raise_alert(self):
        spatial = assess([frame(0, .8), frame(2.05, .65), frame(4.1, .1)], [], 6, 0)
        self.assertEqual(spatial['verdict'], 'uncertain')
        self.assertEqual(spatial['review_segments'][0]['evidence'], ['spatial_persistence'])
        motion = assess([frame(0, .1)], [clip(0, .7), clip(1.5, .8)], 4, 2)
        self.assertEqual(motion['review_segments'], [dict(start=0, end=3.75, evidence=['motion_persistence'])])
        self.assertEqual(motion['verdict'], 'uncertain')
        self.assertIn('single_branch_persistence', motion['reasons'])
        cross = assess([frame(2, .8)], [clip(1.5, .7)], 4, 1)
        self.assertEqual(cross['segments'], [dict(start=2, end=3.75, evidence=['cross_branch'])])
        self.assertEqual(cross['policy'], 'trace-evidence-v3')
        self.assertEqual(cross['verdict'], 'suspected')

    def test_review_intervals_are_not_duplicated_over_confirmed_intervals(self):
        result = assess([frame(0, .8), frame(2, .8)], [clip(0, .7), clip(1.5, .8)], 4, 2)
        self.assertEqual(result['verdict'], 'suspected')
        self.assertTrue(result['segments'])
        self.assertEqual(result['review_segments'], [])

    def test_missing_or_nearly_duplicated_motion_windows_do_not_confirm(self):
        for times in ((0, .2), (0, 3)):
            result = assess([frame(0, .1)], [clip(t, .8) for t in times], 6, 3)
            self.assertEqual(result['coverage']['motion'], 'partial')
            self.assertEqual(result['verdict'], 'uncertain')
        result = assess([frame(1, .8)], [clip(1.1, .8)], 4, 1)
        self.assertEqual(result['segments'], [])  # A frame before the clip does not corroborate it.

    def test_low_scores_need_complete_motion(self):
        for clips, expected, status, verdict in [([], 0, 'unsupported', 'uncertain'),
                ([], 2, 'failed', 'uncertain'), ([clip(0, .1)], 2, 'partial', 'uncertain'),
                ([clip(0, .1)], 1, 'complete', 'no_signals')]:
            result = assess([frame(0, .1)], clips, 4, expected)
            self.assertEqual(result['coverage']['motion'], status)
            self.assertEqual(result['verdict'], verdict)

    def test_elevated_mean_and_disagreement_are_not_suspected_by_themselves(self):
        result = assess([frame(0, .6)], [clip(0, .1)], 3, 1)
        self.assertEqual(result['verdict'], 'uncertain')
        self.assertEqual(result['reasons'], ['elevated_mean_score', 'branch_disagreement'])
        self.assertFalse(result['segments'])

    def test_rejects_nonfinite_scores_and_wrong_coverage(self):
        for frames, clips, expected in [([frame(0, float('nan'))], [], 0),
                                         ([frame(0, .2)], [clip(0, .1)], 0), ([], [], 0)]:
            with self.assertRaises(ValueError):
                assess(frames, clips, 4, expected)
