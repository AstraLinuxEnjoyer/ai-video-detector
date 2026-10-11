import unittest

import torch

from training.temporal_v2.motion_audit import temporal_variant, paired_summary, summarize, metadata_control, VARIANTS, DERIVED


class MotionAuditTests(unittest.TestCase):
    def test_permutations_preserve_frames_and_input(self):
        x = torch.arange(2*3*16*2*2).reshape(2, 3, 16, 2, 2).float()
        old = x.clone()
        identities = [('a', 0), ('b', 1)]
        for variant in ('reverse', 'shuffle_0', 'shuffle_1', 'shuffle_2'):
            result = temporal_variant(x, variant, identities, 42)
            torch.testing.assert_close(result.sort(2).values, x.sort(2).values)
            torch.testing.assert_close(result, temporal_variant(x, variant, identities, 42))
            self.assertFalse(torch.equal(x, result))
        torch.testing.assert_close(x, old)

    def test_shuffle_independent_of_batch_composition(self):
        x = torch.rand(2, 3, 16, 2, 2)
        a = temporal_variant(x, 'shuffle_0', [('a', 0), ('b', 1)], 42)
        b = temporal_variant(x[1:], 'shuffle_0', [('b', 1)], 42)
        torch.testing.assert_close(a[1:], b)

    def test_freeze_has_zero_temporal_change(self):
        x = torch.rand(2, 3, 16, 2, 2)
        for variant, anchor in (('freeze_first', 0), ('freeze_middle', 8), ('freeze_last', 15)):
            result = temporal_variant(x, variant, [('a', 0), ('b', 0)], 42)
            for frame in range(16):
                torch.testing.assert_close(result[:, :, frame], x[:, :, anchor])

    def test_paired_summary_counts_correctness_not_score_difference(self):
        rows = [dict(id='a', group='a', source='camera', label=0, native_fps=30,
                     scores={name: .1 if name == 'clean' else .8 for name in (*VARIANTS, *DERIVED)},
                     logits={name: [1., 2.] for name in VARIANTS}),
                dict(id='b', group='b', source='generator', label=1, native_fps=8,
                     scores={name: .2 if name == 'clean' else .8 for name in (*VARIANTS, *DERIVED)},
                     logits={name: [1., 2.] for name in VARIANTS})]
        result = paired_summary(rows, 'reverse')
        self.assertEqual(result['changed_decisions'], 2)
        self.assertEqual(result['clean_correct_variant_wrong'], 1)
        self.assertEqual(result['clean_wrong_variant_correct'], 1)
        self.assertEqual(result['clean_minus_variant_accuracy'], 0.)
        reports = summarize(rows, rows, [], 42)
        self.assertEqual(reports['clean']['n'], 2)
        self.assertEqual(reports['clean']['accuracy'], .5)
        self.assertEqual(reports['reverse']['false_positive_rate'], 1.)
        self.assertEqual(reports['reverse']['coverage']['coverage'], 1.)
        self.assertEqual(reports['clean']['by_native_fps']['below_16']['n'], 1)
        control = metadata_control(rows)
        self.assertEqual(control['accuracy'], 1.)
        self.assertEqual(control['false_positive_rate'], 0.)


if __name__ == '__main__':
    unittest.main()
