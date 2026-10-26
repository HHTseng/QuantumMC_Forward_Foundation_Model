"""Exact small-array checks for hadron probability/kinematic closure."""

import unittest

import numpy as np

from forwardfm_hadron.evaluation import (
    binary_metrics,
    binned_rows,
    closure_summary,
    reliability_rows,
    wilson_interval,
)


class HadronEvaluationTests(unittest.TestCase):
    def test_reliability_zero_one_and_empty_bins(self):
        y = np.array([0, 1, 1])
        p = np.array([0., .5, 1.])
        rows = reliability_rows(y, p, n_bins=20)
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(row["n"] for row in rows), 3)
        self.assertEqual(rows[0]["n"], 1)
        self.assertEqual(rows[10]["n"], 1)
        self.assertEqual(rows[19]["n"], 1)
        self.assertEqual(rows[19]["observed_efficiency"], 1.)
        self.assertIsNone(rows[1]["observed_efficiency"])
        self.assertIsNone(rows[1]["absolute_calibration_gap"])

    def test_known_calibration_and_constant_baseline(self):
        y = np.array([0, 1, 0, 1])
        constant = binary_metrics(y, np.full(4, .5), train_rate=.5, n_bins=20)
        self.assertAlmostEqual(constant["expected_calibration_error"], 0.)
        self.assertAlmostEqual(constant["brier_score"], .25)
        self.assertAlmostEqual(constant["log_loss"], np.log(2))
        self.assertAlmostEqual(constant["brier_skill_vs_constant"], 0.)
        perfect = binary_metrics(y, y.astype(float), train_rate=.5)
        self.assertEqual(perfect["brier_score"], 0.)
        self.assertAlmostEqual(perfect["expected_calibration_error"], 0.)
        self.assertAlmostEqual(perfect["brier_skill_vs_constant"], 1.)

    def test_wilson_intervals_remain_nonzero_at_boundaries(self):
        lo, hi = wilson_interval(0, 100)
        self.assertAlmostEqual(lo, 0.)
        self.assertGreater(hi, 0.)
        lo, hi = wilson_interval(100, 100)
        self.assertLess(lo, 1.)
        self.assertAlmostEqual(hi, 1.)
        lo, hi = wilson_interval(50, 100)
        self.assertLess(lo, .5)
        self.assertGreater(hi, .5)

    def test_fixed_edges_preserve_right_boundary_low_count_and_outside(self):
        values = np.array([-1., 0., .5, 1., 2., 3.])
        y = np.array([0, 0, 1, 0, 1, 0])
        p = np.array([.1, .2, .6, .1, .9, .2])
        rows = binned_rows(values, y, p, np.array([0., 1., 2.]), "gen_p", min_bin_count=3)
        self.assertEqual([row["n"] for row in rows], [2, 2])
        self.assertEqual(sum(row["n"] for row in rows), 4)
        self.assertEqual(sum(row["successes"] for row in rows), 2)
        self.assertTrue(rows[-1]["upper_edge_inclusive"])
        self.assertFalse(rows[0]["eligible"])
        summary = closure_summary(rows, total_rows=6)
        self.assertEqual(summary["out_of_range_rows"], 2)
        self.assertEqual(summary["low_count_rows"], 4)
        self.assertIsNone(summary["weighted_mae"])
        rows = binned_rows(values, y, p, np.array([0., 1., 2.]), "gen_p", min_bin_count=2)
        summary = closure_summary(rows, total_rows=6)
        self.assertEqual(summary["eligible_rows"], 4)
        self.assertAlmostEqual(summary["weighted_mae"], .05)

    def test_empty_bin_retained_and_bad_probabilities_rejected(self):
        rows = binned_rows(np.array([0.]), np.array([0]), np.array([.2]),
                           np.array([0., 1., 2.]), "gen_p", min_bin_count=1)
        self.assertEqual(rows[1]["n"], 0)
        self.assertIsNone(rows[1]["fm_mean_probability"])
        with self.assertRaises(ValueError):
            reliability_rows(np.array([0]), np.array([1.01]))


if __name__ == "__main__":
    unittest.main()
