from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from forwardfm_electron.data import OUTCOME_LABELS, denominator_sql, encode_outcomes, feature_matrix
from forwardfm_electron.evaluation import (
    calibration_rows,
    efficiency_closure_rows,
    expected_calibration_error,
)
from forwardfm_electron.model import ElectronEfficiencyNet


class ElectronDataTests(unittest.TestCase):
    def test_denominator_is_truth_pid_not_success_flag(self) -> None:
        self.assertEqual(denominator_sql(), "gen_pid = 11")
        self.assertNotIn("is_generated_trigger_electron", denominator_sql())

    def test_feature_map_and_outcome_encoding(self) -> None:
        frame = pd.DataFrame(
            {
                "gen_p": [1.0, 2.0, 3.0],
                "gen_theta": [0.1, 0.2, 0.3],
                "gen_phi": [-np.pi, 0.0, np.pi],
                "gen_vx": [0.0, 0.1, 0.2],
                "gen_vy": [0.0, -0.1, -0.2],
                "gen_vz": [-2.0, -1.0, 0.0],
                "reconstructed": [False, True, True],
                "matched_pindex": [np.nan, 0.0, 1.0],
                "rec_detector_region": [None, "FD", "CD"],
            }
        )
        features = feature_matrix(frame)
        self.assertEqual(features.shape, (3, 7))
        np.testing.assert_allclose(features[0, 2:4], [0.0, -1.0], atol=1e-6)
        labels = encode_outcomes(frame)
        self.assertEqual(
            labels.tolist(),
            [
                OUTCOME_LABELS.index("unreconstructed"),
                OUTCOME_LABELS.index("FD"),
                OUTCOME_LABELS.index("CD"),
            ],
        )


class ElectronModelTests(unittest.TestCase):
    def test_shapes_and_gradients(self) -> None:
        model = ElectronEfficiencyNet(7, len(OUTCOME_LABELS), 16, 2, 0.0)
        values = torch.randn(13, 7)
        output = model(values)
        self.assertEqual(tuple(output.trigger_logit.shape), (13,))
        self.assertEqual(tuple(output.outcome_logits.shape), (13, 5))
        loss = output.trigger_logit.square().mean() + output.outcome_logits.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


class EfficiencyClosureTests(unittest.TestCase):
    def test_bins_use_observed_rate_and_mean_probability(self) -> None:
        rows = efficiency_closure_rows(
            np.asarray([0.0, 0.5, 1.0, 2.0]),
            np.asarray([0.0, 1.0, 1.0, 0.0]),
            np.asarray([0.2, 0.8, 0.9, 0.3]),
            np.asarray([0.0, 1.0, 2.0]),
            "gen_p",
            1,
        )
        self.assertEqual([row["n"] for row in rows], [2, 2])
        self.assertAlmostEqual(rows[0]["observed_efficiency"], 0.5)
        self.assertAlmostEqual(rows[0]["fm_mean_probability"], 0.5)
        self.assertAlmostEqual(rows[1]["fm_mean_probability"], 0.6)

    def test_perfect_calibration_has_zero_ece(self) -> None:
        rows = calibration_rows(
            np.asarray([0.0, 0.0, 1.0, 1.0]),
            np.asarray([0.0, 0.0, 1.0, 1.0]),
            n_bins=2,
        )
        self.assertAlmostEqual(expected_calibration_error(rows), 0.0)


if __name__ == "__main__":
    unittest.main()
