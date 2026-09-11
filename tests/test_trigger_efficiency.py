from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from forwardfm_full.data import TRIGGER_FEATURES, denominator_sql, feature_matrix
from forwardfm_full.efficiency import TriggerEfficiencyNet
from forwardfm_full.evaluation import (
    calibration_rows,
    efficiency_closure_2d_rows,
    efficiency_closure_rows,
    expected_calibration_error,
)
from forwardfm_full.labels import (
    broadcast_event_trigger_label,
    reconstruction_exists,
    response_quality_pass,
    trigger_electron_pass,
)


class TriggerLabelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "mcindex": [0, 0, 0],
                "has_valid_trigger_electron": [False, True, True],
                "trigger_mcindex": [np.nan, 0, 0],
                "reconstructed": [False, True, True],
                "matched_pindex": [np.nan, 2, 3],
                "match_reciprocal": [False, True, False],
                "delta_p": [np.nan, 0.01, 0.02],
                "delta_theta": [np.nan, 0.001, 0.002],
                "delta_phi": [np.nan, 0.003, 0.004],
            }
        )

    def test_labels_are_distinct_and_ordered(self) -> None:
        np.testing.assert_array_equal(
            reconstruction_exists(self.frame), [False, True, True]
        )
        np.testing.assert_array_equal(
            response_quality_pass(self.frame), [False, True, False]
        )
        np.testing.assert_array_equal(
            trigger_electron_pass(self.frame), [False, True, True]
        )

    def test_positive_trigger_requires_truth_association(self) -> None:
        broken = self.frame.copy()
        broken.loc[1, "trigger_mcindex"] = 7
        with self.assertRaises(AssertionError):
            trigger_electron_pass(broken)

    def test_event_trigger_label_broadcasts_to_every_particle(self) -> None:
        electrons = pd.DataFrame(
            {
                "source_file_id": [0, 0],
                "event_id": [10, 11],
                "mcindex": [0, 0],
                "has_valid_trigger_electron": [True, False],
                "trigger_mcindex": [0, np.nan],
            }
        )
        particles = pd.DataFrame(
            {
                "source_file_id": [0, 0, 0, 0],
                "event_id": [10, 10, 11, 11],
            }
        )
        np.testing.assert_array_equal(
            broadcast_event_trigger_label(particles, electrons),
            [True, True, False, False],
        )

    def test_event_trigger_broadcast_rejects_missing_electron(self) -> None:
        electrons = pd.DataFrame(
            {
                "source_file_id": [0],
                "event_id": [10],
                "mcindex": [0],
                "has_valid_trigger_electron": [True],
                "trigger_mcindex": [0],
            }
        )
        particles = pd.DataFrame(
            {"source_file_id": [0, 0], "event_id": [10, 11]}
        )
        with self.assertRaises(AssertionError):
            broadcast_event_trigger_label(particles, electrons)


class TriggerFeatureTests(unittest.TestCase):
    def test_denominator_keeps_failures(self) -> None:
        self.assertEqual(denominator_sql(), "gen_pid = 11")
        self.assertNotIn("has_valid_trigger_electron", denominator_sql())

    def test_five_truth_features_and_periodic_phi(self) -> None:
        frame = pd.DataFrame(
            {
                "gen_p": [1.0, 2.0],
                "gen_theta": [0.1, 0.2],
                "gen_phi": [-np.pi, np.pi],
                "gen_vz": [-3.0, -2.0],
            }
        )
        values = feature_matrix(frame)
        self.assertEqual(values.shape, (2, len(TRIGGER_FEATURES)))
        np.testing.assert_allclose(values[:, 2], 0.0, atol=1e-6)
        np.testing.assert_allclose(values[:, 3], -1.0, atol=1e-6)


class TriggerModelTests(unittest.TestCase):
    def test_scalar_logit_and_gradient(self) -> None:
        model = TriggerEfficiencyNet(5, hidden_width=16, hidden_layers=2, dropout=0.0)
        values = torch.randn(11, 5)
        logits = model(values)
        self.assertEqual(tuple(logits.shape), (11,))
        logits.square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))


class TriggerClosureTests(unittest.TestCase):
    def test_one_dimensional_closure_uses_mean_probability(self) -> None:
        rows = efficiency_closure_rows(
            np.asarray([0.1, 0.8, 1.1, 1.8]),
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

    def test_two_dimensional_bins_are_not_flattened(self) -> None:
        rows = efficiency_closure_2d_rows(
            np.asarray([0.2, 0.8, 1.2, 1.8]),
            np.asarray([2.0, 8.0, 2.0, 8.0]),
            np.asarray([0.0, 1.0, 1.0, 0.0]),
            np.asarray([0.1, 0.9, 0.8, 0.2]),
            np.asarray([0.0, 1.0, 2.0]),
            np.asarray([0.0, 5.0, 10.0]),
            1,
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual({(row["p_bin"], row["theta_bin"]) for row in rows}, {
            (0, 0), (0, 1), (1, 0), (1, 1)
        })

    def test_perfect_reliability_has_zero_ece(self) -> None:
        rows = calibration_rows(
            np.asarray([0.0, 0.0, 1.0, 1.0]),
            np.asarray([0.0, 0.0, 1.0, 1.0]),
            2,
        )
        self.assertAlmostEqual(expected_calibration_error(rows), 0.0)


if __name__ == "__main__":
    unittest.main()
