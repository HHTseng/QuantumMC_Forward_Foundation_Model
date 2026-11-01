"""Tests of probability closure and physical coordinate inversion."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from forwardfm_step1 import data_asinh_dp as data
from forwardfm_step1.model_split_dp import ConditionalSplitMDN
from forwardfm_transfer.response import (
    inverse_sample_coordinates, predict_response, response_tables, robust_summary,
)


class TransferResponseTests(unittest.TestCase):
    def make_frame(self):
        return pd.DataFrame({
            "gen_pid": [-211, -211, 211, 211, 2212, 2212],
            "gen_p": [1.2]*6, "gen_theta": [.2]*6, "gen_phi": [0.]*6,
            "rec_pid": [-211, 321, 211, 2212, 2212, 211],
            "delta_p": [-.01, .01]*3, "delta_theta": [.001]*6,
            "delta_phi": [.002]*6, "delta_beta": [.003]*6,
            "corrected_rec_p": [1.2]*6,
        })

    def test_pid_response_uses_mean_softmax_and_preserves_other(self):
        frame = self.make_frame()
        # Every argmax is pi+, but P(pi+)=.51, so its correct-ID closure
        # differs from observed .50 by 1pp, rather than the 50pp top-1 error.
        probability = np.tile([.2, .51, .2, .09], (6, 1))
        sampled = frame.loc[:, data.TARGET_COLUMNS].to_numpy(float)
        tables, raw, corrected = response_tables(frame, sampled, probability,
            [-211, 211, 2212], min_bin_count=1)
        pip = next(r for r in tables["pid_closure"] if r["generated_pid"] == 211)
        self.assertAlmostEqual(pip["absolute_error_pp"], 1.)
        other = next(r for r in tables["pid_channels"]
                     if r["generated_pid"] == -211 and r["reconstructed_pid"] == "OTHER")
        self.assertEqual(other["observed_fraction"], .5)
        np.testing.assert_array_equal(corrected[:, 1:], raw[:, 1:])
        np.testing.assert_array_equal(corrected[:, 0], np.zeros(6))
        self.assertEqual({r["n"] for r in tables["response_summary"]}, {2})

    def test_float64_inverse_preserves_extreme_draw_and_wrap(self):
        scaler = data.Standardizer(np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32))
        standardized = np.array([[100., .01, 3*np.pi, .03], [0., 0., 0., 0.]])
        physical = inverse_sample_coordinates(standardized, scaler)
        self.assertTrue(np.isfinite(physical).all())
        self.assertGreater(physical[0, 0], 1e40)
        self.assertAlmostEqual(physical[0, 0], .020*np.sinh(100))
        self.assertAlmostEqual(physical[0, 2], -np.pi)
        np.testing.assert_array_equal(physical[:, [1, 3]], standardized[:, [1, 3]])

    def test_nonfinite_mass_is_counted_without_contaminating_quantiles(self):
        result = robust_summary([0., 1., 2., np.inf, np.nan])
        self.assertEqual(result["finite_fraction"], .6)
        self.assertEqual(result["q50"], 1.)
        self.assertEqual(robust_summary([np.inf])["sigma68"], None)

    def test_checkpoint_coordinates_are_used_and_cpu_draws_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            frame = self.make_frame()
            torch.manual_seed(5)
            model = ConditionalSplitMDN(5, 3, 4, hidden_width=8, hidden_layers=1,
                                        mixture_components=2, dropout=0)
            checkpoint = {
                "model_type": "split_delta_p_asinh", "target_transform": data.TARGET_TRANSFORM,
                "species_pids": list(data.SPECIES), "target_names": list(data.TARGET_COLUMNS),
                "particle_masses_gev": {str(k): v for k, v in data.GEN_MASS_GEV.items()},
                "architecture": model.architecture_dict(), "model_state": model.state_dict(),
                "feature_names": list(data.CONTINUOUS_FEATURES),
                "feature_scaler": {"mean": [0.]*5, "scale": [2.]*5},
                "target_scaler": {"mean": [0., .1, .2, .3], "scale": [.1, .01, .01, .01]},
                "rec_pid_vocabulary": [-211, 211, 2212],
            }
            path = Path(temp)/"model.pt"
            torch.save(checkpoint, path)
            first = predict_response(path, frame, torch.device("cpu"), sample_seed=17)
            second = predict_response(path, frame, torch.device("cpu"), sample_seed=17)
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            np.testing.assert_allclose(first[1].sum(1), 1., atol=1e-7)
            self.assertGreater(float(first[0][:, 3].mean()), .25)


if __name__ == "__main__":
    unittest.main()
