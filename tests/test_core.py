from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from forwardfm_step1.data import (
    BASE_TARGET_COLUMNS,
    BETA_TARGET_COLUMN,
    PreparedSplit,
    Standardizer,
    _target_matrix,
    assert_event_disjoint,
    data_order_seed,
    data_split_seed,
    generated_beta,
    response_target_names,
    selection_sql,
    split_predicate,
)
from forwardfm_step1.evaluation import (
    beta_closure_rows,
    conditional_pid_response_rows,
    integrated_correct_pid_response,
)
from forwardfm_step1.model import ConditionalMDN, mixture_nll, sample_standardized_residuals


class StandardizerTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        values = np.asarray([[1.0, 2.0], [3.0, 8.0], [5.0, 4.0]], dtype=np.float32)
        scaler = Standardizer.fit(values)
        restored = scaler.inverse(scaler.transform(values))
        np.testing.assert_allclose(restored, values, rtol=1e-6, atol=1e-6)


class BetaTargetTests(unittest.TestCase):
    def test_relativistic_generated_beta_uses_species_mass(self) -> None:
        momentum = np.asarray([0.0, 1.0, 1.0])
        pid = np.asarray([211, 211, 2212])
        beta = generated_beta(momentum, pid)
        self.assertEqual(beta[0], 0.0)
        self.assertAlmostEqual(beta[1], 1.0 / np.sqrt(1.0 + 0.13957039**2))
        self.assertAlmostEqual(beta[2], 1.0 / np.sqrt(1.0 + 0.93827208816**2))
        self.assertGreater(beta[1], beta[2])

    def test_beta_target_round_trip(self) -> None:
        frame = pd.DataFrame(
            {
                "gen_p": [0.8, 1.4],
                "gen_pid": [211, 2212],
                "rec_beta": [0.97, 0.81],
                "delta_p": [0.01, -0.02],
                "delta_theta": [0.001, -0.002],
                "delta_phi": [0.003, -0.004],
            }
        )
        target_names = (*BASE_TARGET_COLUMNS, BETA_TARGET_COLUMN)
        matrix = _target_matrix(frame, target_names)
        restored_beta = generated_beta(
            frame.gen_p.to_numpy(), frame.gen_pid.to_numpy()
        ) + matrix[:, 3]
        np.testing.assert_allclose(restored_beta, frame.rec_beta, atol=1e-7)

    def test_beta_is_opt_in_and_adds_explicit_validity_selection(self) -> None:
        config = {
            "data": {
                "selection": {
                    "theta_max_deg": 33.0,
                    "vz_min_cm": -5.5,
                    "vz_max_cm": -0.5,
                    "require_reciprocal_match": True,
                    "reject_rec_pid_zero": True,
                    "reject_beta_sentinel": True,
                    "max_abs_delta_p_gev": 10.0,
                },
                "beta_response": {
                    "enabled": True,
                    "target": "delta_beta",
                    "rec_beta_min_exclusive": 0.0,
                    "rec_beta_max_inclusive": 1.2,
                },
            }
        }
        self.assertEqual(
            response_target_names(config),
            (*BASE_TARGET_COLUMNS, BETA_TARGET_COLUMN),
        )
        selected = selection_sql(config)
        self.assertIn("isfinite(rec_beta)", selected)
        self.assertIn("rec_beta > 0.0", selected)
        self.assertIn("rec_beta <= 1.2", selected)

    def test_unknown_mass_hypothesis_fails(self) -> None:
        with self.assertRaises(ValueError):
            generated_beta(np.asarray([1.0]), np.asarray([321]))

    def test_beta_validity_selection_can_be_used_without_beta_target(self) -> None:
        config = {
            "project": {"seed": 99},
            "data": {
                "selection": {
                    "theta_max_deg": 33.0,
                    "vz_min_cm": -5.5,
                    "vz_max_cm": -0.5,
                    "require_reciprocal_match": True,
                    "reject_rec_pid_zero": True,
                    "reject_beta_sentinel": True,
                    "max_abs_delta_p_gev": 10.0,
                },
                "beta_response": {
                    "enabled": False,
                    "apply_validity_selection": True,
                    "rec_beta_min_exclusive": 0.0,
                    "rec_beta_max_inclusive": 1.2,
                },
            },
        }
        self.assertEqual(response_target_names(config), BASE_TARGET_COLUMNS)
        selected = selection_sql(config)
        self.assertIn("isfinite(rec_beta)", selected)
        self.assertIn("rec_beta > 0.0", selected)
        self.assertIn("rec_beta <= 1.2", selected)


class SplitTests(unittest.TestCase):
    @staticmethod
    def split(name: str, keys: list[str]) -> PreparedSplit:
        n = len(keys)
        return PreparedSplit(
            name=name,
            event_keys=np.asarray(keys, dtype=object),
            continuous=np.zeros((n, 4), dtype=np.float32),
            species_index=np.zeros(n, dtype=np.int64),
            targets=np.zeros((n, 3), dtype=np.float32),
            rec_pid_index=np.zeros(n, dtype=np.int64),
            raw_species=np.full(n, 211, dtype=np.int64),
        )

    def test_disjoint_splits_pass(self) -> None:
        assert_event_disjoint(
            {
                "train": self.split("train", ["1:1"]),
                "validation": self.split("validation", ["1:2"]),
                "test": self.split("test", ["1:3"]),
            }
        )

    def test_overlap_fails(self) -> None:
        with self.assertRaises(AssertionError):
            assert_event_disjoint(
                {
                    "train": self.split("train", ["1:1"]),
                    "validation": self.split("validation", ["1:1"]),
                    "test": self.split("test", ["1:3"]),
                }
            )

    def test_explicit_data_seeds_are_independent_of_model_seed(self) -> None:
        config = {
            "project": {"seed": 101},
            "data": {
                "split_modulus": 10000,
                "train_boundary": 8000,
                "validation_boundary": 9000,
                "split_seed": 17,
                "order_seed": 23,
            },
        }
        self.assertEqual(data_split_seed(config), 17)
        self.assertEqual(data_order_seed(config), 23)
        self.assertIn("event_id, 17", split_predicate("test", config))


class ModelTests(unittest.TestCase):
    def test_shapes_loss_and_sample(self) -> None:
        torch.manual_seed(7)
        model = ConditionalMDN(
            n_continuous=4,
            n_species=3,
            n_rec_pid_classes=6,
            hidden_width=16,
            hidden_layers=2,
            pid_embedding_dim=4,
            mixture_components=3,
            target_dim=3,
            dropout=0.0,
        )
        features = torch.randn(11, 4)
        species = torch.randint(0, 3, (11,))
        targets = torch.randn(11, 3)
        output = model(features, species)
        self.assertEqual(tuple(output.mixture_logits.shape), (11, 3))
        self.assertEqual(tuple(output.means.shape), (11, 3, 3))
        self.assertEqual(tuple(output.pid_logits.shape), (11, 6))
        loss = mixture_nll(output, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        sample = sample_standardized_residuals(output)
        self.assertEqual(tuple(sample.shape), (11, 3))
        self.assertTrue(torch.isfinite(sample).all())

    def test_four_response_model_shapes(self) -> None:
        model = ConditionalMDN(
            n_continuous=4,
            n_species=3,
            n_rec_pid_classes=6,
            hidden_width=12,
            hidden_layers=2,
            pid_embedding_dim=4,
            mixture_components=3,
            target_dim=4,
            dropout=0.0,
        )
        output = model(torch.randn(7, 4), torch.randint(0, 3, (7,)))
        self.assertEqual(tuple(output.means.shape), (7, 3, 4))
        sample = sample_standardized_residuals(output)
        self.assertEqual(tuple(sample.shape), (7, 4))

    def test_seeded_reset_pairs_shared_and_pid_parameters_across_target_dims(self) -> None:
        arguments = {
            "n_continuous": 4,
            "n_species": 3,
            "n_rec_pid_classes": 6,
            "hidden_width": 12,
            "hidden_layers": 2,
            "pid_embedding_dim": 4,
            "mixture_components": 3,
            "dropout": 0.0,
        }
        control = ConditionalMDN(**arguments, target_dim=3)
        treatment = ConditionalMDN(**arguments, target_dim=4)
        control.reset_parameters(seed=31415)
        treatment.reset_parameters(seed=31415)
        control_state = control.state_dict()
        treatment_state = treatment.state_dict()
        paired_names = [
            name
            for name in control_state
            if name.startswith("species_embedding")
            or name.startswith("backbone")
            or name.startswith("mixture_head")
            or name.startswith("pid_head")
        ]
        self.assertTrue(paired_names)
        for name in paired_names:
            torch.testing.assert_close(control_state[name], treatment_state[name])


class ConditionalPIDClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generated_species = np.asarray([211, 211, 211, 211, 2212])
        # The points at 1 and 2 GeV test [low,high) and final-edge-inclusive handling.
        self.generated_momentum = np.asarray([0.0, 0.5, 1.0, 2.0, 0.5])
        self.observed_pid_index = np.asarray([0, 1, 1, 0, 1])
        self.probabilities = np.asarray(
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.1, 0.9],
                [0.7, 0.3],
                [0.2, 0.8],
            ]
        )
        self.labels: list[int | str] = [211, 2212]

    def test_fixed_bins_use_mean_softmax_and_no_boundary_double_count(self) -> None:
        rows, summaries = conditional_pid_response_rows(
            self.generated_species,
            self.generated_momentum,
            self.observed_pid_index,
            self.probabilities,
            self.labels,
            np.asarray([0.0, 1.0, 2.0]),
        )
        pi_summaries = [row for row in summaries if row["generated_pid"] == 211]
        self.assertEqual([row["n"] for row in pi_summaries], [2, 2])
        first_pi = next(
            row
            for row in rows
            if row["generated_pid"] == 211
            and row["bin_index"] == 0
            and row["reconstructed_pid"] == 211
        )
        self.assertAlmostEqual(first_pi["coatjava_fraction"], 0.5)
        self.assertAlmostEqual(first_pi["fm_mean_probability"], 0.6)
        self.assertAlmostEqual(pi_summaries[0]["total_variation_distance"], 0.1)
        for summary in summaries:
            self.assertAlmostEqual(summary["coatjava_row_sum"], 1.0)
            self.assertAlmostEqual(summary["fm_row_sum"], 1.0)

    def test_integrated_correct_response_uses_diagonal_class_probability(self) -> None:
        rows = integrated_correct_pid_response(
            self.generated_species,
            self.observed_pid_index,
            self.probabilities,
            self.labels,
        )
        pi_row = next(row for row in rows if row["generated_pid"] == 211)
        proton_row = next(row for row in rows if row["generated_pid"] == 2212)
        self.assertAlmostEqual(pi_row["coatjava_correct_fraction"], 0.5)
        self.assertAlmostEqual(pi_row["fm_correct_mean_probability"], 0.5)
        self.assertAlmostEqual(proton_row["coatjava_correct_fraction"], 1.0)
        self.assertAlmostEqual(proton_row["fm_correct_mean_probability"], 0.8)

    def test_misaligned_inputs_fail(self) -> None:
        with self.assertRaises(ValueError):
            conditional_pid_response_rows(
                self.generated_species[:-1],
                self.generated_momentum,
                self.observed_pid_index,
                self.probabilities,
                self.labels,
                np.asarray([0.0, 1.0]),
            )


class BetaClosureTests(unittest.TestCase):
    def test_identical_beta_sample_has_exact_bin_closure(self) -> None:
        momentum = np.repeat(np.asarray([0.25, 0.75, 1.25, 2.0]), 10)
        n = len(momentum)
        raw_targets = np.zeros((n, 4), dtype=np.float32)
        raw_targets[:, 3] = np.linspace(-0.01, 0.01, n)
        features = np.column_stack(
            [
                np.log1p(momentum),
                np.full(n, 0.2),
                np.zeros(n),
                np.ones(n),
            ]
        ).astype(np.float32)
        split = PreparedSplit(
            name="test",
            event_keys=np.asarray([f"event:{i}" for i in range(n)], dtype=object),
            continuous=features,
            species_index=np.ones(n, dtype=np.int64),
            targets=raw_targets,
            rec_pid_index=np.zeros(n, dtype=np.int64),
            raw_species=np.full(n, 211, dtype=np.int64),
            target_names=(*BASE_TARGET_COLUMNS, BETA_TARGET_COLUMN),
        )
        feature_scaler = Standardizer(
            mean=np.zeros(4, dtype=np.float32), scale=np.ones(4, dtype=np.float32)
        )
        target_scaler = Standardizer(
            mean=np.zeros(4, dtype=np.float32), scale=np.ones(4, dtype=np.float32)
        )
        rows, overall = beta_closure_rows(
            split,
            feature_scaler,
            target_scaler,
            raw_targets.copy(),
            np.asarray([0.0, 1.0, 2.0]),
            0.0,
            1.2,
        )
        self.assertEqual([row["n"] for row in rows], [20, 20])
        self.assertTrue(all(row["wasserstein_1d"] < 1e-7 for row in rows))
        self.assertLess(overall[0]["wasserstein_1d"], 1e-7)


if __name__ == "__main__":
    unittest.main()
