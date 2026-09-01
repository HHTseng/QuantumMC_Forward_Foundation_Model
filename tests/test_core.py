from __future__ import annotations

import unittest

import numpy as np
import torch

from forwardfm_step1.data import PreparedSplit, Standardizer, assert_event_disjoint
from forwardfm_step1.evaluation import (
    conditional_pid_response_rows,
    integrated_correct_pid_response,
)
from forwardfm_step1.model import ConditionalMDN, mixture_nll, sample_standardized_residuals
from forwardfm_step1.training import (
    EpochMetrics,
    build_loader,
    run_epoch,
    seed_everything,
    selection_value,
)


class StandardizerTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        values = np.asarray([[1.0, 2.0], [3.0, 8.0], [5.0, 4.0]], dtype=np.float32)
        scaler = Standardizer.fit(values)
        restored = scaler.inverse(scaler.transform(values))
        np.testing.assert_allclose(restored, values, rtol=1e-6, atol=1e-6)


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


def _synthetic_split(n: int = 1024, seed: int = 0) -> PreparedSplit:
    rng = np.random.default_rng(seed)
    return PreparedSplit(
        name="train",
        event_keys=np.arange(n),
        continuous=rng.normal(size=(n, 4)).astype(np.float32),
        species_index=rng.integers(0, 3, n).astype(np.int64),
        targets=rng.normal(size=(n, 3)).astype(np.float32),
        rec_pid_index=rng.integers(0, 12, n).astype(np.int64),
        raw_species=rng.choice([-211, 211, 2212], n).astype(np.int64),
    )


class DeviceResidentLoaderTests(unittest.TestCase):
    """The fast loader is a throughput optimization, not a numerical change."""

    def setUp(self) -> None:
        self.split = _synthetic_split()
        self.device = torch.device("cpu")

    def _train(self, fast: bool, shuffle: bool) -> tuple[list[float], ConditionalMDN]:
        seed_everything(7)
        model = ConditionalMDN(
            4, 3, 12, hidden_width=32, hidden_layers=2, mixture_components=3, dropout=0.0
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.to(self.device)
        losses = []
        for _ in range(3):
            loader = build_loader(self.split, 256, shuffle, 7, 0, self.device, fast)
            metrics = run_epoch(model, loader, self.device, 0.2, optimizer=optimizer)
            losses.append(metrics.total_loss)
        return losses, model

    def test_matches_dataloader_without_shuffling(self) -> None:
        slow_losses, slow_model = self._train(fast=False, shuffle=False)
        fast_losses, fast_model = self._train(fast=True, shuffle=False)
        np.testing.assert_allclose(slow_losses, fast_losses, rtol=1e-5, atol=1e-6)
        for key, value in slow_model.state_dict().items():
            torch.testing.assert_close(value, fast_model.state_dict()[key])

    def test_is_reproducible_when_shuffling(self) -> None:
        first, _ = self._train(fast=True, shuffle=True)
        second, _ = self._train(fast=True, shuffle=True)
        np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-7)

    def test_visits_every_row_once_per_epoch(self) -> None:
        loader = build_loader(self.split, 300, True, 7, 0, self.device, True)
        rows = torch.cat([batch[3] for batch in loader])
        self.assertEqual(len(rows), len(self.split))
        self.assertEqual(len(loader), 4)


class SelectionMetricTests(unittest.TestCase):
    """Checkpoint ranking must not depend on the training pid_loss_weight."""

    @staticmethod
    def metrics(nll: float, pid_ce: float, weight: float) -> EpochMetrics:
        return EpochMetrics(
            total_loss=nll + weight * pid_ce,
            residual_nll=nll,
            pid_cross_entropy=pid_ce,
            pid_accuracy=0.5,
            examples_per_second=1.0,
        )

    def test_joint_nll_ignores_the_training_weight(self) -> None:
        light = self.metrics(-4.0, 1.2, 0.2)
        heavy = self.metrics(-4.0, 1.2, 5.0)
        self.assertNotAlmostEqual(
            selection_value(light, "total_loss"), selection_value(heavy, "total_loss")
        )
        self.assertAlmostEqual(
            selection_value(light, "joint_nll"), selection_value(heavy, "joint_nll")
        )
        self.assertAlmostEqual(selection_value(light, "joint_nll"), -2.8)

    def test_unknown_metric_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            selection_value(self.metrics(-4.0, 1.2, 0.2), "accuracy")


if __name__ == "__main__":
    unittest.main()
