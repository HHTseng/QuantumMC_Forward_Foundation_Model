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
    build_optimizer,
    seed_everything,
    selection_value,
    train_model,
    warmup_cosine_factor,
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


class WarmupScheduleTests(unittest.TestCase):
    """Warm-up must ramp inside the first epoch and then hand over to the decay."""

    def test_ramps_linearly_then_reaches_full_rate(self) -> None:
        factors = [warmup_cosine_factor(s, 10, 100, 0.01, False) for s in range(10)]
        self.assertAlmostEqual(factors[0], 0.1)
        self.assertAlmostEqual(factors[-1], 1.0)
        differences = np.diff(factors)
        np.testing.assert_allclose(differences, differences[0], rtol=1e-9)
        self.assertAlmostEqual(warmup_cosine_factor(50, 10, 100, 0.01, False), 1.0)

    def test_cosine_tail_decays_to_the_floor(self) -> None:
        self.assertAlmostEqual(warmup_cosine_factor(9, 10, 100, 0.01, True), 1.0)
        self.assertAlmostEqual(warmup_cosine_factor(99, 10, 100, 0.01, True), 0.01, places=3)
        middle = warmup_cosine_factor(54, 10, 100, 0.01, True)
        self.assertTrue(0.4 < middle < 0.6, middle)

    def test_never_exceeds_the_base_rate(self) -> None:
        values = [warmup_cosine_factor(s, 25, 200, 0.01, True) for s in range(200)]
        self.assertLessEqual(max(values), 1.0)
        self.assertGreater(min(values), 0.0)


class TrainingScheduleIntegrationTests(unittest.TestCase):
    """Enabling warm-up must not perturb the runs that do not ask for it."""

    @staticmethod
    def config(**training: object) -> dict:
        base = {
            "project": {"seed": 3},
            "training": {
                "epochs": 3,
                "batch_size": 256,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "pid_loss_weight": 0.2,
                "gradient_clip_norm": 5.0,
                "early_stopping_patience": 5,
                "num_workers": 0,
                "fast_loader": True,
            },
        }
        base["training"].update(training)
        return base

    def splits(self) -> dict:
        return {
            "train": _synthetic_split(768, seed=1),
            "validation": _synthetic_split(256, seed=2),
        }

    def fit(self, config: dict) -> list[dict]:
        seed_everything(3)
        model = ConditionalMDN(
            4, 3, 12, hidden_width=16, hidden_layers=2, mixture_components=3, dropout=0.0
        )
        _model, history, _best = train_model(
            model, self.splits(), config, torch.device("cpu")
        )
        return history

    def test_default_path_is_unchanged_when_warmup_is_absent(self) -> None:
        without = self.fit(self.config())
        explicit_zero = self.fit(self.config(lr_warmup_epochs=0.0))
        for left, right in zip(without, explicit_zero):
            self.assertAlmostEqual(
                left["validation"]["total_loss"], right["validation"]["total_loss"]
            )
            self.assertAlmostEqual(left["learning_rate"], right["learning_rate"])

    def test_warmup_starts_below_the_configured_rate(self) -> None:
        history = self.fit(self.config(lr_warmup_epochs=1.0, lr_schedule="cosine"))
        # Three batches per epoch, so the first epoch ends mid-ramp or just at
        # the peak; what matters is that it never overshoots the base rate.
        self.assertLessEqual(history[0]["learning_rate"], 1e-3 + 1e-12)
        self.assertGreater(history[0]["learning_rate"], 0.0)

    def test_warmup_longer_than_the_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.fit(self.config(lr_warmup_epochs=99.0))


class OptimizerGroupTests(unittest.TestCase):
    """Decoupled head learning rates, without disturbing the default."""

    @staticmethod
    def model() -> ConditionalMDN:
        return ConditionalMDN(
            4, 3, 12, hidden_width=16, hidden_layers=2, mixture_components=3
        )

    def test_default_builds_a_single_group(self) -> None:
        optimizer = build_optimizer(
            self.model(), {"learning_rate": 1e-3, "weight_decay": 1e-5}
        )
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-3)

    def test_multipliers_scale_only_their_own_group(self) -> None:
        optimizer = build_optimizer(
            self.model(),
            {
                "learning_rate": 1e-3,
                "weight_decay": 1e-5,
                "pid_head_lr_multiplier": 4.0,
                "backbone_lr_multiplier": 0.25,
            },
        )
        rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        self.assertAlmostEqual(rates["backbone"], 2.5e-4)
        self.assertAlmostEqual(rates["pid_head"], 4e-3)
        self.assertAlmostEqual(rates["density_heads"], 1e-3)

    def test_every_parameter_is_optimized_exactly_once(self) -> None:
        model = self.model()
        optimizer = build_optimizer(
            model,
            {
                "learning_rate": 1e-3,
                "weight_decay": 1e-5,
                "pid_head_lr_multiplier": 2.0,
            },
        )
        grouped = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(grouped), len(list(model.parameters())))
        self.assertEqual(len({id(p) for p in grouped}), len(grouped))

    def test_freezing_the_backbone_excludes_it(self) -> None:
        model = self.model()
        optimizer = build_optimizer(
            model,
            {"learning_rate": 1e-3, "weight_decay": 1e-5, "freeze_backbone": True},
        )
        names = {group["name"] for group in optimizer.param_groups}
        self.assertNotIn("backbone", names)
        self.assertFalse(model.backbone[0].weight.requires_grad)
        self.assertTrue(model.pid_head.weight.requires_grad)
        grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertNotIn(id(model.backbone[0].weight), grouped)

    def test_unit_multipliers_train_identically_to_the_default(self) -> None:
        split = _synthetic_split(512, seed=5)
        device = torch.device("cpu")

        def fit(training: dict) -> list[float]:
            seed_everything(11)
            model = ConditionalMDN(
                4, 3, 12, hidden_width=16, hidden_layers=2, mixture_components=3,
                dropout=0.0,
            )
            optimizer = build_optimizer(model, training)
            model.to(device)
            losses = []
            for _ in range(3):
                loader = build_loader(split, 128, False, 11, 0, device, True)
                losses.append(
                    run_epoch(model, loader, device, 0.2, optimizer=optimizer).total_loss
                )
            return losses

        base = {"learning_rate": 1e-3, "weight_decay": 1e-5}
        explicit = {**base, "pid_head_lr_multiplier": 1.0, "backbone_lr_multiplier": 1.0}
        np.testing.assert_allclose(fit(base), fit(explicit), rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
