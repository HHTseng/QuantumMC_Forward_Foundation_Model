from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from forwardfm_electron.data import OUTCOME_LABELS, encode_outcomes, feature_matrix
from forwardfm_hadron.data import denominator_sql, load_hadron_splits
from forwardfm_hadron.inference import predict_checkpoint
from forwardfm_hadron.model import HadronEfficiencyNet
from forwardfm_step1.data import Standardizer


def synthetic_particles(n_events: int = 600) -> pd.DataFrame:
    """One electron and each hadron per event, including failed reconstruction."""
    rows = []
    for event in range(n_events):
        trigger = event % 3 != 0
        for mcindex, species in enumerate((11, -211, 211, 2212)):
            reconstructed = trigger if species == 11 else event % 5 != 0
            region = "FD" if species == 11 else ("FD", "CD", "FT", "unknown")[event % 4]
            rows.append(
                {
                    "source_file_id": 7,
                    "event_id": event,
                    "mcindex": mcindex,
                    "gen_pid": species,
                    "gen_p": 0.3 + (event % 100) / 10,
                    "gen_theta": 0.1 + (event % 20) / 50,
                    "gen_phi": -np.pi + (event % 60) * np.pi / 30,
                    "gen_vx": 0.0,
                    "gen_vy": 0.0,
                    "gen_vz": -3.0 + np.sin(event),
                    "has_valid_trigger_electron": trigger,
                    "trigger_mcindex": 0 if trigger else -1,
                    "is_generated_trigger_electron": species == 11 and trigger,
                    "reconstructed": reconstructed,
                    "matched_pindex": mcindex if reconstructed else -1,
                    "rec_detector_region": region if reconstructed else None,
                    # Deliberately unsuitable for residual/PID training: none of
                    # these fields may remove a reconstruction-efficiency row.
                    "usable_for_hadron_response_training": False,
                    "match_reciprocal": False,
                    "rec_pid": 11 if species == 11 else 0,
                    "rec_beta": -999.0,
                    "delta_p": 1000.0,
                    "delta_theta": 0.0,
                    "delta_phi": 0.0,
                }
            )
    return pd.DataFrame(rows)


class HadronDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.parquet = Path(self.directory.name) / "particles.parquet"
        self.frame = synthetic_particles()
        self.config = {
            "project": {"seed": 20260822},
            "data": {
                "parquet_glob": str(self.parquet),
                "split_seed": 20260822,
                "split_modulus": 10000,
                "train_boundary": 8000,
                "validation_boundary": 9000,
                "max_rows": {"train": None, "validation": None, "test": None},
            },
        }
        self.write_particles(self.frame)

    def write_particles(self, frame: pd.DataFrame) -> None:
        with duckdb.connect() as connection:
            connection.register("fixture", frame)
            quoted = str(self.parquet).replace("'", "''")
            connection.execute(f"COPY fixture TO '{quoted}' (FORMAT PARQUET)")

    def test_denominator_contains_trigger_and_truth_species_only(self) -> None:
        for species in (-211, 211, 2212):
            sql = denominator_sql(species).lower()
            self.assertIn("gen_pid", sql)
            self.assertIn(str(species), sql)
            self.assertIn("has_valid_trigger_electron", sql)
            for forbidden in (
                "usable_for_hadron_response_training", "reconstructed",
                "matched_pindex", "rec_pid", "rec_beta", "match_reciprocal",
                "delta_p", "rec_detector_region",
            ):
                self.assertNotIn(forbidden, sql)
        for unsupported in (11, 321, -321, 0):
            with self.subTest(species=unsupported), self.assertRaises(ValueError):
                denominator_sql(unsupported)

    def test_all_triggered_truth_rows_survive_pid_and_response_pathologies(self) -> None:
        for species in (-211, 211, 2212):
            splits, scaler, _ = load_hadron_splits(self.config, species)
            truth = self.frame[
                (self.frame.gen_pid == species) & self.frame.has_valid_trigger_electron
            ]
            self.assertEqual(sum(len(split) for split in splits.values()), len(truth))
            expected = np.bincount(encode_outcomes(truth), minlength=len(OUTCOME_LABELS))
            actual = sum(
                (np.bincount(split.outcome_target, minlength=len(OUTCOME_LABELS))
                 for split in splits.values()),
                np.zeros(len(OUTCOME_LABELS), dtype=np.int64),
            )
            np.testing.assert_array_equal(actual, expected)
            self.assertGreater(actual[OUTCOME_LABELS.index("unreconstructed")], 0)
            self.assertGreater(actual[OUTCOME_LABELS.index("CD")], 0)
            # Constant vx,vy are omitted; no reconstructed label is a feature.
            self.assertEqual(splits["train"].continuous.shape[1], 5)
            self.assertEqual(len(scaler.mean), 5)

    def test_region_and_reconstruction_are_not_pid_identification(self) -> None:
        frame = self.frame[(self.frame.gen_pid == 211) & self.frame.has_valid_trigger_electron]
        labels = encode_outcomes(frame)
        any_reconstruction = labels != OUTCOME_LABELS.index("unreconstructed")
        fd_reconstruction = labels == OUTCOME_LABELS.index("FD")
        self.assertGreater(int(any_reconstruction.sum()), int(fd_reconstruction.sum()))
        self.assertTrue(np.all(frame.rec_pid == 0))
        np.testing.assert_array_equal(any_reconstruction, frame.reconstructed.to_numpy())

    def test_feature_map_is_truth_only(self) -> None:
        altered = self.frame.copy()
        altered["reconstructed"] = ~altered["reconstructed"]
        altered["rec_pid"] = 321
        altered["rec_beta"] = np.nan
        altered["delta_p"] = -1e9
        np.testing.assert_array_equal(feature_matrix(self.frame), feature_matrix(altered))

    def test_fixed_split_is_independent_of_network_seed(self) -> None:
        reference = None
        for seed in (20260822, 20260823, 20260824):
            self.config["project"]["seed"] = seed
            splits, _, _ = load_hadron_splits(self.config, 211)
            keys = {name: set(split.event_keys.tolist()) for name, split in splits.items()}
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
                self.assertFalse(keys[left] & keys[right])
            if reference is None:
                reference = keys
            else:
                self.assertEqual(keys, reference)

    def test_duplicate_particle_is_rejected(self) -> None:
        duplicate = self.frame[(self.frame.gen_pid == 211) & self.frame.has_valid_trigger_electron].iloc[[0]]
        self.write_particles(pd.concat([self.frame, duplicate], ignore_index=True))
        with self.assertRaises((ValueError, AssertionError)):
            load_hadron_splits(self.config, 211)

    def test_missing_generated_species_row_is_rejected(self) -> None:
        index = self.frame.index[self.frame.gen_pid == 211][0]
        self.write_particles(self.frame.drop(index=index))
        with self.assertRaises((ValueError, AssertionError)):
            load_hadron_splits(self.config, 211)

    def test_null_trigger_is_rejected_not_silently_excluded(self) -> None:
        altered = self.frame.copy()
        index = altered.index[altered.gen_pid == 211][0]
        altered["has_valid_trigger_electron"] = altered.has_valid_trigger_electron.astype(object)
        altered.loc[index, "has_valid_trigger_electron"] = None
        self.write_particles(altered)
        with self.assertRaises((ValueError, AssertionError)):
            load_hadron_splits(self.config, 211)

    def test_particles_in_one_event_must_share_trigger_label(self) -> None:
        altered = self.frame.copy()
        index = altered.index[(altered.gen_pid == 211) & altered.has_valid_trigger_electron][0]
        altered.loc[index, "has_valid_trigger_electron"] = False
        self.write_particles(altered)
        with self.assertRaises((ValueError, AssertionError)):
            load_hadron_splits(self.config, 211)

    def test_reconstruction_match_inconsistency_is_rejected(self) -> None:
        for reconstructed, matched in ((True, -1), (True, np.nan), (False, 3)):
            altered = self.frame.copy()
            index = altered.index[(altered.gen_pid == 211) & altered.has_valid_trigger_electron][0]
            altered.loc[index, "reconstructed"] = reconstructed
            altered.loc[index, "matched_pindex"] = matched
            self.write_particles(altered)
            with self.subTest(reconstructed=reconstructed, matched=matched):
                with self.assertRaises((ValueError, AssertionError)):
                    load_hadron_splits(self.config, 211)


class HadronModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        torch.manual_seed(17)
        self.model = HadronEfficiencyNet(
            n_continuous=5, n_outcomes=5, hidden_width=16,
            hidden_layers=2, dropout=0.0,
        )
        self.checkpoint = Path(self.directory.name) / "model.pt"
        torch.save(
            {
                "format_version": 1,
                "task": "trigger_conditioned_hadron_efficiency",
                "species": 211,
                "architecture": self.model.architecture_dict(),
                "model_state": self.model.state_dict(),
                "feature_names": [
                    "log1p_gen_p", "gen_theta", "sin_gen_phi", "cos_gen_phi", "gen_vz",
                ],
                "dropped_constant_feature_values": {"gen_vx": 0.0, "gen_vy": 0.0},
                "feature_scaler": Standardizer(
                    mean=np.zeros(5, dtype=np.float32),
                    scale=np.ones(5, dtype=np.float32),
                ).as_dict(),
                "outcome_labels": list(OUTCOME_LABELS),
                "conditioning": "has_valid_trigger_electron = true",
            },
            self.checkpoint,
        )
        self.truth = pd.DataFrame(
            {
                "gen_pid": [211, 211, 211],
                "gen_p": [0.3, 2.0, 8.0],
                "gen_theta": [0.1, 0.3, 0.6],
                "gen_phi": [-np.pi, 0.0, np.pi],
                "gen_vz": [-3.0, -2.0, 0.0],
            }
        )

    def test_shapes_probabilities_and_gradients(self) -> None:
        logits = self.model(torch.randn(13, 5))
        self.assertEqual(tuple(logits.shape), (13, 5))
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(13) % 5)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_checkpoint_predictions_are_coherent_and_batch_independent(self) -> None:
        result = predict_checkpoint(self.checkpoint, self.truth, device="cpu", batch_size=1)
        full_batch = predict_checkpoint(self.checkpoint, self.truth, device="cpu", batch_size=16)
        names = [f"prob_{label}" for label in OUTCOME_LABELS]
        probability = result[names].to_numpy()
        np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.all((probability >= 0) & (probability <= 1)))
        np.testing.assert_allclose(result.prob_reconstructed_any, 1 - result.prob_unreconstructed)
        np.testing.assert_allclose(result.prob_reconstructed_fd, result.prob_FD)
        self.assertTrue(np.all(result.prob_reconstructed_fd <= result.prob_reconstructed_any + 1e-7))
        np.testing.assert_allclose(result[names], full_batch[names], atol=1e-6)

    def test_inference_rejects_another_species(self) -> None:
        altered = self.truth.copy()
        altered.loc[0, "gen_pid"] = 2212
        with self.assertRaises(ValueError):
            predict_checkpoint(self.checkpoint, altered)

    def test_inference_rejects_missing_active_feature(self) -> None:
        with self.assertRaises(ValueError):
            predict_checkpoint(self.checkpoint, self.truth.drop(columns="gen_vz"))

    def test_checkpoint_task_and_outcome_order_are_validated(self) -> None:
        original = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        for key, value in (
            ("task", "trigger_electron_efficiency"),
            ("outcome_labels", list(reversed(OUTCOME_LABELS))),
        ):
            altered = dict(original)
            altered[key] = value
            torch.save(altered, self.checkpoint)
            with self.subTest(key=key), self.assertRaises(ValueError):
                predict_checkpoint(self.checkpoint, self.truth)

    def test_azimuth_is_periodic(self) -> None:
        shifted = self.truth.copy()
        shifted["gen_phi"] += 2 * np.pi
        first = predict_checkpoint(self.checkpoint, self.truth)
        second = predict_checkpoint(self.checkpoint, shifted)
        names = [f"prob_{label}" for label in OUTCOME_LABELS]
        np.testing.assert_allclose(first[names], second[names], atol=1e-6)

    def test_inference_rejects_invalid_truth_or_changed_constant_vertex(self) -> None:
        for column, value in (
            ("gen_p", -0.1), ("gen_p", np.nan),
            ("gen_theta", -0.1), ("gen_theta", np.pi + 0.1),
            ("gen_vz", np.inf), ("gen_vx", 0.1),
        ):
            altered = self.truth.copy()
            altered.loc[0, column] = value
            if column == "gen_vx":
                altered["gen_vx"] = altered.gen_vx.fillna(0.0)
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                predict_checkpoint(self.checkpoint, altered)


if __name__ == "__main__":
    unittest.main()
