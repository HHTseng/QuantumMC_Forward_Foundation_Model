from __future__ import annotations

import unittest

import numpy as np
import torch

from forwardfm_step1.data import PreparedSplit, Standardizer, assert_event_disjoint
from forwardfm_step1.model import ConditionalMDN, mixture_nll, sample_standardized_residuals


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


if __name__ == "__main__":
    unittest.main()
