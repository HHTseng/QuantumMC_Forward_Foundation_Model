"""Efficiency tests keep failure denominators and event partitions explicit."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from forwardfm_electron.data import OUTCOME_LABELS, encode_outcomes
from forwardfm_hadron.data import load_frame, prepare_splits
from forwardfm_hadron.evaluation import binary_metrics


def test_native_outcomes_keep_failures_and_forward_tagger():
    frame = pd.DataFrame({
        "reconstructed": [False, True, True, True, True],
        "matched_pindex": [-1, 2, 3, 4, 5],
        "rec_detector_region": [None, "FD", "FT", "CD", "other"],
        "pid_rec": [0, 2212, 11, 211, 0],
    })
    assert encode_outcomes(frame).tolist() == list(range(5))
    assert OUTCOME_LABELS[2] == "FT"


def test_event_split_and_caps_do_not_select_reconstructed_only():
    n = 2000
    frame = pd.DataFrame({
        "source_file_id": np.zeros(n, int), "event_id": np.arange(n),
        "mcindex": np.ones(n, int), "gen_pid": np.full(n, 211),
        "gen_p": np.linspace(.1, 8, n), "gen_theta": np.linspace(.01, 2, n),
        "gen_phi": np.linspace(-3, 3, n), "gen_vx": np.zeros(n),
        "gen_vy": np.zeros(n), "gen_vz": np.linspace(-5.5, .5, n),
        "has_valid_trigger_electron": np.ones(n, bool),
        "reconstructed": np.arange(n) % 2 == 0,
        "matched_pindex": np.where(np.arange(n) % 2 == 0, 1, -1),
        "rec_detector_region": np.where(np.arange(n) % 2 == 0, "FD", None),
    })
    con = duckdb.connect()
    con.register("particles", frame)
    try:
        frames = {name: load_frame(con, 211, name, 100, 20261101)
                  for name in ("train", "validation", "test")}
        repeat = load_frame(con, 211, "train", 100, 20261101)
        assert repeat["event_id"].tolist() == frames["train"]["event_id"].tolist()
        splits, scaler, audit = prepare_splits(frames)
        for name in frames:
            assert len(frames[name]) == 100
            assert 0 < np.count_nonzero(splits[name].outcome_target == 0) < 100
        assert audit["phase_event_split_overlap"] == 0
        assert audit["feature_names"] == ["log1p_gen_p", "gen_theta", "sin_gen_phi", "cos_gen_phi", "gen_vz"]
        assert scaler.mean[0] == pytest.approx(np.log1p(frames["train"]["gen_p"]).mean(), abs=1e-6)
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            assert not set(frames[left]["event_id"]) & set(frames[right]["event_id"])
    finally:
        con.close()


def test_efficiency_scores_use_probability_not_thresholded_decision():
    target = np.array([0, 1, 0, 1], bool)
    probability = np.array([.1, .8, .4, .7])
    metrics = binary_metrics(target, probability, .5)
    assert metrics["mean_predicted_probability"] == pytest.approx(.5)
    assert metrics["observed_rate"] == .5
    assert metrics["brier_score"] == pytest.approx(np.mean((target - probability) ** 2))
