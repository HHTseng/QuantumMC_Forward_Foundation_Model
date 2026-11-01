"""Synthetic checks of physics labels, units and response coordinates."""

from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd

from forwardfm_transfer.data import audit_native, connect_native


class NativeTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "native.parquet"
        records = []
        for event in range(3):
            for mcrow, pid in enumerate((11, -211, 211, 2212)):
                triggered = event < 2
                unmatched = event == 2 or (event == 0 and pid == 2212)
                ft = event == 0 and pid == -211
                records.append({
                    "file_id": 0, "event": event, "mcrow": mcrow,
                    "pid_gen": pid, "pid_rec": 0 if unmatched else (211 if event == 0 and pid == 11 else pid),
                    "recrow": -1 if unmatched else mcrow,
                    "reconstructed": int(not unmatched),
                    "triggered": int(triggered), "trigger_recrow": 0 if triggered else -1,
                    "trigger_pid_rec": (211 if event == 0 else 11) if triggered else 0,
                    "is_trigger_particle": int(triggered and pid == 11),
                    "detector": 0 if unmatched or ft else 1,
                    "status_rec": 0 if unmatched else (1500 if ft else (-2000 if pid == 11 else 2000)),
                    "px_gen": 0.0, "py_gen": 0.0, "pz_gen": 2.0, "p_gen": 2.0,
                    "theta_gen": 20.0, "phi_gen": 179.0,
                    "vx_gen": 0.0, "vy_gen": 0.0, "vz_gen": -2.0,
                    "px_rec": np.nan if unmatched else 0.0,
                    "py_rec": np.nan if unmatched else 0.0,
                    "pz_rec": np.nan if unmatched else 2.2,
                    "p_rec": np.nan if unmatched else 2.2,
                    "theta_rec": np.nan if unmatched else 22.0,
                    "phi_rec": np.nan if unmatched else -179.0,
                    "vx_rec": np.nan if unmatched else 0.0,
                    "vy_rec": np.nan if unmatched else 0.0,
                    "vz_rec": np.nan if unmatched else -2.0,
                    "beta_rec": np.nan if unmatched else 0.95,
                    "p_rec_uncorrected": np.nan if unmatched else 1.9,
                    "px_rec_uncorrected": np.nan if unmatched else 0.0,
                    "py_rec_uncorrected": np.nan if unmatched else 0.0,
                    "pz_rec_uncorrected": np.nan if unmatched else 1.9,
                    "genmatch_quality": np.nan if unmatched else 0.91,
                    "recmatch_quality": np.nan if unmatched else 0.98,
                    # Deliberately inconsistent native residuals must be ignored.
                    "delta_p": 999.0, "delta_theta": 999.0, "delta_phi": 999.0,
                })
        self.frame = pd.DataFrame(records)
        self.write_parquet()

    def tearDown(self):
        self.temp.cleanup()

    def write_parquet(self):
        with duckdb.connect() as con:
            con.register("fixture", self.frame)
            con.execute("COPY fixture TO ? (FORMAT PARQUET)", [str(self.path)])

    def test_trigger_broadcast_preserves_denominators_and_pid_mistakes(self):
        with connect_native(self.path) as con:
            self.assertEqual(con.execute("SELECT count(*) FROM particles").fetchone()[0], 12)
            rows = con.execute("""SELECT event_id,count(*) FILTER(WHERE has_valid_trigger_electron)
                FROM particles GROUP BY event_id ORDER BY event_id""").fetchall()
            self.assertEqual(rows, [(0, 4), (1, 4), (2, 0)])
            self.assertEqual(con.execute("""SELECT count(*) FROM particles
                WHERE usable_for_hadron_response_training""").fetchone()[0], 6)
            # The positive class is not restricted to correctly identified electrons.
            electron = con.execute("""SELECT rec_pid,has_valid_trigger_electron,match_reciprocal
                FROM particles WHERE event_id=0 AND gen_pid=11""").fetchone()
            self.assertEqual(electron, (211, True, True))
        with connect_native(self.path, trigger="identified_fd") as con:
            self.assertEqual(con.execute("SELECT count(*) FROM particles WHERE has_valid_trigger_electron").fetchone()[0], 4)

    def test_reconstruction_and_detector_are_distinct(self):
        with connect_native(self.path) as con:
            rows = con.execute("""SELECT gen_pid,reconstructed,rec_detector_region,match_reciprocal
                FROM particles WHERE event_id=0 AND gen_pid IN (-211,2212) ORDER BY gen_pid""").fetchall()
            self.assertEqual(rows, [(-211, True, "FT", True), (2212, False, None, False)])
            quality = con.execute("""SELECT genmatch_quality,recmatch_quality
                FROM particles WHERE event_id=0 AND gen_pid=-211""").fetchone()
            self.assertNotEqual(*quality)
            audit = audit_native(con)
            self.assertEqual(audit["counts"]["events"], 3)
            self.assertEqual(audit["counts"]["particle_rows"], 12)
            self.assertTrue(all(value == 0 for value in audit["violations"].values()))

    def test_units_sign_wrap_and_raw_corrected_choice(self):
        with connect_native(self.path) as con:
            row = con.execute("""SELECT gen_theta,delta_p,delta_theta,delta_phi,rec_p,
                raw_delta_p,corrected_delta_p FROM particles WHERE event_id=0 AND gen_pid=11""").fetchone()
            np.testing.assert_allclose(row, [np.deg2rad(20), -0.1, np.deg2rad(2),
                np.deg2rad(2), 1.9, -0.1, 0.2], atol=1e-12)
            self.assertEqual(con.execute("SELECT raw_momentum_source FROM transfer_metadata").fetchone()[0], "uncorrected_backup")
        with connect_native(self.path, momentum="corrected") as con:
            row = con.execute("""SELECT rec_p,delta_p FROM particles
                WHERE event_id=0 AND gen_pid=11""").fetchone()
            np.testing.assert_allclose(row, [2.2, 0.2], atol=1e-12)
        self.frame.loc[(self.frame.event == 1) & (self.frame.pid_gen == 211), ["phi_gen", "phi_rec"]] = [0.0, 180.0]
        self.write_parquet()
        with connect_native(self.path) as con:
            delta = con.execute("SELECT delta_phi FROM particles WHERE event_id=1 AND gen_pid=211").fetchone()[0]
            self.assertAlmostEqual(delta, -np.pi)

    def test_missing_raw_backup_is_explicit(self):
        self.frame.drop(columns=[c for c in self.frame if c.endswith("_uncorrected")], inplace=True)
        self.write_parquet()
        with connect_native(self.path) as con:
            self.assertEqual(con.execute("SELECT raw_momentum_source FROM transfer_metadata").fetchone()[0], "native_export_no_raw_backup")
            self.assertAlmostEqual(con.execute("SELECT delta_p FROM particles WHERE event_id=0 AND gen_pid=11").fetchone()[0], 0.2)

    def test_incomplete_schema_fails(self):
        self.frame.drop(columns=["is_trigger_particle"], inplace=True)
        self.write_parquet()
        with self.assertRaisesRegex(ValueError, "is_trigger_particle"):
            connect_native(self.path)

    def test_partial_backup_fails(self):
        self.frame.drop(columns=["px_rec_uncorrected"], inplace=True)
        self.write_parquet()
        with self.assertRaisesRegex(ValueError, "incomplete raw-momentum"):
            connect_native(self.path)


if __name__ == "__main__":
    unittest.main()
