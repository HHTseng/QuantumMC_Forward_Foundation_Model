"""Common-bin comparisons must not confound momentum coverage with closure."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1]/"experiments/analyze_transfer_study.py"
SPEC = importlib.util.spec_from_file_location("transfer_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class TransferAnalysisTests(unittest.TestCase):
    def test_common_bins_intersect_pid_and_valid_fit_support_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            records=[]
            for dataset in ("august", "rho"):
                directory=Path(temp)/dataset
                directory.mkdir()
                is_august=dataset=="august"
                pd.DataFrame({
                    "generated_pid":[2212]*3,"p_low_GeV":[0.,1.,2.],"p_high_GeV":[1.,2.,3.],
                    "eligible":[True,True,True] if is_august else [False,True,True],
                    "n":[100,200,300] if is_august else [150,250,350],
                    "absolute_error_pp":[10.,20.,30.] if is_august else [1.,2.,3.],
                    "total_variation":[.1,.2,.3] if is_august else [.01,.02,.03],
                    "observed_fraction":[.8,.7,.6],"fm_fraction":[.7,.5,.3] if is_august else [.79,.68,.57],
                    "observed_binomial_se":[.01]*3,
                }).to_csv(directory/"pid_closure.csv",index=False)
                pd.DataFrame({
                    "generated_pid":[2212]*3,"p_low_GeV":[0.,1.,2.],"p_high_GeV":[1.,2.,3.],
                    "eligible":[True,True,True] if is_august else [False,True,True],
                    "teacher":["raw"]*3,
                    "gaussian_sigma_ratio":[1.,1.1,None] if is_august else [1.,.8,1.2],
                    "observed_gaussian_sigma_MeV":[10.,20.,30.],
                    "fm_gaussian_sigma_MeV":[10.,22.,None] if is_august else [10.,16.,36.],
                    "observed_gaussian_sigma_error_MeV":[.1]*3,
                }).to_csv(directory/"momentum_resolution.csv",index=False)
                records.append(dict(host="tara",seed=7,dataset=dataset,path=directory))
            result=analysis.common_bin_response(records)
            proton=result[result.species=="proton"].set_index("dataset")
            self.assertEqual(proton.loc["august","common_pid_bins"],"1-2;2-3")
            self.assertEqual(proton.loc["rho","common_width_bins"],"1-2")
            self.assertEqual(proton.loc["august","correct_id_mae_pp"],25.)
            self.assertEqual(proton.loc["rho","correct_id_mae_pp"],2.5)
            self.assertAlmostEqual(proton.loc["august","pid_tv_pp"],26.)
            self.assertAlmostEqual(proton.loc["rho","gaussian_width_error_percent"],20.)
            analysis.pid_figure(records,"tara",Path(temp))
            analysis.resolution_figure(records,"tara",Path(temp))
            self.assertGreater((Path(temp)/"pid_correct_id_transfer.png").stat().st_size,1000)
            self.assertGreater((Path(temp)/"momentum_resolution_transfer.pdf").stat().st_size,1000)

    def test_seed_bands_reject_changed_teacher_particles(self):
        with tempfile.TemporaryDirectory() as temp:
            records=[]
            for seed in (1,2):
                directory=Path(temp)/str(seed)
                directory.mkdir()
                np.savez_compressed(directory/"response_events.npz",file_id=[0],event_id=[1],mcindex=[2],
                    species=[2212],raw_truth=[[seed,0.,0.,0.]],corrected_truth=[[0.,0.,0.,0.]],
                    gen_p=[1.],gen_theta=[.2])
                records.append(dict(host="tara",seed=seed,dataset="rho",path=directory))
            with self.assertRaisesRegex(ValueError,"different teacher particles"):
                analysis.check_seed_populations(records,[])


if __name__=="__main__":
    unittest.main()
