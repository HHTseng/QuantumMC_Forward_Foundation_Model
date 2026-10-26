#!/usr/bin/env python3
"""Audit all inverse-asinh draws without clipping or changing the checkpoint.

The RNG, CPU device, batch size and feature/scaler order match the study.
Compare recomputed W1 with the frozen evaluation to verify the same draws.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from forwardfm_step1 import data_asinh_dp as data
from forwardfm_step1.model_split_dp import sample_split_standardized_residuals
from forwardfm_step1.evaluation_split_dp import write_csv, _wasserstein_equal_sample
from sample_split_dp_asinh import load_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',default='runs/raw_momentum_ablation')
    parser.add_argument('--seeds',nargs='+',type=int,default=[20260822,20260823,20260824])
    args=parser.parse_args(); root=Path(args.run_root)
    first=root/f'seed_{args.seeds[0]}'/'baseline'
    config=yaml.safe_load((first/'resolved_config.yaml').read_text())
    torch.set_num_threads(int(config['training']['torch_threads']))
    con=data.connect(config['data']['parquet_glob'])
    frames={split:data._load_frame(con,split,config) for split in ('validation','test')}
    con.close(); rows=[]
    for seed in args.seeds:
        for condition in ('baseline','raw'):
            path=root/f'seed_{seed}'/condition
            model,checkpoint=load_checkpoint(path/'model.pt',torch.device('cpu'))
            fs=data.Standardizer.from_dict(checkpoint['feature_scaler'])
            ts=data.Standardizer.from_dict(checkpoint['target_scaler'])
            for split,frame in frames.items():
                features=fs.transform(data.build_features(frame,tuple(checkpoint['feature_names'])))
                species=np.array([data.SPECIES.index(int(pid)) for pid in frame.gen_pid],dtype=np.int64)
                generator=torch.Generator().manual_seed(int(config['evaluation']['sampling_seed']))
                samples=[]
                with torch.no_grad():
                    for start in range(0,len(frame),8192):
                        sl=slice(start,start+8192)
                        output=model(torch.from_numpy(features[sl]),torch.from_numpy(species[sl]))
                        samples.append(sample_split_standardized_residuals(output,generator).numpy())
                physical=data.inverse_targets_from_model(ts.inverse(np.concatenate(samples)))[:,0]
                truth=frame.delta_p.to_numpy(dtype=np.float32)
                w1=float(np.mean([_wasserstein_equal_sample(truth[frame.gen_pid==pid],physical[frame.gen_pid==pid]) for pid in data.SPECIES]))
                closure=pd.read_csv(path/split/'response_closure.csv')
                expected=float(closure[closure.target=='delta_p'].wasserstein_1d.mean())
                np.testing.assert_allclose(w1,expected,rtol=3e-5,atol=2e-6)
                extreme=int(np.argmax(np.abs(physical)))
                q99,q999=np.quantile(np.abs(physical),[.99,.999])
                row={'seed':seed,'condition':condition,'split':split,'n':len(frame),
                    'delta_p_w1_macro_GeV':w1,'q99_abs_delta_p_GeV':float(q99),
                    'q999_abs_delta_p_GeV':float(q999),'max_abs_delta_p_GeV':float(abs(physical[extreme])),
                    'count_abs_delta_p_gt_10GeV':int((np.abs(physical)>10).sum()),
                    'extreme_generated_pid':int(frame.gen_pid.iloc[extreme]),
                    'extreme_generated_p_GeV':float(frame.gen_p.iloc[extreme]),
                    'extreme_source_file_id':str(frame.source_file_id.iloc[extreme]),
                    'extreme_event_id':str(frame.event_id.iloc[extreme]),
                    'extreme_mcindex':int(frame.mcindex.iloc[extreme])}
                rows.append(row)
    write_csv(root/'summary'/'tail_audit.csv',rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
