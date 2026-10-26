#!/usr/bin/env python3
"""Run predeclared paired 5/6-input seeds, with one worker per GPU.

Example: worker A uses --condition baseline and worker B --condition raw.
Model seeds vary; data.split_seed and data.order_seed remain fixed in YAML.
No hyperparameters or epochs are selected using the test split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from forwardfm_step1.config import load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/gpu_split_dp_asinh.yaml')
    parser.add_argument('--output-dir',default='runs/raw_momentum_ablation')
    parser.add_argument('--condition',choices=['baseline','raw'],required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--parquet-glob',required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[20260822,20260823,20260824])
    args=parser.parse_args()
    base=load_config(args.config)
    if 'split_seed' not in base['data'] or 'order_seed' not in base['data']:
        raise ValueError('Paired experiments require explicit fixed data seeds')
    output=Path(args.output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for seed in args.seeds:
        run=output/f'seed_{seed}'/args.condition
        command=[sys.executable,'-u',str(ROOT/'train_split_dp_asinh.py'),'--config',str(Path(args.config).resolve()),
                 '--device',args.device,'--seed',str(seed),'--parquet-glob',args.parquet_glob,'--run-dir',str(run)]
        if args.condition=='raw': command.append('--raw-momentum')
        if (run/'model.pt').exists():
            raise FileExistsError(f'Existing run must be inspected explicitly before resuming: {run}')
        run.mkdir(parents=True,exist_ok=True)
        print(f'starting seed={seed} condition={args.condition}',flush=True)
        with (run/'training.log').open('w') as handle:
            subprocess.run(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
        metrics=json.loads((run/'metrics.json').read_text())
        if metrics['test']['high_pion_width_mae'] is None or len(metrics['test']['high_pion_cells'])!=6:
            raise RuntimeError(f'Incomplete primary endpoint: {run}')
        manifest.append({'seed':seed,'condition':args.condition,'command':command,
            'model_sha256':hashlib.sha256((run/'model.pt').read_bytes()).hexdigest(),
            'validation_high_pion_width_mae':metrics['validation']['high_pion_width_mae'],
            'test_high_pion_width_mae':metrics['test']['high_pion_width_mae']})
        (output/f'manifest_{args.condition}.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'finished seed={seed} condition={args.condition}',flush=True)


if __name__=='__main__': main()
