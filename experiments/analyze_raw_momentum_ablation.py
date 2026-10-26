#!/usr/bin/env python3
"""Analyze complete paired seeds; reject mismatched data or partial runs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from forwardfm_step1.evaluation_split_dp import write_csv

SPECIES=[('pi-',r'$\pi^-$'),('pi+',r'$\pi^+$'),('proton','proton')]
LABELS={'baseline':'5 inputs','raw':'6 inputs (+ raw p)'}
COLORS={'baseline':'tab:blue','raw':'tab:orange'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',default='runs/raw_momentum_ablation')
    parser.add_argument('--seeds',nargs='+',type=int,default=[20260822,20260823,20260824])
    args=parser.parse_args()
    root=Path(args.run_root).resolve(); output=root/'summary'; output.mkdir(exist_ok=True)
    provenance=[]; run_rows=[]; core_rows=[]; all_bins=[]; all_histograms=[]
    reference_data=None; reference_source=None
    for seed in args.seeds:
        paired_configs=[]
        for condition in ('baseline','raw'):
            path=root/f'seed_{seed}'/condition
            required=[path/name for name in ('model.pt','history.json','metrics.json','data_audit.json','resolved_config.yaml')]
            if not all(p.exists() for p in required):
                raise RuntimeError(f'Incomplete run: {path}')
            config=yaml.safe_load((path/'resolved_config.yaml').read_text())
            history=json.loads((path/'history.json').read_text())
            audit=json.loads((path/'data_audit.json').read_text())
            checkpoint=torch.load(path/'model.pt',map_location='cpu',weights_only=True)
            metrics=json.loads((path/'metrics.json').read_text())
            if len(history)!=int(config['training']['epochs']):
                raise RuntimeError(f'Incomplete epoch budget: {path}')
            if config['project']['seed']!=seed or bool(config['data']['include_raw_generated_momentum'])!=(condition=='raw'):
                raise RuntimeError(f'Mislabeled condition: {path}')
            if checkpoint['architecture']['n_continuous']!=(5 if condition=='baseline' else 6):
                raise RuntimeError(f'Input dimension mismatch: {path}')
            if checkpoint['paired_initial_output_max_difference']>3e-6:
                raise RuntimeError(f'Initial functions differ: {path}')
            best=int(checkpoint['best_epoch'])
            if history[best-1]['validation']['total_loss']>min(h['validation']['total_loss'] for h in history)+1e-5:
                raise RuntimeError(f'Checkpoint selection mismatch: {path}')
            data_provenance={key:audit[key] for key in ('dataset_metadata_sha256','particle_order_sha256','sampled_counts','split_seed','order_seed','selection_sql')}
            if reference_data is None: reference_data=data_provenance
            if reference_source is None: reference_source=checkpoint['source_sha256']
            if data_provenance!=reference_data or checkpoint['source_sha256']!=reference_source:
                raise RuntimeError(f'Nonpaired data or source: {path}')
            normalized=copy.deepcopy(config)
            normalized.pop('output'); normalized['data'].pop('include_raw_generated_momentum')
            paired_configs.append(normalized)
            provenance.append({'seed':seed,'condition':condition,'checkpoint':str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                'best_epoch':best,'epochs':len(history),'sha256':hashlib.sha256((path/'model.pt').read_bytes()).hexdigest(),
                'initial_max_difference':checkpoint['paired_initial_output_max_difference']})
            for split in ('validation','test'):
                m=metrics[split]
                if m['n']!=sum(audit['sampled_counts'][split].values()) or len(m['high_pion_cells'])!=6 or m['failed_gaussian_fits']!=0:
                    raise RuntimeError(f'Incomplete evaluation: {path}/{split}')
                core=pd.read_csv(path/split/'core_summary.csv')
                core['seed']=seed; core['condition']=condition; core['split']=split
                core_rows.extend(core.to_dict(orient='records'))
                closure=pd.read_csv(path/split/'response_closure.csv')
                run_rows.append({'seed':seed,'condition':condition,'split':split,'best_epoch':best,
                    'high_pion_width_mae':m['high_pion_width_mae'],
                    'core_abs_median_bias_MeV':float(np.mean(np.abs(core['median_bias_MeV']))),
                    'core_sigma68_ratio_mae':float(np.mean(np.abs(core['sigma68_ratio']-1))),
                    'pid_tv_particle_weighted':m['pid_tv_particle_weighted'],
                    'pid_cross_entropy':m['loss']['pid_cross_entropy'],'pid_accuracy':m['loss']['pid_accuracy'],
                    'beta_w1_macro':m['beta_w1_macro'],
                    'delta_p_w1_macro_GeV':float(closure[closure.target=='delta_p'].wasserstein_1d.mean()),
                    'negative_reconstructed_p_fraction':m['negative_reconstructed_p_fraction'],
                    'abs_delta_p_gt_10_fraction':m['sampled_abs_delta_p_gt_10_fraction']})
                if split=='test':
                    bins=pd.read_csv(path/split/'delta_p_gaussian_vs_momentum.csv')
                    bins['seed']=seed; bins['condition']=condition; all_bins.append(bins)
                    hist=pd.read_csv(path/split/'core_histograms.csv')
                    hist['seed']=seed; hist['condition']=condition; all_histograms.append(hist)
        if paired_configs[0]!=paired_configs[1]:
            raise RuntimeError(f'Hyperparameters differ in seed pair {seed}')
    runs=pd.DataFrame(run_rows); cores=pd.DataFrame(core_rows)
    bins=pd.concat(all_bins,ignore_index=True); hist=pd.concat(all_histograms,ignore_index=True)
    numeric=[c for c in runs.columns if c not in ('seed','condition','split','best_epoch')]
    summaries=[]; contrasts=[]
    for split in ('validation','test'):
        subset=runs[runs.split==split]
        for metric in numeric:
            paired=subset.pivot(index='seed',columns='condition',values=metric).loc[args.seeds]
            for condition in ('baseline','raw'):
                values=paired[condition].to_numpy()
                summaries.append({'split':split,'metric':metric,'condition':condition,'n_seeds':len(values),
                    'mean':float(values.mean()),'seed_sd':float(values.std(ddof=1)) if len(values)>1 else 0.})
            delta=paired['raw']-paired['baseline']
            contrasts.append({'split':split,'metric':metric,'mean_raw_minus_baseline':float(delta.mean()),
                'seed_sd_of_difference':float(delta.std(ddof=1)) if len(delta)>1 else 0.,
                'min_difference':float(delta.min()),'max_difference':float(delta.max()),
                'raw_lower_pairs':int((delta<0).sum()),'pairs':len(delta)})
    write_csv(output/'per_run_metrics.csv',run_rows)
    write_csv(output/'condition_summary.csv',summaries)
    write_csv(output/'paired_differences.csv',contrasts)
    write_csv(output/'provenance.csv',provenance)
    cores.to_csv(output/'core_summary_all_seeds.csv',index=False)
    bins.to_csv(output/'momentum_fits_all_seeds.csv',index=False)

    figure,axes=plt.subplots(2,3,figsize=(14.5,8),sharex='col')
    for col,(species,label) in enumerate(SPECIES):
        cells=bins[bins.species==species]
        truth=cells[cells.condition=='baseline'].groupby('p_low_GeV').first()
        x=truth.index.to_numpy()+.5
        axes[0,col].plot(x,truth.coat_sigma_MeV,'ko-',label='COATJAVA',markersize=4)
        for condition in ('baseline','raw'):
            values=cells[cells.condition==condition].groupby('p_low_GeV')
            mean=values.fm_sigma_MeV.mean(); sd=values.fm_sigma_MeV.std().fillna(0)
            ratio=values.sigma_ratio.mean(); ratio_sd=values.sigma_ratio.std().fillna(0)
            axes[0,col].plot(x,mean,'o-',color=COLORS[condition],label=LABELS[condition],markersize=4)
            axes[0,col].fill_between(x,mean-sd,mean+sd,color=COLORS[condition],alpha=.15)
            axes[1,col].plot(x,ratio,'o-',color=COLORS[condition],markersize=4)
            axes[1,col].fill_between(x,ratio-ratio_sd,ratio+ratio_sd,color=COLORS[condition],alpha=.15)
        axes[0,col].set_title(f'generated {label}')
        axes[1,col].axhline(1,color='black',linestyle='--',linewidth=1)
        for axis in axes[:,col]:
            axis.axvspan(6,9,color='grey',alpha=.08)
            axis.grid(alpha=.2)
            axis.set_xlim(0,9)
        axes[1,col].set_xlabel(r'$p_{\rm gen}$ [GeV]')
    axes[0,0].set_ylabel(r'Gaussian core $\sigma_{\Delta p}$ [MeV]')
    axes[1,0].set_ylabel(r'$\sigma_{\rm FM}/\sigma_{\rm COATJAVA}$')
    axes[0,0].legend(fontsize=8)
    figure.suptitle(f'Raw generated momentum: {len(args.seeds)} paired seeds (bands: seed SD)')
    figure.tight_layout(); figure.savefig(output/'momentum_resolution_comparison.png',dpi=180); plt.close(figure)

    figure,axes=plt.subplots(1,3,figsize=(14.5,4.2))
    for axis,(species,label) in zip(axes,SPECIES):
        subset=hist[hist.species==species]
        truth=subset[subset.condition=='baseline'].groupby('delta_p_MeV').coat_density.mean()
        axis.step(truth.index,truth,where='mid',color='black',label='COATJAVA')
        for condition in ('baseline','raw'):
            values=subset[subset.condition==condition].groupby('delta_p_MeV').fm_density
            mean=values.mean(); sd=values.std().fillna(0)
            axis.step(mean.index,mean,where='mid',color=COLORS[condition],label=LABELS[condition])
            axis.fill_between(mean.index,mean-sd,mean+sd,color=COLORS[condition],alpha=.12,step='mid')
        axis.set(title=label,xlabel=r'$\Delta p$ [MeV]',xlim=(-100,100))
        axis.grid(alpha=.2)
    axes[0].set_ylabel('density conditioned on [-100,100] MeV')
    axes[0].legend(fontsize=8)
    figure.suptitle('Central momentum response (bands: seed SD)')
    figure.tight_layout(); figure.savefig(output/'central_response_comparison.png',dpi=180); plt.close(figure)

    summary={'seeds':args.seeds,'n_pairs':len(args.seeds),'model_source_sha256':reference_source,
        'data_provenance':reference_data,'primary_endpoint':'unweighted mean absolute Gaussian-width ratio error in pion cells at 6-9 GeV',
        'selection':'minimum validation joint loss within fixed 30 epochs; no test-based tuning',
        'condition_summary':summaries,'paired_differences':contrasts}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(runs.to_string(index=False))
    print('\nPrimary paired differences:',[r for r in contrasts if r['metric']=='high_pion_width_mae'])


if __name__=='__main__': main()
