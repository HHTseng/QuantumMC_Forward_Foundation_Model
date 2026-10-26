#!/usr/bin/env python3
"""Train and evaluate the integrated Joo split-asinh detector response."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from forwardfm_step1.config import load_config, apply_smoke_overrides, resolve_run_dir
from forwardfm_step1.data_asinh_dp import (
    CONTINUOUS_FEATURES, GEN_MASS_GEV, SPECIES, TARGET_COLUMNS,
    TARGET_TRANSFORM, load_all_splits,
)
from forwardfm_step1.evaluation_split_dp import evaluate_split
from forwardfm_step1.model_split_dp import ConditionalSplitMDN, initialize_raw_momentum_from_control
from forwardfm_step1.training_split_dp import seed_everything, choose_device, train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gpu_split_dp_asinh.yaml")
    parser.add_argument("--raw-momentum", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--parquet-glob")
    parser.add_argument("--run-dir")
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke:
        config = apply_smoke_overrides(config)
        config["training"]["preload_to_device"] = False
    if args.raw_momentum:
        config["data"]["include_raw_generated_momentum"] = True
    if args.seed is not None:
        config["project"]["seed"] = args.seed
    if args.parquet_glob:
        config["data"]["parquet_glob"] = args.parquet_glob
    if args.run_dir:
        config["output"]["run_dir"] = args.run_dir
    if args.device:
        config["training"]["device"] = args.device
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    device = choose_device(config["training"]["device"])
    run_dir = resolve_run_dir(config)
    if (run_dir/"model.pt").exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {run_dir}")
    run_dir.mkdir(parents=True,exist_ok=True)
    splits, fs, ts, vocabulary, audit = load_all_splits(config)
    names = splits["train"].feature_names
    kwargs = dict(config["model"], n_continuous=5, n_species=len(SPECIES),
                  n_rec_pid_classes=len(vocabulary)+1, target_dim=4)
    control = ConditionalSplitMDN(**kwargs)
    max_difference = 0.0
    if len(names)==6:
        model = ConditionalSplitMDN(**{**kwargs,"n_continuous":6})
        initialize_raw_momentum_from_control(control,model)
        control.eval(); model.eval()
        with torch.no_grad():
            features = torch.from_numpy(splits["train"].continuous[:128])
            species = torch.from_numpy(splits["train"].species_index[:128])
            reference, extended = control(features[:,:5],species),model(features,species)
            for name, reference_value in vars(reference).items():
                value = getattr(extended,name)
                torch.testing.assert_close(value,reference_value,rtol=2e-5,atol=3e-6)
                max_difference = max(max_difference,float((value-reference_value).abs().max()))
    else:
        model = control
    del control
    # Pair dropout randomness as well as parameters and particle order.
    seed_everything(seed)
    print(f"device={device} inputs={len(names)} split_counts="+str({n:len(s) for n,s in splits.items()}),flush=True)
    model,history,best_epoch = train_model(model,splits,config,device)
    root = Path(__file__).resolve().parent
    source_files = [Path(__file__).name,*[f"forwardfm_step1/{name}.py" for name in (
        "data_asinh_dp","model_split_dp","training_split_dp","evaluation_split_dp")]]
    checkpoint = {
        "format_version":2,"model_type":"split_delta_p_asinh",
        "model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "architecture":model.architecture_dict(),"feature_names":list(names),
        "target_names":list(TARGET_COLUMNS),"target_transform":dict(TARGET_TRANSFORM),
        "particle_masses_gev":{str(k):v for k,v in GEN_MASS_GEV.items()},
        "species_pids":list(SPECIES),"rec_pid_vocabulary":vocabulary,
        "feature_scaler":fs.as_dict(),"target_scaler":ts.as_dict(),
        "selection_sql":audit["selection_sql"],"dataset_metadata_sha256":audit["dataset_metadata_sha256"],
        "particle_order_sha256":audit["particle_order_sha256"],
        "seed":seed,"split_seed":audit["split_seed"],"order_seed":audit["order_seed"],
        "best_epoch":best_epoch,"checkpoint_metric":"validation_total_loss",
        "paired_initial_output_max_difference":max_difference,
        "source_sha256":{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_files},
    }
    torch.save(checkpoint,run_dir/"model.pt")
    serializable = {k:v for k,v in config.items() if not k.startswith("_")}
    (run_dir/"resolved_config.yaml").write_text(yaml.safe_dump(serializable,sort_keys=False))
    (run_dir/"data_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    (run_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n")
    (run_dir/"environment.json").write_text(json.dumps({"python":sys.version,"torch":str(torch.__version__),
        "numpy":np.__version__,"platform":platform.platform(),"device":str(device),
        "cuda":torch.version.cuda,"source_sha256":checkpoint["source_sha256"]},indent=2)+"\n")
    metrics = {}
    for split in ("validation","test"):
        metrics[split] = evaluate_split(model,splits[split],fs,ts,vocabulary,config,run_dir/split)
    (run_dir/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(f"checkpoint={run_dir/'model.pt'} best_epoch={best_epoch}",flush=True)
    print(json.dumps({k:{name:v[name] for name in ("pid_tv_particle_weighted","high_pion_width_mae","beta_w1_macro")} for k,v in metrics.items()}),flush=True)


if __name__=="__main__":
    main()
