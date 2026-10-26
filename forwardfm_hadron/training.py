"""Unweighted CE estimates P(C|x,s,T=1); no class resampling/weighting."""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.nn import functional as F


def fit(model, splits, config, device):
    settings = config["training"]
    model.to(device)
    tensors = {
        name: (torch.from_numpy(split.continuous).to(device),
               torch.from_numpy(split.outcome_target).to(device))
        for name, split in splits.items() if name in ("train", "validation")
    }
    generator = torch.Generator(device=device).manual_seed(int(config["project"]["seed"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]),
                                  weight_decay=float(settings["weight_decay"]))
    history, best_state = [], None
    best_loss, stale, best_epoch = float("inf"), 0, 0
    for epoch in range(1, int(settings["epochs"]) + 1):
        started = time.perf_counter()
        row = {"epoch": epoch}
        for name, (features, labels) in tensors.items():
            training = name == "train"
            model.train(training)
            order = torch.randperm(len(labels), generator=generator, device=device) if training else None
            total = torch.zeros((), device=device)
            with torch.set_grad_enabled(training):
                for start in range(0, len(labels), int(settings["batch_size"])):
                    stop = min(start + int(settings["batch_size"]), len(labels))
                    index = order[start:stop] if training else slice(start, stop)
                    loss = F.cross_entropy(model(features[index]), labels[index])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite categorical efficiency loss")
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_norm"]))
                        optimizer.step()
                    total += loss.detach() * (stop - start)
            row[f"{name}_loss"] = float(total / len(labels))
        row["seconds"] = time.perf_counter() - started
        history.append(row)
        print(f"epoch={epoch:02d} train={row['train_loss']:.6f} "
              f"validation={row['validation_loss']:.6f} seconds={row['seconds']:.2f}", flush=True)
        if row["validation_loss"] < best_loss - 1e-6:
            best_loss, stale, best_epoch = row["validation_loss"], 0, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(settings["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError("No valid efficiency checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch
