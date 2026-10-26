#!/usr/bin/env python3
"""Predict P(C_h|x_h,s_h,T=1) from generated-only CSV rows, not trigger rates."""

import argparse
from pathlib import Path

import pandas as pd

from forwardfm_hadron.inference import predict_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    predictions = predict_checkpoint(args.checkpoint, pd.read_csv(args.input), args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Wrote {len(predictions)} trigger-conditioned hadron probabilities")
