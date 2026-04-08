#!/usr/bin/env python3
"""
Train the LSTM on a dataset in data/ from the terminal (recommended on HPC — avoids Flask timeouts).

Examples:
  # Offline Hugging Face cache (download model once elsewhere, then point here):
  export HF_HOME=/path/to/hf_cache
  export TRANSFORMERS_OFFLINE=1

  # Use all 16 CPU threads for PyTorch:
  export TORCH_NUM_THREADS=16

  python3 train.py --dataset dataset1.csv --epochs 20
  python3 train.py --dataset my_power.csv --lookback 48 --horizon 72
"""
from __future__ import annotations

import argparse
import json
import sys

from utils.train_job import run_training


def main() -> int:
    p = argparse.ArgumentParser(description="Train LSTM energy forecaster (CLI only; UI does not train).")
    p.add_argument("--dataset", required=True, help="Filename in data/, e.g. dataset1.csv")
    p.add_argument("--epochs", type=int, default=None, help="Override LSTM_EPOCHS env default")
    p.add_argument("--lookback", type=int, default=24)
    p.add_argument("--horizon", type=int, default=None, help="Steps for later UI 'Predict future' (saved in meta)")
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    args = p.parse_args()

    try:
        out = run_training(
            args.dataset,
            epochs=args.epochs,
            lookback=args.lookback,
            horizon=args.horizon,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print("Training finished.")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
