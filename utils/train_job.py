"""
LSTM training job (CLI + optional programmatic use). Training is not run from the Flask UI.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch

from utils.cpu_config import configure_cpu_threads
from utils.dataset_io import ARTIFACTS_DIR, dataset_id, load_processed_dataset
from utils.forecasting import save_artifacts, train_lstm
from utils.preprocessing import make_lstm_sequences, preprocess_timeseries


def run_training(
    raw_dataset_name: str,
    *,
    epochs: Optional[int] = None,
    lookback: int = 24,
    horizon: Optional[int] = None,
    hidden_size: int = 64,
    num_layers: int = 2,
) -> Dict[str, Any]:
    """
    Train LSTM on a file already under data/ (e.g. dataset1.csv).
    Saves checkpoints under models/artifacts/<dataset_id>.*
    """
    configure_cpu_threads()
    epochs = epochs if epochs is not None else int(os.environ.get("LSTM_EPOCHS", "12"))
    horizon = horizon if horizon is not None else int(os.environ.get("FORECAST_HORIZON", "24"))

    df, _meta = load_processed_dataset(raw_dataset_name)
    extra_cols = [c for c in ["temperature", "humidity", "price", "demand"] if c in df.columns]
    prep = preprocess_timeseries(df, timestamp_col="timestamp", target_col="energy", extra_feature_cols=extra_cols)
    X, y = make_lstm_sequences(prep.df, prep.feature_cols, prep.target_col, lookback=lookback)

    artifacts = train_lstm(
        X,
        y,
        epochs=epochs,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    did = dataset_id(raw_dataset_name)

    meta: Dict[str, Any] = {
        "dataset": raw_dataset_name,
        "timestamp_col": "timestamp",
        "energy_col": "energy",
        "extra_cols": extra_cols,
        "lookback": lookback,
        "horizon": horizon,
        "feature_cols": prep.feature_cols,
        "metrics": artifacts.metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "torch": {
            "cuda_available": bool(torch.cuda.is_available()),
            "num_threads": int(os.environ.get("TORCH_NUM_THREADS", "16")),
        },
    }

    save_artifacts(
        str(ARTIFACTS_DIR),
        dataset_id=did,
        model=artifacts.model,
        scaler=prep.scaler,
        meta=meta,
    )

    return {"dataset_id": did, "metrics": artifacts.metrics, "meta_path": str(ARTIFACTS_DIR / f"{did}.meta.json")}
