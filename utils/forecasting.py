from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.lstm_model import LSTMForecaster


@dataclass
class TrainArtifacts:
    model: LSTMForecaster
    device: str
    metrics: Dict[str, float]


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 12,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> TrainArtifacts:
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = np.arange(n)
    split = int(n * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    dev = _device()
    torch.manual_seed(seed)

    model = LSTMForecaster(n_features=X.shape[-1], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    model.to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    def batches(Xb: np.ndarray, yb: np.ndarray):
        order = rng.permutation(len(Xb))
        for i in range(0, len(order), batch_size):
            j = order[i : i + batch_size]
            yield Xb[j], yb[j]

    for _ in range(epochs):
        model.train()
        for xb, yb in batches(X_train, y_train):
            xb_t = torch.from_numpy(xb).to(dev)
            yb_t = torch.from_numpy(yb).to(dev)
            pred = model(xb_t)
            loss = loss_fn(pred, yb_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(torch.from_numpy(X_val).to(dev)).cpu().numpy()

    metrics = {
        "val_mae": float(mean_absolute_error(y_val, val_pred)),
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
    }
    return TrainArtifacts(model=model, device=dev, metrics=metrics)


def predict_one_step(model: LSTMForecaster, X: np.ndarray) -> np.ndarray:
    dev = _device()
    model = model.to(dev)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(dev)).cpu().numpy()
    return pred


def save_artifacts(
    artifacts_dir: str,
    dataset_id: str,
    model: LSTMForecaster,
    scaler,
    meta: Dict,
) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, f"{dataset_id}.pt")
    scaler_path = os.path.join(artifacts_dir, f"{dataset_id}.scaler.joblib")
    meta_path = os.path.join(artifacts_dir, f"{dataset_id}.meta.json")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_artifacts(
    artifacts_dir: str,
    dataset_id: str,
    n_features: int,
) -> Tuple[Optional[LSTMForecaster], Optional[object], Optional[Dict]]:
    model_path = os.path.join(artifacts_dir, f"{dataset_id}.pt")
    scaler_path = os.path.join(artifacts_dir, f"{dataset_id}.scaler.joblib")
    meta_path = os.path.join(artifacts_dir, f"{dataset_id}.meta.json")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(meta_path)):
        return None, None, None

    model = LSTMForecaster(n_features=n_features)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, scaler, meta


def infer_frequency(timestamps: pd.Series) -> pd.Timedelta:
    ts = pd.to_datetime(timestamps).sort_values()
    if len(ts) < 3:
        return pd.Timedelta(hours=1)
    diffs = ts.diff().dropna()
    if diffs.empty:
        return pd.Timedelta(hours=1)
    # median is robust
    return diffs.median()


def _build_features_unscaled(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    extra_feature_cols: List[str],
    lags: Tuple[int, ...],
    rolling_windows: Tuple[int, ...],
) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=False)
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").interpolate(limit_direction="both")
    for c in extra_feature_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").interpolate(limit_direction="both")

    df["hour"] = df[timestamp_col].dt.hour.astype(float)
    df["dayofweek"] = df[timestamp_col].dt.dayofweek.astype(float)
    df["month"] = df[timestamp_col].dt.month.astype(float)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    for w in rolling_windows:
        df[f"{target_col}_roll_mean_{w}"] = df[target_col].rolling(w).mean()
        df[f"{target_col}_roll_std_{w}"] = df[target_col].rolling(w).std()

    base_features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month"]
    used_extra = [c for c in extra_feature_cols if c in df.columns]
    lag_features = [f"{target_col}_lag_{lag}" for lag in lags]
    roll_features = [f"{target_col}_roll_mean_{w}" for w in rolling_windows] + [
        f"{target_col}_roll_std_{w}" for w in rolling_windows
    ]
    feature_cols = base_features + used_extra + lag_features + roll_features
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    return df, feature_cols


def forecast_future(
    model: LSTMForecaster,
    scaler,
    history_df_unscaled: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    extra_feature_cols: List[str],
    lookback: int,
    horizon: int,
    lags: Tuple[int, ...] = (1, 2, 3, 24),
    rolling_windows: Tuple[int, ...] = (3, 7, 24),
) -> pd.DataFrame:
    """
    Iterative multi-step forecast by appending predicted target values and
    recomputing engineered features, then scaling with the saved scaler.
    """
    freq = infer_frequency(history_df_unscaled[timestamp_col])

    work = history_df_unscaled[[timestamp_col, target_col] + [c for c in extra_feature_cols if c in history_df_unscaled.columns]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce", utc=False)
    work = work.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)

    # Use last known extras for the future if present.
    last_extras = {c: float(work[c].dropna().iloc[-1]) for c in extra_feature_cols if c in work.columns and not work[c].dropna().empty}

    preds = []
    for _ in range(horizon):
        feat_df, feature_cols = _build_features_unscaled(
            work, timestamp_col, target_col, extra_feature_cols, lags=lags, rolling_windows=rolling_windows
        )
        if len(feat_df) < lookback:
            raise ValueError("Not enough history to forecast with the chosen lookback.")

        X_last = feat_df[feature_cols].values.astype(np.float32)
        X_last_scaled = scaler.transform(X_last)
        X_seq = X_last_scaled[-lookback:].reshape(1, lookback, -1).astype(np.float32)

        y_hat = float(predict_one_step(model, X_seq)[0])
        preds.append(y_hat)

        next_ts = work[timestamp_col].iloc[-1] + freq
        row = {timestamp_col: next_ts, target_col: y_hat}
        for c, v in last_extras.items():
            row[c] = v
        work = pd.concat([work, pd.DataFrame([row])], ignore_index=True)

    future_ts = pd.date_range(start=work[timestamp_col].iloc[-horizon], periods=horizon, freq=freq)
    return pd.DataFrame({timestamp_col: future_ts, "forecast": preds})

