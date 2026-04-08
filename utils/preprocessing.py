from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessResult:
    df: pd.DataFrame
    df_unscaled: pd.DataFrame
    feature_cols: List[str]
    target_col: str
    scaler: StandardScaler


def preprocess_timeseries(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    extra_feature_cols: Optional[List[str]] = None,
    lags: Tuple[int, ...] = (1, 2, 3, 24),
    rolling_windows: Tuple[int, ...] = (3, 7, 24),
) -> PreprocessResult:
    df = df.copy()

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=False)
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[target_col] = df[target_col].interpolate(limit_direction="both")

    # Optional features
    extra_feature_cols = extra_feature_cols or []
    for c in extra_feature_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").interpolate(limit_direction="both")

    # Time features (simple seasonal indicators)
    df["hour"] = df[timestamp_col].dt.hour.astype(float)
    df["dayofweek"] = df[timestamp_col].dt.dayofweek.astype(float)
    df["month"] = df[timestamp_col].dt.month.astype(float)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    # Lag + rolling features for target
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    for w in rolling_windows:
        df[f"{target_col}_roll_mean_{w}"] = df[target_col].rolling(w).mean()
        df[f"{target_col}_roll_std_{w}"] = df[target_col].rolling(w).std()

    # Build feature list
    base_features = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month",
    ]
    lag_features = [f"{target_col}_lag_{lag}" for lag in lags]
    roll_features = [f"{target_col}_roll_mean_{w}" for w in rolling_windows] + [
        f"{target_col}_roll_std_{w}" for w in rolling_windows
    ]

    used_extra = [c for c in extra_feature_cols if c in df.columns]
    feature_cols = base_features + used_extra + lag_features + roll_features

    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    df_unscaled = df.copy()

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].values)

    return PreprocessResult(
        df=df,
        df_unscaled=df_unscaled,
        feature_cols=feature_cols,
        target_col=target_col,
        scaler=scaler,
    )


def make_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lookback: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: (N, lookback, n_features)
      y: (N,)
    """
    X_list = []
    y_list = []

    feats = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)

    for i in range(lookback, len(df)):
        X_list.append(feats[i - lookback : i])
        y_list.append(target[i])

    if not X_list:
        raise ValueError("Not enough data after preprocessing to build LSTM sequences.")

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.float32)

