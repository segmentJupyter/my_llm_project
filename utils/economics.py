from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class EconInsights:
    available: bool
    summary: str
    details: Dict[str, float]


def _time_index(df: pd.DataFrame, timestamp_col: str) -> np.ndarray:
    t = pd.to_datetime(df[timestamp_col], errors="coerce")
    t = t.astype("int64") // 10**9  # seconds
    t = (t - t.min()).astype(float)
    return t.values.reshape(-1, 1)


def economic_insights(
    df: pd.DataFrame,
    timestamp_col: str,
    energy_col: str,
    price_col: str = "price",
    demand_col: str = "demand",
) -> EconInsights:
    cols = set(df.columns.str.lower())
    has_price = price_col in cols
    has_demand = demand_col in cols

    if not (has_price or has_demand):
        return EconInsights(available=False, summary="No economic columns (price/demand) detected.", details={})

    # Map to actual column names
    lower_to_orig = {c.lower(): c for c in df.columns}
    price_col_o = lower_to_orig.get(price_col)
    demand_col_o = lower_to_orig.get(demand_col)

    work = df.copy()
    X = _time_index(work, timestamp_col)

    details: Dict[str, float] = {}
    lines = []

    if has_price and price_col_o:
        y = pd.to_numeric(work[price_col_o], errors="coerce").interpolate(limit_direction="both").values
        reg = LinearRegression().fit(X, y)
        slope = float(reg.coef_[0])
        details["price_trend_slope_per_sec"] = slope
        lines.append("Price shows an overall " + ("upward" if slope > 0 else "downward" if slope < 0 else "flat") + " trend.")

    if has_demand and demand_col_o:
        y = pd.to_numeric(work[demand_col_o], errors="coerce").interpolate(limit_direction="both").values
        reg = LinearRegression().fit(X, y)
        slope = float(reg.coef_[0])
        details["demand_trend_slope_per_sec"] = slope
        lines.append("Demand shows an overall " + ("upward" if slope > 0 else "downward" if slope < 0 else "flat") + " trend.")

    # Simple association with energy
    energy = pd.to_numeric(work[energy_col], errors="coerce").interpolate(limit_direction="both")
    if has_price and price_col_o:
        price = pd.to_numeric(work[price_col_o], errors="coerce").interpolate(limit_direction="both")
        corr = float(price.corr(energy))
        details["corr_price_energy"] = corr
        lines.append(f"Price–energy correlation: {corr:+.2f}.")
    if has_demand and demand_col_o:
        demand = pd.to_numeric(work[demand_col_o], errors="coerce").interpolate(limit_direction="both")
        corr = float(demand.corr(energy))
        details["corr_demand_energy"] = corr
        lines.append(f"Demand–energy correlation: {corr:+.2f}.")

    summary = " ".join(lines) if lines else "Economic columns detected, but insufficient data to summarize."
    return EconInsights(available=True, summary=summary, details=details)

