from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from plotly.utils import PlotlyJSONEncoder

from utils.anomaly import detect_anomalies
from utils.cpu_config import configure_cpu_threads
from utils.dataset_io import (
    ARTIFACTS_DIR,
    DATA_DIR,
    dataset_id,
    has_trained_model,
    list_raw_datasets,
    load_processed_dataset,
    read_training_meta,
    safe_name,
)
from utils.economics import economic_insights
from utils.forecasting import (
    forecast_future,
    load_artifacts,
    predict_one_step,
)
from utils.llm_module import generate_briefing
from utils.preprocessing import make_lstm_sequences, preprocess_timeseries

configure_cpu_threads()

ALLOWED_EXT = {".csv", ".txt"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# Plotly “dark neon” theme for a more distinctive dashboard
_PLOT_TEMPLATE = "plotly_dark"
_PLOT_FONT = dict(family="system-ui, Segoe UI, sans-serif", color="#e8e8f0")


def _active_dataset() -> Optional[str]:
    ds = session.get("active_dataset")
    if ds and (DATA_DIR / ds).exists():
        return ds
    ds_list = list_raw_datasets()
    if ds_list:
        session["active_dataset"] = ds_list[0]
        return ds_list[0]
    return None


def _datasets_with_train_status() -> List[Dict[str, Any]]:
    out = []
    for name in list_raw_datasets():
        out.append({"name": name, "trained": has_trained_model(name)})
    return out


def _plot_json(fig: go.Figure) -> str:
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def _empty_fig(title: str) -> go.Figure:
    return (
        go.Figure()
        .update_layout(
            template=_PLOT_TEMPLATE,
            title=dict(text=title, font=_PLOT_FONT),
            paper_bgcolor="#12141c",
            plot_bgcolor="#12141c",
            font=_PLOT_FONT,
        )
        .add_annotation(
            text="No data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#888"),
        )
    )


@app.get("/")
def index():
    return render_template(
        "index.html",
        active_dataset=_active_dataset(),
        message=None,
    )


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html", error=None, uploaded_name=None)

    if "file" not in request.files:
        return render_template("upload.html", error="No file provided.", uploaded_name=None)

    f = request.files["file"]
    if not f.filename:
        return render_template("upload.html", error="Empty filename.", uploaded_name=None)

    name = safe_name(f.filename)
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_EXT:
        return render_template("upload.html", error="Only CSV and TXT are supported.", uploaded_name=None)

    DATA_DIR.mkdir(exist_ok=True, parents=True)
    path = DATA_DIR / name
    f.save(path)

    try:
        df, meta = load_processed_dataset(name)
    except Exception as e:
        return render_template("upload.html", error=str(e), uploaded_name=None)

    session["active_dataset"] = name
    preview_table = df.head(10).to_html(classes="table table-sm table-striped", index=False)

    return render_template(
        "upload.html",
        error=None,
        uploaded_name=name,
        ts_col=meta["timestamp_col"],
        energy_col=meta["energy_col"],
        preview_table=preview_table,
    )


@app.route("/select_dataset", methods=["GET", "POST"])
def select_dataset():
    datasets = _datasets_with_train_status()
    if not datasets:
        return render_template(
            "select_dataset.html",
            datasets=[],
            active_dataset=None,
            message="No datasets found. Upload first.",
        )

    if request.method == "POST":
        ds = request.form.get("dataset")
        if ds and (DATA_DIR / ds).exists():
            session["active_dataset"] = ds
            return redirect(url_for("dashboard"))

    return render_template(
        "select_dataset.html",
        datasets=datasets,
        active_dataset=_active_dataset(),
        message=None,
    )


@app.post("/predict")
def predict():
    """UI-only: multi-step forecast using trained checkpoint (train via `python3 train.py`)."""
    ds = _active_dataset()
    if not ds:
        return jsonify({"error": "No dataset selected."}), 400

    df, _ = load_processed_dataset(ds)
    extra_cols = [c for c in ["temperature", "humidity", "price", "demand"] if c in df.columns]
    tmeta = read_training_meta(ds)
    lookback = int((tmeta or {}).get("lookback", 24))
    prep = preprocess_timeseries(df, timestamp_col="timestamp", target_col="energy", extra_feature_cols=extra_cols)
    X, y = make_lstm_sequences(prep.df, prep.feature_cols, prep.target_col, lookback=lookback)

    did = dataset_id(ds)
    model, scaler, meta = load_artifacts(str(ARTIFACTS_DIR), dataset_id=did, n_features=X.shape[-1])
    if meta and int(meta.get("lookback", lookback)) != lookback:
        lookback = int(meta["lookback"])
        X, y = make_lstm_sequences(prep.df, prep.feature_cols, prep.target_col, lookback=lookback)
        model, scaler, meta = load_artifacts(str(ARTIFACTS_DIR), dataset_id=did, n_features=X.shape[-1])
    if model is None or scaler is None or meta is None:
        return jsonify({"error": "No trained model for this dataset. Run: python3 train.py --dataset " + ds}), 400

    horizon = int(meta.get("horizon", 24))
    lb = int(meta.get("lookback", 24))
    future = forecast_future(
        model=model,
        scaler=scaler,
        history_df_unscaled=prep.df_unscaled[["timestamp", "energy"] + extra_cols],
        timestamp_col="timestamp",
        target_col="energy",
        extra_feature_cols=extra_cols,
        lookback=lb,
        horizon=horizon,
    )
    out_path = ARTIFACTS_DIR / f"{did}.future.csv"
    future.to_csv(out_path, index=False)
    return jsonify({"ok": True, "horizon": horizon, "saved": str(out_path)})


@app.get("/dashboard")
def dashboard():
    ds = _active_dataset()
    empty = _empty_fig

    def blank_payload(err: Optional[str] = None, msg: str = "Select a dataset"):
        return render_template(
            "dashboard.html",
            active_dataset=None,
            error=err,
            model_status=msg,
            trained_hint=True,
            metrics=None,
            trained_at=None,
            kpis={},
            preview_table=None,
            has_future=False,
            plot1_json=_plot_json(_empty_fig("Actual vs predicted")),
            plot2_json=_plot_json(_empty_fig("Anomalies")),
            plot3_json=_plot_json(_empty_fig("Residuals")),
            plot4_json=_plot_json(_empty_fig("Hour-of-day profile")),
            plot5_json=_plot_json(_empty_fig("Weekly pattern")),
            plot6_json=_plot_json(_empty_fig("Distribution")),
            llm_text="Select a dataset and ensure you trained the LSTM from the terminal.",
            llm_used=False,
            llm_model="",
            econ_summary="",
            econ_details={},
        )

    if not ds:
        return blank_payload(None, "No dataset selected")

    try:
        df, _ = load_processed_dataset(ds)
    except Exception as e:
        return blank_payload(str(e), "Load error")

    preview_table = df.head(8).to_html(classes="table table-sm table-striped table-dark", index=False)
    extra_cols = [c for c in ["temperature", "humidity", "price", "demand"] if c in df.columns]

    tmeta = read_training_meta(ds)
    lookback = int((tmeta or {}).get("lookback", 24))

    prep = preprocess_timeseries(df, timestamp_col="timestamp", target_col="energy", extra_feature_cols=extra_cols)
    X, y = make_lstm_sequences(prep.df, prep.feature_cols, prep.target_col, lookback=lookback)

    did = dataset_id(ds)
    model, scaler, meta = load_artifacts(str(ARTIFACTS_DIR), dataset_id=did, n_features=X.shape[-1])
    if meta and int(meta.get("lookback", lookback)) != lookback:
        lookback = int(meta.get("lookback", lookback))
        X, y = make_lstm_sequences(prep.df, prep.feature_cols, prep.target_col, lookback=lookback)
        model, scaler, meta = load_artifacts(str(ARTIFACTS_DIR), dataset_id=did, n_features=X.shape[-1])

    has_model = model is not None and meta is not None
    trained_at = (tmeta or {}).get("trained_at") if has_model else None

    if has_model:
        model_status = "LSTM checkpoint loaded — run Predict future for horizon forecast"
    else:
        model_status = "No checkpoint — train from terminal: python3 train.py --dataset " + ds

    metrics_obj = None
    if meta and isinstance(meta.get("metrics"), dict):
        @dataclass
        class M:
            val_mae: float
            val_rmse: float

        metrics_obj = M(
            val_mae=float(meta["metrics"].get("val_mae", 0.0)),
            val_rmse=float(meta["metrics"].get("val_rmse", 0.0)),
        )

    y_pred = None
    if has_model:
        y_pred = predict_one_step(model, X)

    ts_full = pd.to_datetime(prep.df["timestamp"]).iloc[lookback:].reset_index(drop=True)
    y_true = pd.Series(y, name="actual").reset_index(drop=True)
    y_pred_s = pd.Series(y_pred, name="predicted").reset_index(drop=True) if y_pred is not None else None

    # --- Plot 1: actual vs predicted + optional future
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ts_full, y=y_true, name="Actual", line=dict(color="#5eead4", width=2)))
    if y_pred_s is not None:
        fig1.add_trace(go.Scatter(x=ts_full, y=y_pred_s, name="LSTM 1-step", line=dict(color="#fbbf24", width=1.8)))
    future_path = ARTIFACTS_DIR / f"{did}.future.csv"
    has_future = future_path.exists()
    if has_future:
        fut = pd.read_csv(future_path)
        if "timestamp" in fut.columns and "forecast" in fut.columns:
            fut["timestamp"] = pd.to_datetime(fut["timestamp"])
            fig1.add_trace(
                go.Scatter(
                    x=fut["timestamp"],
                    y=fut["forecast"],
                    name="Forecast",
                    line=dict(color="#a78bfa", width=2, dash="dot"),
                )
            )
    fig1.update_layout(
        template=_PLOT_TEMPLATE,
        title="Energy: actual vs model & forecast",
        paper_bgcolor="#12141c",
        plot_bgcolor="#1a1d28",
        font=_PLOT_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Time",
        yaxis_title="kWh / units",
    )

    # --- Anomalies
    energy_series = pd.to_numeric(prep.df_unscaled["energy"], errors="coerce").interpolate(limit_direction="both").values
    anom = detect_anomalies(energy_series, contamination=float(os.environ.get("ANOM_CONTAMINATION", "0.02")))
    anom_idx = np.where(anom.is_anomaly)[0]
    ts_u = pd.to_datetime(prep.df_unscaled["timestamp"])

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ts_u, y=energy_series, name="Energy", line=dict(color="#34d399", width=1.5)))
    if len(anom_idx) > 0:
        fig2.add_trace(
            go.Scatter(
                x=ts_u.iloc[anom_idx],
                y=energy_series[anom_idx],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#fb7185", size=9, symbol="x"),
            )
        )
    fig2.update_layout(
        template=_PLOT_TEMPLATE,
        title="Isolation Forest anomalies",
        paper_bgcolor="#12141c",
        plot_bgcolor="#1a1d28",
        font=_PLOT_FONT,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Time",
        yaxis_title="Energy",
    )

    # --- Residuals
    fig3 = go.Figure()
    if y_pred_s is not None:
        resid = (y_true - y_pred_s).values
        fig3.add_trace(go.Scatter(x=ts_full, y=resid, name="Residual", line=dict(color="#94a3b8", width=1)))
        fig3.add_hline(y=0, line_dash="dash", line_color="#64748b")
    fig3.update_layout(
        template=_PLOT_TEMPLATE,
        title="Residuals (actual − predicted)",
        paper_bgcolor="#12141c",
        plot_bgcolor="#1a1d28",
        font=_PLOT_FONT,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Time",
        yaxis_title="Error",
    )

    # --- Hour-of-day profile
    dfp = prep.df_unscaled.copy()
    dfp["timestamp"] = pd.to_datetime(dfp["timestamp"])
    dfp["hour"] = dfp["timestamp"].dt.hour
    hourly = dfp.groupby("hour")["energy"].mean().reset_index()
    fig4 = go.Figure(data=[go.Bar(x=hourly["hour"], y=hourly["energy"], marker_color="#38bdf8")])
    fig4.update_layout(
        template=_PLOT_TEMPLATE,
        title="Average load by hour of day",
        paper_bgcolor="#12141c",
        plot_bgcolor="#1a1d28",
        font=_PLOT_FONT,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Hour (0–23)",
        yaxis_title="Mean energy",
    )

    # --- Heatmap: day of week vs hour
    dfp["dow"] = dfp["timestamp"].dt.dayofweek
    try:
        pivot = dfp.pivot_table(index="dow", columns="hour", values="energy", aggfunc="mean")
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        y_labels = [days[int(i)] for i in pivot.index]
        fig5 = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[int(c) for c in pivot.columns],
                y=y_labels,
                colorscale="Viridis",
            )
        )
    except Exception:
        fig5 = _empty_fig("Weekly pattern")
    fig5.update_layout(
        template=_PLOT_TEMPLATE,
        title="Weekly rhythm — mean energy (dow × hour)",
        paper_bgcolor="#12141c",
        plot_bgcolor="#1a1d28",
        font=_PLOT_FONT,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Hour",
        yaxis_title="Day",
    )

    # --- Histogram
    fig6 = go.Figure()
    fig6.add_trace(go.Histogram(x=energy_series, nbinsx=40, marker_color="#22d3ee", opacity=0.85))
    fig6.update_layout(
        template=_PLOT_TEMPLATE,
        title="Energy distribution",
        paper_bgcolor="#12141c",
        plot_bgcolor="#1a1d28",
        font=_PLOT_FONT,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Energy",
        yaxis_title="Count",
    )

    # KPIs
    kpis: Dict[str, Any] = {
        "rows": len(df),
        "mean": float(np.nanmean(energy_series)),
        "std": float(np.nanstd(energy_series)),
        "min": float(np.nanmin(energy_series)),
        "max": float(np.nanmax(energy_series)),
        "anomalies": int(anom.is_anomaly.sum()),
        "anomaly_pct": float(100.0 * anom.is_anomaly.mean()),
    }
    if y_pred_s is not None:
        kpis["mae"] = float(np.mean(np.abs(y_true - y_pred_s)))

    econ = economic_insights(df, timestamp_col="timestamp", energy_col="energy")

    llm_text = ""
    llm_used = False
    llm_model = ""
    if not has_model:
        llm_text = "Train the LSTM from the terminal first, then refresh this page.\n\npython3 train.py --dataset " + ds
    else:
        last_n = min(24 * 7, len(prep.df_unscaled))
        recent = prep.df_unscaled.tail(last_n).copy()
        recent["energy"] = pd.to_numeric(recent["energy"], errors="coerce").interpolate(limit_direction="both")
        pct_change = float((recent["energy"].iloc[-1] - recent["energy"].iloc[0]) / max(1e-9, recent["energy"].iloc[0]) * 100.0)
        temp_change = None
        if "temperature" in recent.columns:
            t = pd.to_numeric(recent["temperature"], errors="coerce").interpolate(limit_direction="both")
            temp_change = float(t.iloc[-1] - t.iloc[0])
        anomalies_recent = int(anom.is_anomaly[-last_n:].sum())
        trend_word = (
            "rising"
            if (y_pred_s is not None and y_pred_s.tail(10).mean() > y_true.tail(10).mean())
            else "stable/declining"
        )
        vma = meta.get("metrics", {}) if meta else {}
        llm = generate_briefing(
            {
                "dataset": ds,
                "pct_change": pct_change,
                "temp_change": temp_change,
                "anomalies_recent": anomalies_recent,
                "trend_word": trend_word,
                "horizon": int(meta.get("horizon", 24)),
                "window_points": last_n,
                "anomaly_series_pct": float(kpis.get("anomaly_pct", 0.0)),
                "mean_energy": float(np.nanmean(energy_series)),
                "val_mae": f"{float(vma.get('val_mae', 0.0)):.4f}" if vma else "n/a",
                "val_rmse": f"{float(vma.get('val_rmse', 0.0)):.4f}" if vma else "n/a",
            }
        )
        llm_text = llm.text
        llm_used = llm.used_llm
        llm_model = llm.model_name

    return render_template(
        "dashboard.html",
        active_dataset=ds,
        error=None,
        model_status=model_status,
        trained_hint=not has_model,
        metrics=metrics_obj,
        trained_at=trained_at,
        kpis=kpis,
        preview_table=preview_table,
        has_future=has_future,
        plot1_json=_plot_json(fig1),
        plot2_json=_plot_json(fig2),
        plot3_json=_plot_json(fig3),
        plot4_json=_plot_json(fig4),
        plot5_json=_plot_json(fig5),
        plot6_json=_plot_json(fig6),
        llm_text=llm_text,
        llm_used=llm_used,
        llm_model=llm_model,
        econ_summary=econ.summary,
        econ_details=econ.details,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5000, type=int)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()

import os

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)