# LLM-based Energy Consumption Forecasting with Economic Insights (Flask)

End-to-end final-year project:

- Upload/select CSV or TXT datasets (TXT: delimiters + log-style parsing)
- Preprocess time-series (lags, rolling stats, seasonality)
- **Train LSTM from the terminal** (stable on HPC; avoids Flask timeouts)
- **Use the web UI** to select a dataset, **run forecast**, and explore dashboards
- Anomalies (Isolation Forest), economic correlations, optional Hugging Face LLM explanations

## Project structure

```
my_llm_project/
  app.py                 # Flask UI (no training here)
  train.py               # CLI: train LSTM
  requirements.txt
  data/                  # your CSV/TXT files
  models/
    lstm_model.py
    artifacts/           # *.pt, *.meta.json, *.future.csv
  templates/
  static/
  utils/
    dataset_io.py
    cpu_config.py
    train_job.py
    ...
```

## Workflow

1. Put datasets in `data/` (or upload via the web UI).
2. **Train on the login node / batch job** (not in the browser):

   ```bash
   cd my_llm_project
   source .venv/bin/activate
   export TORCH_NUM_THREADS=16
   python3 train.py --dataset your_file.csv --epochs 20
   ```

3. Start the web app and **select the dataset** — the dropdown shows **“✓ LSTM checkpoint”** when `models/artifacts/<name>.pt` exists.
4. Open **Dashboard** → **Run forecast** to write `models/artifacts/<id>.future.csv` and refresh plots.

## Quickstart (local)

```bash
cd my_llm_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py --host 0.0.0.0 --port 5000
```

## HPC: offline Hugging Face cache + CPU threads

On air-gapped clusters, **download the model once** on a machine with internet, copy the cache directory, then:

```bash
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_CACHE=/path/to/hf_cache
export TRANSFORMERS_OFFLINE=1
# Optional — skip LLM entirely:
export DISABLE_LLM=1
```

PyTorch CPU threading (e.g. **16 cores, 32 GB RAM**):

```bash
export TORCH_NUM_THREADS=16
export OMP_NUM_THREADS=16
```

These are applied automatically in `utils/cpu_config.py` (defaults tuned for 16 threads).

## OPSD + UCI household datasets

**Suggested filenames in `data/` (two OPSD CSVs):**

| File in `data/` | Typical source |
|-----------------|----------------|
| `dataset1.csv` | OPSD `household_data_60min_singleindex.csv` |
| `dataset2.csv` | OPSD `household_data_15min_singleindex.csv` |

See `data/README_DATASETS.txt`. Copy real CSV files into `data/` (not Windows path strings).

You can still upload **`.txt`** via the web UI if you add another source later; the default demo uses **two CSVs only**.

These file types are supported directly (copy into `data/`; Windows paths are fine on your PC, but the app on Linux/HPC expects files under `my_llm_project/data/`):

| File | Origin | Behaviour |
|------|--------|------------|
| `household_data_15min_singleindex.csv` | OPSD household | Timestamps: `utc_timestamp` / `cet_cest_timestamp` or first column; energy: columns matching `grid_import` / consumption patterns, or set `ENERGY_COLUMN`. |
| `household_data_60min_singleindex.csv` | OPSD household | Same, hourly. |
| `*.txt` (optional) | e.g. UCI | Semicolon logs supported if you add a `.txt` file. |

If the wrong series is selected on a wide OPSD table, force the column:

```bash
export ENERGY_COLUMN=DE_KN_industrial1_grid_import
python3 train.py --dataset household_data_60min_singleindex.csv
```

After parser updates, refresh cached normalised CSVs:

```bash
rm -f data/*.processed.csv
```

## Main routes

- `GET /` — home
- `GET/POST /upload` — upload dataset
- `GET/POST /select_dataset` — choose active dataset (shows training status)
- `POST /predict` — horizon forecast using saved checkpoint (UI button)
- `GET /dashboard` — charts + LLM + economics

Training is **not** exposed as a web route; use `python3 train.py`.

## Briefing / “LLM” panel (dashboard)

1. **Data-driven briefing** is always built from your real metrics (trend %, anomalies, validation MAE/RMSE, horizon, etc.) — no hard-coded dataset paths.
2. Optionally, a **very short** Hugging Face generation step may append commentary. Small base models like **DistilGPT-2** often repeat or hallucinate, so degenerate text is **dropped** automatically.
3. For **only** the grounded text: `export FAITHFUL_ONLY=1` before starting Flask.

Your **Windows paths are not in the code**. Copy OPSD/UCI files into `data/` on the machine where the app runs; the parser recognises those *formats*.

## Notes

- Checkpoints: `models/artifacts/<dataset_id>.{pt,scaler.joblib,meta.json}`
- `meta.json` includes `trained_at`, `lookback`, `horizon`, and validation metrics.
