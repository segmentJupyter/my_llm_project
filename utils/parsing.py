import io
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class ParsedDataset:
    df: pd.DataFrame
    timestamp_col: str
    energy_col: str


_TS_CANDIDATES = (
    "timestamp",
    "utc_timestamp",
    "cet_cest_timestamp",
    "date",
    "datetime",
    "time",
)
# Substrings matched against lowercase column names
_ENERGY_CANDIDATES = (
    "energy",
    "consumption",
    "global_active_power",  # UCI — before generic "power" so we don’t pick reactive power
    "active_power",
    "load",
    "kwh",
    "power",
    "usage",
    "grid_import",
    "import",
)


def _find_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c in lower:
            return lower[c]
    for c in candidates:
        for k, orig in lower.items():
            if c in k:
                return orig
    return None


def _na_values() -> List[str]:
    return ["?", "NA", "NaN", "nan", ""]


def _read_csv_standard(path: str) -> pd.DataFrame:
    return pd.read_csv(path, na_values=_na_values(), low_memory=False)


def _read_csv_index_first(path: str) -> pd.DataFrame:
    """OPSD / exports where the first column is the datetime index with no header name."""
    df = pd.read_csv(path, index_col=0, parse_dates=True, na_values=_na_values(), low_memory=False)
    return df.reset_index()


def _combine_uci_date_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    UCI Household Power Consumption: Date + Time columns (often dd/mm/yyyy + HH:MM:SS).
    """
    df = df.copy()
    lower = {str(c).lower().strip(): c for c in df.columns}
    d_col = lower.get("date")
    t_col = lower.get("time")
    if not d_col or not t_col:
        return df
    combined = df[d_col].astype(str).str.strip() + " " + df[t_col].astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(combined, dayfirst=True, errors="coerce")
    return df


def _first_col_looks_like_timestamp(df: pd.DataFrame) -> bool:
    if df.empty or len(df.columns) < 2:
        return False
    c0 = df.columns[0]
    if str(c0).lower() in ("unnamed: 0", "index"):
        s = df[c0]
        parsed = pd.to_datetime(s.head(500), errors="coerce")
        return float(parsed.notna().mean()) > 0.85
    return False


def _promote_first_column_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c0 = df.columns[0]
    df["timestamp"] = pd.to_datetime(df[c0], errors="coerce")
    return df


def _pick_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Prefer combined UCI timestamp; then standard names; then fall back."""
    if "timestamp" in df.columns:
        s = pd.to_datetime(df["timestamp"], errors="coerce")
        if s.notna().sum() >= max(10, int(0.4 * len(df))):
            return "timestamp"
    ts = _find_column(df, _TS_CANDIDATES) or _find_column(df, ("ts",))
    return ts


def _pick_opsd_style_energy(df: pd.DataFrame, exclude: Optional[str] = None) -> Optional[str]:
    """
    OPSD household CSVs: many columns like DE_KN_*_grid_import.
    Prefer any *grid_import* / *consumption* numeric column with good coverage.
    """
    candidates = []
    for c in df.columns:
        if exclude and c == exclude:
            continue
        lc = str(c).lower()
        if any(k in lc for k in ("grid_import", "consumption", "load_kwh", "residential")):
            candidates.append(c)
    best = None
    best_score = -1.0
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        cov = float(s.notna().mean())
        if cov < 0.3:
            continue
        std = float(s.dropna().std()) if s.dropna().shape[0] > 2 else 0.0
        score = std * cov
        if score > best_score:
            best = c
            best_score = score
    return best


def _guess_numeric_target(df: pd.DataFrame, exclude: Optional[str]) -> Optional[str]:
    best = None
    best_score = None

    for c in df.columns:
        if exclude is not None and c == exclude:
            continue

        s = pd.to_numeric(df[c], errors="coerce")
        non_na = float(s.notna().mean())
        if non_na < 0.7:
            continue

        std = float(s.dropna().std()) if s.dropna().shape[0] > 2 else 0.0
        score = std + 0.1 * non_na
        if best_score is None or score > best_score:
            best = c
            best_score = score

    return best


def _try_parse_delimited_text(text: str, prefer_semicolon_first: bool = False) -> Optional[pd.DataFrame]:
    seps = [",", "\t", ";", "|", r"\s+"]
    if prefer_semicolon_first:
        seps = [";", ",", "\t", "|", r"\s+"]

    best = None
    best_score = -1

    for sep in seps:
        try:
            df = pd.read_csv(
                io.StringIO(text),
                sep=sep,
                engine="python",
                comment="#",
                na_values=_na_values(),
            )
        except Exception:
            continue

        if df is None or df.empty:
            continue

        ncols = df.shape[1]
        if ncols < 2:
            continue
        sample = df.head(50)
        nan_ratio = float(sample.isna().mean().mean()) if not sample.empty else 1.0
        score = min(ncols, 10) - nan_ratio
        if score > best_score:
            best = df
            best_score = score

    return best


_LOG_TS_PATTERNS = [
    r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})",
    r"(?P<ts>\d{4}-\d{2}-\d{2})",
]


def _try_parse_logs(text: str) -> Optional[pd.DataFrame]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    rows = []
    ts_re = re.compile("|".join(_LOG_TS_PATTERNS))
    num_re = re.compile(r"(-?\d+(?:\.\d+)?)")
    energy_kv_re = re.compile(r"(?:energy|consumption|load|kwh)\s*=\s*(?P<v>-?\d+(?:\.\d+)?)", re.IGNORECASE)

    for ln in lines:
        m = ts_re.search(ln)
        if not m:
            continue
        ts = m.group("ts")
        kv = energy_kv_re.search(ln)
        if kv:
            rows.append((ts, float(kv.group("v"))))
            continue

        nums = [float(x) for x in num_re.findall(ln)]
        if not nums:
            continue
        rows.append((ts, nums[-1]))

    if len(rows) < 10:
        return None

    return pd.DataFrame(rows, columns=["timestamp", "energy"])


def _resolve_energy_column(df: pd.DataFrame, ts_col: str) -> Optional[str]:
    # Explicit override (useful on OPSD when many series exist)
    forced = os.environ.get("ENERGY_COLUMN", "").strip()
    if forced and forced in df.columns:
        return forced

    energy_col = _find_column(df, _ENERGY_CANDIDATES)
    if energy_col is not None:
        return energy_col
    energy_col = _pick_opsd_style_energy(df, exclude=ts_col)
    if energy_col is not None:
        return energy_col
    return _guess_numeric_target(df, exclude=ts_col)


def parse_dataset_file(path: str) -> ParsedDataset:
    """
    Load CSV or TXT and locate timestamp + energy columns.

    Supported out of the box:
    - OPSD household *_singleindex.csv (utc_timestamp / CET + DE_*_grid_import columns)
    - UCI household_power_consumption.txt (Date;Time;Global_active_power;... semicolon)
    """
    ext = os.path.splitext(path)[1].lower()
    basename = os.path.basename(path).lower()

    if ext == ".csv":
        df = _read_csv_standard(path)
        df = _combine_uci_date_time(df)

        ts_col = _pick_timestamp_column(df)
        if ts_col is None and _first_col_looks_like_timestamp(df):
            df = _promote_first_column_timestamp(df)
            ts_col = "timestamp"
        if ts_col is None:
            try:
                df_alt = _read_csv_index_first(path)
                df_alt = _combine_uci_date_time(df_alt)
                ts_col = _pick_timestamp_column(df_alt)
                if ts_col is None and len(df_alt.columns) >= 1:
                    c0 = df_alt.columns[0]
                    if pd.api.types.is_datetime64_any_dtype(df_alt[c0]):
                        df_alt = df_alt.rename(columns={c0: "timestamp"})
                        ts_col = "timestamp"
                if ts_col is not None:
                    df = df_alt
            except Exception:
                pass

        if ts_col is None:
            raise ValueError(f"Could not find a timestamp column. Columns: {list(df.columns)}")
        energy_col = _resolve_energy_column(df, ts_col)
        if energy_col is None:
            raise ValueError(
                "Could not find/guess an energy column. "
                f"Columns: {list(df.columns)}. "
                "For OPSD with many series, set ENERGY_COLUMN to the exact column name, "
                "or pick one containing grid_import / consumption."
            )
        df = df.copy()
        return ParsedDataset(df=df, timestamp_col=ts_col, energy_col=energy_col)

    # TXT: UCI often semicolon — try that first when name suggests it
    prefer_sc = "household" in basename and "power" in basename
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    df = _try_parse_delimited_text(text, prefer_semicolon_first=prefer_sc or ";" in text[:2000])
    if df is None:
        df = _try_parse_logs(text)
    if df is None:
        raise ValueError("Could not parse TXT file (neither delimited nor log-like).")

    df = _combine_uci_date_time(df)

    if df is None or df.empty:
        raise ValueError("Parsed dataset is empty.")

    ts_col = _pick_timestamp_column(df)
    if ts_col is None:
        raise ValueError(f"Could not find a timestamp column. Columns: {list(df.columns)}")
    energy_col = _resolve_energy_column(df, ts_col)
    if energy_col is None:
        raise ValueError(
            "Could not find/guess an energy column. "
            f"Columns: {list(df.columns)}. "
            "Set ENERGY_COLUMN if needed."
        )

    df = df.copy()
    return ParsedDataset(df=df, timestamp_col=ts_col, energy_col=energy_col)
