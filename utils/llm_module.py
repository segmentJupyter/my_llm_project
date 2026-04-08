"""
LLM / explanation layer.

DistilGPT-2 and similar *base* LMs are not instruction-tuned; long free-form prompts cause
repetition and hallucinations. We therefore:

1. Always build a **faithful briefing** from the numeric context (no hallucinated metrics).
2. Optionally append a **very short** generated sentence with strict decoding + repetition penalty;
   degenerate output is discarded automatically.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional

SMALL_CAUSAL_BASE = frozenset({"distilgpt2", "gpt2", "gpt2-medium"})


@dataclass
class LLMResult:
    used_llm: bool
    text: str
    model_name: str


_PIPE = None
_PIPE_MODEL = None


def _fallback_block() -> str:
    return (
        "Use the data-driven briefing above. "
        "If the neural commentary is missing, the server is in offline or disabled-LLM mode.\n"
    )


def build_faithful_briefing(ctx: Dict[str, Any]) -> str:
    """
    Grounded narrative from *actual* dashboard statistics (no HF model).
    """
    name = ctx.get("dataset", "selected series")
    pct = float(ctx.get("pct_change", 0.0))
    anomalies_recent = int(ctx.get("anomalies_recent", 0))
    trend = ctx.get("trend_word", "mixed")
    horizon = int(ctx.get("horizon", 24))
    last_n = int(ctx.get("window_points", 0))
    anomaly_series_pct = float(ctx.get("anomaly_series_pct", 0.0))
    val_mae = ctx.get("val_mae")
    val_rmse = ctx.get("val_rmse")
    mean_e = ctx.get("mean_energy")

    lines = [
        "=== Data-driven briefing (grounded in your loaded series) ===",
        "",
        f"• Dataset: {name}",
        f"• Recent analysis window: last ~{last_n} time steps.",
        f"• Change in consumption from start → end of that window: {pct:+.1f}%.",
    ]
    if mean_e is not None:
        lines.append(f"• Level check: approximate mean load in the processed series: {float(mean_e):.2f} (same units as your file).")

    tc = ctx.get("temp_change")
    if tc is not None:
        lines.append(f"• Temperature over the same window moved {float(tc):+.1f} °C (if available in your file).")

    lines.extend(
        [
            f"• Isolation Forest flagged {anomalies_recent} point(s) in this window as unusual.",
            f"• Overall, about {anomaly_series_pct:.2f}% of all points in the series are flagged as anomalies.",
            f"• LSTM validation quality: MAE={val_mae}, RMSE={val_rmse} (from training; lower is better).",
            f"• Short-term shape: recent model predictions vs actuals suggest trend is **{trend}**.",
            f"• Forecast button: generates the next **{horizon}** steps beyond the latest timestamp.",
            "",
            "What this usually means:",
        ]
    )

    if abs(pct) < 3:
        lines.append("  – Load is fairly stable over the window; small changes often reflect routine/daily mix.")
    elif pct > 5:
        lines.append("  – Demand increased noticeably; check weather, occupancy, holidays, or tariff/price if you have economic columns.")
    else:
        lines.append("  – Demand decreased somewhat; mild weather, reduced activity, or efficiency changes are common explanations.")

    if anomalies_recent > 0:
        lines.append(
            "  – Anomalies deserve a second look: verify sensors/metering, single-day events, or true demand spikes before trusting long forecasts."
        )
    else:
        lines.append("  – Few anomalies in-window suggests smoother behaviour; forecasts are more trustworthy under stable regimes.")

    lines.extend(["", "Next steps:", "  – Re-train after uploading a new file or changing resolution.", "  – Use “Run forecast” after training, then compare predicted curve to anomalies."])
    return "\n".join(lines)


def _is_degenerate(text: str) -> bool:
    if not text or len(text) < 20:
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    if len(lines) >= 4:
        top = Counter(lines).most_common(1)[0][1]
        if top >= 3:
            return True
    # Repeated n-gram in a single line
    words = text.split()
    if len(words) > 12:
        sample = " ".join(words[:24])
        rest = " ".join(words[24:])
        if sample in rest:
            return True
    # Gibberish / policy spam often contains "risks and risks"
    if text.lower().count("risks") > 4 or text.lower().count("aware of") > 3:
        return True
    return False


def _get_pipeline(model_name: str):
    global _PIPE, _PIPE_MODEL
    if _PIPE is not None and _PIPE_MODEL == model_name:
        return _PIPE

    if os.environ.get("DISABLE_LLM", "").strip() == "1":
        return None

    try:
        from transformers import pipeline

        _PIPE = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
        )
        _PIPE_MODEL = model_name
        return _PIPE
    except Exception:
        return None


def _short_neural_addon(model_name: str, compact_facts: str) -> Optional[str]:
    """
    One short continuation; strict anti-repetition. Not used for “research grade” truth — only flavor.
    """
    pipe = _get_pipeline(model_name)
    if pipe is None:
        return None

    base = model_name.split("/")[-1].lower()
    # Tiny base LMs: keep it trivial
    max_new = 45 if base in SMALL_CAUSAL_BASE else 90

    prompt = (
        "Facts about household or grid electricity (numeric summary):\n"
        f"{compact_facts}\n"
        "In one or two short sentences, give plausible everyday reasons for the trend. "
        "Do not invent numbers. Do not repeat the same sentence.\nAnswer:"
    )

    gen_kw: Dict[str, Any] = dict(
        max_new_tokens=max_new,
        do_sample=True,
        temperature=0.65,
        top_p=0.88,
        repetition_penalty=1.25,
        no_repeat_ngram_size=4,
        num_return_sequences=1,
    )
    try:
        out = pipe(prompt, **gen_kw)
    except TypeError:
        # Older transformers: drop unsupported keys
        gen_kw.pop("no_repeat_ngram_size", None)
        try:
            out = pipe(prompt, **gen_kw)
        except Exception:
            gen_kw = dict(max_new_tokens=min(max_new, 40), do_sample=False, num_return_sequences=1)
            out = pipe(prompt, **gen_kw)
    except Exception:
        return None

    text = out[0].get("generated_text", "")
    if text.startswith(prompt):
        text = text[len(prompt) :].lstrip()
    text = re.sub(r"\s+", " ", text).strip()
    if _is_degenerate(text):
        return None
    return text


def generate_briefing(ctx: Dict[str, Any], model_name: Optional[str] = None) -> LLMResult:
    """
    Primary entry: faithful text always; optional short neural sentence if quality checks pass.
    """
    model_name = model_name or os.environ.get("HF_LLM_MODEL", "distilgpt2")

    faithful = build_faithful_briefing(ctx)

    if os.environ.get("DISABLE_LLM", "").strip() == "1":
        return LLMResult(used_llm=False, text=faithful + "\n\n" + _fallback_block(), model_name=model_name)

    compact = (
        f"Load change in window: {float(ctx.get('pct_change', 0)):+.1f}%. "
        f"Anomalies in window: {int(ctx.get('anomalies_recent', 0))}. "
        f"Trend vs actuals: {ctx.get('trend_word', '')}."
    )
    if os.environ.get("FAITHFUL_ONLY", "").strip() == "1":
        return LLMResult(used_llm=False, text=faithful + "\n\n(Neural commentary disabled: FAITHFUL_ONLY=1)", model_name=model_name)

    addon = _short_neural_addon(model_name, compact)
    if addon:
        body = faithful + "\n\n--- Short model commentary (verify against figures above) ---\n" + addon
        return LLMResult(used_llm=True, text=body, model_name=model_name)

    body = faithful + "\n\n--- Neural commentary omitted (model off, offline cache, or output failed quality checks) ---\n" + _fallback_block()
    return LLMResult(used_llm=False, text=body, model_name=model_name)


# Backwards compatibility
def generate_explanation(prompt: str, model_name: Optional[str] = None, max_new_tokens: int = 220) -> LLMResult:
    del prompt, max_new_tokens
    return generate_briefing(
        {
            "dataset": "(legacy prompt)",
            "pct_change": 0.0,
            "anomalies_recent": 0,
            "trend_word": "unknown",
            "horizon": 24,
            "window_points": 0,
            "anomaly_series_pct": 0.0,
            "val_mae": "n/a",
            "val_rmse": "n/a",
        },
        model_name=model_name,
    )
