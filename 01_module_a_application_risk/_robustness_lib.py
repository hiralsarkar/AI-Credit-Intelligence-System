"""
_robustness_lib.py - statistics behind the Module A robustness checks

Pure numpy / scipy functions with no data dependencies, so every one of them is
unit-tested on synthetic data (tests/test_robustness.py) before it touches the
Home Credit file.

Conventions
-----------
score : higher = riskier (a PD). Lenders approve the LOWEST scores first.
y     : 1 = default, 0 = repaid.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


# ── Discrimination ────────────────────────────────────────────────────────────

def auc(y, score) -> float:
    """ROC AUC via the Mann-Whitney rank statistic (ties get average ranks)."""
    y = np.asarray(y)
    r = rankdata(score)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ks(y, score) -> float:
    """Kolmogorov-Smirnov separation between defaulters and non-defaulters."""
    y = np.asarray(y)
    order = np.argsort(score)
    yt = y[order]
    cum_bad = np.cumsum(yt) / max(yt.sum(), 1)
    cum_good = np.cumsum(1 - yt) / max((1 - yt).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


# ── Approval curves (sorted-prefix form: approve the n lowest scores) ─────────

def prefix_default_rates(y, score):
    """Default rate of the approved book when the k lowest-risk applicants are approved, k = 1..n."""
    y = np.asarray(y)
    ys = y[np.argsort(score, kind="mergesort")]
    return np.cumsum(ys) / np.arange(1, len(ys) + 1)


def default_at_approval(y, score, approval: float) -> float:
    """Realised default rate when approving the given share of applicants."""
    rates = prefix_default_rates(y, score)
    k = max(1, int(len(rates) * approval))
    return float(rates[k - 1])


def approval_at_default(y, score, target_default: float) -> float:
    """Largest approval share whose approved book still defaults at or below the target."""
    rates = prefix_default_rates(y, score)
    ok = np.nonzero(rates <= target_default)[0]
    return float((ok.max() + 1) / len(rates)) if len(ok) else 0.0


def swap_set(y, bench_score, model_score, anchor: float = 0.50) -> dict:
    """
    The two ways to take a better model's gain at the benchmark's anchor point.
      SC1: same approval, lower default.  SC2: same default, higher approval.
    """
    bench_def = default_at_approval(y, bench_score, anchor)
    model_def = default_at_approval(y, model_score, anchor)
    model_appr = approval_at_default(y, model_score, bench_def)
    return {
        "bench_default": bench_def,
        "sc1_model_default": model_def,
        "sc1_default_reduction_rel": (bench_def - model_def) / bench_def if bench_def > 0 else 0.0,
        "sc2_model_approval": model_appr,
        "sc2_approval_gain_rel": (model_appr - anchor) / anchor,
    }


# ── Calibration ───────────────────────────────────────────────────────────────

def reliability_table(y, p, bins: int = 10) -> list[dict]:
    """Equal-count bins of predicted PD with the observed default rate in each."""
    y = np.asarray(y)
    p = np.asarray(p)
    order = np.argsort(p, kind="mergesort")
    rows = []
    for idx in np.array_split(order, bins):
        if len(idx) == 0:
            continue
        rows.append({
            "n": int(len(idx)),
            "mean_pd": float(p[idx].mean()),
            "observed": float(y[idx].mean()),
        })
    return rows


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Count-weighted mean gap between predicted and observed default across bins."""
    rows = reliability_table(y, p, bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["mean_pd"] - r["observed"]) for r in rows) / n)


def brier(y, p) -> float:
    """Mean squared error of the probability forecast (lower is better)."""
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap(y, scores: dict, stat, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    """
    Paired, stratified bootstrap of any statistic over one test set.

    Every resample keeps the number of defaulters fixed (stratified) and resamples
    all models' scores with the SAME rows (paired), so differences between models
    are measured on identical borrowers.

    stat(y, scores) -> dict[str, float]
    Returns {name: {"point", "lo", "hi"}} with percentile intervals.
    """
    y = np.asarray(y)
    scores = {k: np.asarray(v) for k, v in scores.items()}
    point = stat(y, scores)
    rng = np.random.default_rng(seed)
    pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        s = stat(y[idx], {k: v[idx] for k, v in scores.items()})
        for k in draws:
            draws[k].append(s[k])
    out = {}
    for k, v in draws.items():
        lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[k] = {"point": float(point[k]), "lo": float(lo), "hi": float(hi)}
    return out
