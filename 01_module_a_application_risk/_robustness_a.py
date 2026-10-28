"""
Module A robustness checks: how much should anyone trust the headline numbers?

Three questions a model validator asks, answered on the same Home Credit data:

  1. Ablation - is the lift from XGBoost, or from giving the model more data?
       Benchmark : logistic regression on the 3 bureau scores (EXT_SOURCE_1/2/3)
       LR (all)  : logistic regression on the full numeric feature set
       XGBoost   : the custom model, same hyperparameters as _pipeline_a.py
     Data effect  = LR(all) - Benchmark.  Model effect = XGBoost - LR(all).

  2. Calibration - can the PD be read as a probability? RAROC, expected loss and
     the breakeven rate all assume it can. Reliability table, Brier score and
     expected calibration error, before and after isotonic calibration fitted on
     the validation set only.

  3. Honest operating point and uncertainty - the recommended approval point is
     chosen on a VALIDATION set and reported on the untouched TEST set, with
     paired, stratified bootstrap 95% intervals on every headline figure.

Split: the same stratified 80/20 split as _pipeline_a.py (seed 42), so the TEST
set is the identical 61,503 loans. The 80% is split again 75/25 into TRAIN (60%)
and VALIDATION (20%). Nothing is fitted or tuned on TEST.

Output: 04_outputs/module_a_robustness.json
Runtime: a few minutes; set BOOTSTRAP_N (default 1000) to trade speed for precision.
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from _operating_point import sc3_point
from _robustness_lib import (
    auc, ks, swap_set, default_at_approval, reliability_table,
    expected_calibration_error, brier, bootstrap,
)

warnings.filterwarnings("ignore")

BASE = str(Path(__file__).resolve().parent)
RAW = os.environ.get("HOME_CREDIT_APP_CSV", os.path.join(BASE, "01_data", "raw", "application_train.csv"))
OUT = os.environ.get("ROBUSTNESS_OUT", os.path.join(BASE, "04_outputs", "module_a_robustness.json"))
SEED = 42
N_BOOT = int(os.environ.get("BOOTSTRAP_N", "1000"))
ANCHOR = 0.50

# ── Data and split ────────────────────────────────────────────────────────────
df = pd.read_csv(RAW)
y = df["TARGET"].values
bureau_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {"TARGET", "SK_ID_CURR"}]

idx_80, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y)
tr, va = train_test_split(idx_80, test_size=0.25, random_state=SEED, stratify=y[idx_80])
ytr, yva, yte = y[tr], y[va], y[te]
print(f"Split  train {len(tr):,}  validation {len(va):,}  test {len(te):,}")


def lr(cols):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(df.iloc[tr][cols], ytr)
    return m


# ── Models (all fitted on TRAIN only) ─────────────────────────────────────────
bench = lr(bureau_cols)
lr_all = lr(num_cols)
xgb = XGBClassifier(n_estimators=450, max_depth=4, learning_rate=0.05, subsample=0.9,
                    colsample_bytree=0.8, reg_lambda=1.0, eval_metric="auc",
                    random_state=SEED, n_jobs=4)
xgb.fit(df.iloc[tr][num_cols], ytr)


def scores(rows):
    X = df.iloc[rows]
    return {
        "benchmark": bench.predict_proba(X[bureau_cols])[:, 1],
        "lr_all": lr_all.predict_proba(X[num_cols])[:, 1],
        "xgboost": xgb.predict_proba(X[num_cols])[:, 1],
    }


s_va, s_te = scores(va), scores(te)

# ── 1. Ablation with bootstrap intervals ──────────────────────────────────────
def ablation_stat(yy, s):
    a = {k: auc(yy, v) for k, v in s.items()}
    return {
        "auc_benchmark": a["benchmark"], "auc_lr_all": a["lr_all"], "auc_xgboost": a["xgboost"],
        "data_effect": a["lr_all"] - a["benchmark"],
        "model_effect": a["xgboost"] - a["lr_all"],
        "total_lift": a["xgboost"] - a["benchmark"],
    }


print(f"Bootstrapping {N_BOOT} resamples ...")
ablation = bootstrap(yte, s_te, ablation_stat, n_boot=N_BOOT, seed=SEED)
ks_te = {k: ks(yte, v) for k, v in s_te.items()}

# ── 2. Calibration (isotonic fitted on VALIDATION only) ───────────────────────
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
iso.fit(s_va["xgboost"], yva)
p_raw, p_cal = s_te["xgboost"], iso.predict(s_te["xgboost"])
calibration = {
    "raw": {
        "brier": brier(yte, p_raw), "ece": expected_calibration_error(yte, p_raw),
        "mean_pd": float(p_raw.mean()), "observed_default": float(yte.mean()),
        "reliability": reliability_table(yte, p_raw),
    },
    "isotonic": {
        "brier": brier(yte, p_cal), "ece": expected_calibration_error(yte, p_cal),
        "mean_pd": float(p_cal.mean()), "observed_default": float(yte.mean()),
        "reliability": reliability_table(yte, p_cal),
    },
    # isotonic is monotone, so ranking (AUC, swap-set, approval order) is unchanged
    "auc_preserved": abs(auc(yte, p_cal) - auc(yte, p_raw)) < 0.005,
}

# ── 3. Operating point chosen on VALIDATION, reported on TEST ─────────────────
bench_def_va = round(default_at_approval(yva, s_va["benchmark"], ANCHOR), 4)
sc3_va = sc3_point(s_va["xgboost"], yva, bench_def_va)
sc3_q = sc3_va[0] if sc3_va else None
threshold = float(np.quantile(s_va["xgboost"], sc3_q)) if sc3_q else None


def headline_stat(yy, s):
    out = swap_set(yy, s["benchmark"], s["xgboost"], ANCHOR)
    if threshold is not None:
        approved = s["xgboost"] <= threshold
        out["sc3_test_approval"] = float(approved.mean())
        out["sc3_test_default"] = float(yy[approved].mean())
    return out


headline = bootstrap(yte, s_te, headline_stat, n_boot=N_BOOT, seed=SEED)

result = {
    "method": {
        "split": "stratified 60/20/20 train/validation/test, seed 42; test = the same 61,503 loans as _pipeline_a.py",
        "bootstrap": f"{N_BOOT} paired, stratified resamples of the test set; 95% percentile intervals",
        "anchor_approval": ANCHOR,
        "operating_point": "SC3 rule from _operating_point.py applied on validation; its score threshold is then applied to test",
    },
    "sizes": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te))},
    "ablation": ablation,
    "ks": ks_te,
    "calibration": calibration,
    "operating_point": {
        "validation_benchmark_default": bench_def_va,
        "validation_sc3": {"approval": sc3_va[0], "default": sc3_va[1]} if sc3_va else None,
        "score_threshold": threshold,
    },
    "headline_with_ci": headline,
}
if threshold is not None:
    h = headline
    result["operating_point"]["test_beats_benchmark_on_both"] = bool(
        h["sc3_test_approval"]["point"] > ANCHOR and h["sc3_test_default"]["point"] < h["bench_default"]["point"]
    )
    result["operating_point"]["test_beats_benchmark_on_both_at_95pct"] = bool(
        h["sc3_test_approval"]["lo"] > ANCHOR and h["sc3_test_default"]["hi"] < h["bench_default"]["point"]
    )
result["calibration"]["recommend_isotonic"] = bool(calibration["isotonic"]["ece"] < calibration["raw"]["ece"])
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)


# ── Summary ───────────────────────────────────────────────────────────────────
def fmt(d, pct=False, digits=3):
    f = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.{digits}f}")
    return f"{f(d['point'])}  [{f(d['lo'])}, {f(d['hi'])}]"


print("\nABLATION (test AUC, 95% CI)")
for k in ("auc_benchmark", "auc_lr_all", "auc_xgboost", "data_effect", "model_effect", "total_lift"):
    print(f"  {k:<16} {fmt(ablation[k])}")
print("\nCALIBRATION (test)")
for k in ("raw", "isotonic"):
    c = calibration[k]
    print(f"  {k:<9} Brier {c['brier']:.4f}  ECE {c['ece']:.4f}  mean PD {c['mean_pd']:.2%} vs observed {c['observed_default']:.2%}")
print("\nHEADLINE (test, 95% CI)")
print(f"  SC1 default reduction at 50% approval  {fmt(headline['sc1_default_reduction_rel'], pct=True)}")
print(f"  SC2 approval gain at benchmark default  {fmt(headline['sc2_approval_gain_rel'], pct=True)}")
if threshold is not None:
    print(f"  SC3 chosen on validation -> test approval {fmt(headline['sc3_test_approval'], pct=True)}, "
          f"default {fmt(headline['sc3_test_default'], pct=True)}")
if threshold is not None:
    op = result["operating_point"]
    print(f"  Benchmark default at 50% approval on test: {headline['bench_default']['point']:.2%}")
    print(f"  Beats benchmark on approval AND default: {op['test_beats_benchmark_on_both']}"
          f" (holds across the 95% interval: {op['test_beats_benchmark_on_both_at_95pct']})")
print(f"  Isotonic calibration recommended: {result['calibration']['recommend_isotonic']}")
print(f"\nSaved {OUT}")
