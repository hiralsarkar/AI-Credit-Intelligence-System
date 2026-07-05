"""
Module C production pipeline: portfolio / default model (LendingClub).
  Incumbent (Champion)  : LogisticRegression on FICO + grade only (standard underwriting)
  Custom model (Challenger): XGBoost on borrower credit-file features
Price-derived fields excluded (int_rate, installment, rate spreads, EL proxy) to avoid
endogeneity / leakage from the lender's own pricing decision.
"""
import warnings, json; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os
from pathlib import Path
_HERE = Path(__file__).resolve().parent
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

BASE = str(_HERE)
P = BASE + r"\01_data\processed"
Xtr = pd.read_csv(P + r"\X_train_c.csv"); ytr = pd.read_csv(P + r"\y_train_c.csv").iloc[:, 0].values
Xte = pd.read_csv(P + r"\X_test_c.csv"); yte = pd.read_csv(P + r"\y_test_c.csv").iloc[:, 0].values

leak = ["int_rate", "installment", "RATE_SPREAD", "RATE_PER_RISK", "EL_RATE_PROXY",
        "grade_n", "sub_grade_n"]
feat = [c for c in Xtr.columns if c not in leak]
inc_cols = ["fico_mid", "grade_n"]
print(f"train {Xtr.shape}  test {Xte.shape}  test default rate {yte.mean():.4f}")
print(f"challenger features: {len(feat)} (price/grade fields excluded)")

def ks_stat(y, s):
    o = np.argsort(s); yt = y[o]
    cb = np.cumsum(yt)/max(yt.sum(),1); cg = np.cumsum(1-yt)/max((1-yt).sum(),1)
    return float(np.max(np.abs(cb-cg)))
def metrics(y, s):
    a = roc_auc_score(y, s); return dict(auc=round(a,4), ks=round(ks_stat(y,s),4), gini=round(2*a-1,4))

inc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
inc.fit(Xtr[inc_cols], ytr); p_inc = inc.predict_proba(Xte[inc_cols])[:, 1]

xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                    colsample_bytree=0.8, reg_lambda=1.0, eval_metric="auc", random_state=42, n_jobs=4)
xgb.fit(Xtr[feat], ytr); p_cust = xgb.predict_proba(Xte[feat])[:, 1]

m_inc, m_cust = metrics(yte, p_inc), metrics(yte, p_cust)
print("INCUMBENT  (FICO+grade):", m_inc)
print("CUSTOM     (full model):", m_cust)

def default_at(score, q):
    thr = np.quantile(score, q); appr = score <= thr
    return yte[appr].mean(), appr.mean()
def approval_at_default(score, target):
    best = 0.0
    for q in np.linspace(0.02, 0.98, 481):
        dr, ar = default_at(score, q)
        if dr <= target: best = ar
    return best

print("\nSWAP-SET @ incumbent 50% approval:")
inc_dr, inc_ar = default_at(p_inc, 0.50)
cust_dr_sv, _ = default_at(p_cust, 0.50)
cust_ar_sd = approval_at_default(p_cust, inc_dr)
print(f"  incumbent: {inc_ar:.1%} approval, {inc_dr:.2%} default")
print(f"  SC1 same volume: custom default {cust_dr_sv:.2%}  ({(inc_dr-cust_dr_sv)/inc_dr:+.1%} rel)")
print(f"  SC2 same default: custom approval {cust_ar_sd:.1%}  ({cust_ar_sd-inc_ar:+.1%} pp)")

scored = Xte.copy()
scored["actual_default"] = yte
scored["pd_custom"] = p_cust
scored["pd_incumbent"] = p_inc
scored.to_csv(P + r"\scored_test_c.csv", index=False)
json.dump({"incumbent": m_inc, "custom": m_cust,
           "swap_set": {"incumbent_approval": round(inc_ar,4), "incumbent_default": round(inc_dr,4),
                        "sc1_custom_default": round(cust_dr_sv,4),
                        "sc2_custom_approval": round(cust_ar_sd,4)}},
          open(BASE + r"\04_outputs\module_c_metrics.json", "w"), indent=2)
print("\nSaved scored_test_c.csv, module_c_metrics.json")
