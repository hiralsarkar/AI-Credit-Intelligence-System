"""
Module B production pipeline: behavioural delinquency model (existing customers).
  Incumbent (Champion)  : LogisticRegression on past-DPD features only (the naive rule)
  Custom model (Challenger): XGBoost on the full behavioural feature set
Business frame = collections SC3: within a fixed intervention budget (contact top X%
riskiest), how many actual delinquents does each model capture? Better capture at the
same budget = catch more at the same cost.
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
Xtr = pd.read_csv(P + r"\X_train_b.csv"); ytr = pd.read_csv(P + r"\y_train_b.csv").iloc[:, 0].values
Xte = pd.read_csv(P + r"\X_test_b.csv"); yte = pd.read_csv(P + r"\y_test_b.csv").iloc[:, 0].values

dpd_cols = [c for c in Xtr.columns if "PastDue" in c or "DaysLate" in c or "DPD" in c]
print("DPD (incumbent) features:", dpd_cols)
print(f"train {Xtr.shape}  test {Xte.shape}  test delinquency rate {yte.mean():.4f}")

def ks_stat(y, s):
    o = np.argsort(s); yt = y[o]
    cb = np.cumsum(yt)/max(yt.sum(),1); cg = np.cumsum(1-yt)/max((1-yt).sum(),1)
    return float(np.max(np.abs(cb-cg)))
def metrics(y, s):
    a = roc_auc_score(y, s); return dict(auc=round(a,4), ks=round(ks_stat(y,s),4), gini=round(2*a-1,4))

inc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
inc.fit(Xtr[dpd_cols], ytr); p_inc = inc.predict_proba(Xte[dpd_cols])[:, 1]

xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                    colsample_bytree=0.8, reg_lambda=1.0, eval_metric="auc", random_state=42, n_jobs=4)
xgb.fit(Xtr, ytr); p_cust = xgb.predict_proba(Xte)[:, 1]

m_inc, m_cust = metrics(yte, p_inc), metrics(yte, p_cust)
print("INCUMBENT  (DPD rule):", m_inc)
print("CUSTOM     (behavioural):", m_cust)

# capture rate at fixed intervention budgets (top X% riskiest contacted)
print("\nCOLLECTIONS CAPTURE (delinquents caught within intervention budget):")
print(f"{'budget':>7} {'incumbent_capture':>18} {'custom_capture':>15} {'lift_pp':>8}")
total_bad = yte.sum()
caps = {}
for b in [0.10, 0.15, 0.20, 0.30]:
    n = int(len(yte) * b)
    inc_caught = yte[np.argsort(-p_inc)[:n]].sum() / total_bad
    cust_caught = yte[np.argsort(-p_cust)[:n]].sum() / total_bad
    caps[b] = dict(incumbent=round(inc_caught,4), custom=round(cust_caught,4), lift_pp=round(cust_caught-inc_caught,4))
    print(f"{b:>7.0%} {inc_caught:>17.1%} {cust_caught:>14.1%} {cust_caught-inc_caught:>+8.1%}")

scored = Xte.copy()
scored["actual_delinquent"] = yte
scored["delinquency_prob"] = p_cust
scored["dpd_rule_prob"] = p_inc
scored.to_csv(P + r"\scored_test_b.csv", index=False)
json.dump({"incumbent": m_inc, "custom": m_cust, "capture": {f"{int(k*100)}pct": v for k,v in caps.items()}},
          open(BASE + r"\04_outputs\module_b_metrics.json", "w"), indent=2)
print("\nSaved scored_test_b.csv, module_b_metrics.json")
