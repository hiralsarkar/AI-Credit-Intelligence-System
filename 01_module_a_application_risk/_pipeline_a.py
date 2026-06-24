"""
Module A production pipeline (real numbers, no leakage).
  Incumbent (Champion)  : LogisticRegression on bureau scores only (EXT_SOURCE_1/2/3)
  Custom model (Challenger): XGBoost on the full numeric feature set
Outputs: scored test CSV, swap-set frontier CSV, metrics JSON.
All fitting on train only. Stratified 80/20, seed 42 (same split as keystone).
"""
import warnings, json; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

BASE = r"C:\Users\hiral\Downloads\AI-Credit-Intelligence-System\01_module_a_application_risk"
RAW = BASE + r"\01_data\raw\application_train.csv"
PROC = BASE + r"\01_data\processed"
OUT = BASE + r"\04_outputs"

df = pd.read_csv(RAW)
y = df["TARGET"].values

bureau_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
drop = {"TARGET", "SK_ID_CURR"}
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]

tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
ytr, yte = y[tr], y[te]

def ks_stat(ytrue, score):
    order = np.argsort(score)
    yt = ytrue[order]
    cum_bad = np.cumsum(yt) / max(yt.sum(), 1)
    cum_good = np.cumsum(1 - yt) / max((1 - yt).sum(), 1)
    return np.max(np.abs(cum_bad - cum_good))

def metrics(ytrue, score):
    auc = roc_auc_score(ytrue, score)
    return dict(auc=round(auc, 4), ks=round(ks_stat(ytrue, score), 4), gini=round(2 * auc - 1, 4))

# incumbent: bureau scores only (no class_weight -> calibrated probabilities,
#      so EL/ECL/RAROC economics are honest; ranking/AUC unaffected)
inc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                    LogisticRegression(max_iter=1000))
inc.fit(df.iloc[tr][bureau_cols], ytr)
pd_inc = inc.predict_proba(df.iloc[te][bureau_cols])[:, 1]

# custom model: XGBoost full numeric
xgb = XGBClassifier(n_estimators=450, max_depth=4, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                    eval_metric="auc", random_state=42, n_jobs=4)
xgb.fit(df.iloc[tr][num_cols], ytr)
pd_cust = xgb.predict_proba(df.iloc[te][num_cols])[:, 1]

m_inc, m_cust = metrics(yte, pd_inc), metrics(yte, pd_cust)
print("INCUMBENT  (bureau only):", m_inc)
print("CUSTOM     (challenger) :", m_cust)
print(f"AUC lift = {m_cust['auc']-m_inc['auc']:+.4f}")

# approval / default curves
def curve(score, ytrue, qs):
    rows = []
    for q in qs:
        thr = np.quantile(score, q)
        appr = score <= thr
        rows.append((appr.mean(), ytrue[appr].mean()))
    return np.array(rows)

qs = np.linspace(0.05, 0.95, 91)
c_inc, c_cust = curve(pd_inc, yte, qs), curve(pd_cust, yte, qs)
frontier = pd.DataFrame({
    "q": qs,
    "incumbent_approval": c_inc[:, 0], "incumbent_default": c_inc[:, 1],
    "custom_approval": c_cust[:, 0], "custom_default": c_cust[:, 1],
})
frontier.to_csv(OUT + r"\swapset_frontier.csv", index=False)

# swap-set at the stated anchor: incumbent 50% approval
def default_at(score, ytrue, q):
    thr = np.quantile(score, q); appr = score <= thr
    return ytrue[appr].mean(), appr.mean()
def approval_at_default(score, ytrue, target):
    best = 0.0
    for q in np.linspace(0.02, 0.98, 481):
        dr, ar = default_at(score, ytrue, q)
        if dr <= target: best = ar
    return best

anchor_q = 0.50
inc_dr, inc_ar = default_at(pd_inc, yte, anchor_q)
cust_dr_same_ar, _ = default_at(pd_cust, yte, anchor_q)         # SC1
cust_ar_same_dr = approval_at_default(pd_cust, yte, inc_dr)     # SC2
# SC3: largest custom approval that still beats incumbent default at anchor
sc3_ar = sc3_dr = None
for q in np.linspace(anchor_q, 0.85, 351):
    dr, ar = default_at(pd_cust, yte, q)
    if dr < inc_dr - 0.0005 and ar > inc_ar + 0.005:
        sc3_ar, sc3_dr = ar, dr
swap = dict(
    anchor_incumbent_approval=round(inc_ar, 4), anchor_incumbent_default=round(inc_dr, 4),
    sc1_custom_default_same_approval=round(cust_dr_same_ar, 4),
    sc1_default_reduction_pp=round(inc_dr - cust_dr_same_ar, 4),
    sc1_default_reduction_rel=round((inc_dr - cust_dr_same_ar) / inc_dr, 4),
    sc2_custom_approval_same_default=round(cust_ar_same_dr, 4),
    sc2_approval_gain_pp=round(cust_ar_same_dr - inc_ar, 4),
    sc2_approval_gain_rel=round((cust_ar_same_dr - inc_ar) / inc_ar, 4),
    sc3_custom_approval=round(sc3_ar, 4) if sc3_ar else None,
    sc3_custom_default=round(sc3_dr, 4) if sc3_dr else None,
)
print("\nSWAP-SET @ incumbent 50% approval:")
for k, v in swap.items(): print(f"  {k}: {v}")

# scored output for downstream (RAROC/ECL) stages
scored = pd.DataFrame({
    "loan_id": df.iloc[te]["SK_ID_CURR"].values,
    "actual_default": yte,
    "pd_incumbent": pd_inc,
    "pd_custom": pd_cust,
    "ead": df.iloc[te]["AMT_CREDIT"].values,
    "amt_annuity": df.iloc[te]["AMT_ANNUITY"].values,
    "amt_income": df.iloc[te]["AMT_INCOME_TOTAL"].values,
    "ext_source_2": df.iloc[te]["EXT_SOURCE_2"].values,
    "contract_type": df.iloc[te]["NAME_CONTRACT_TYPE"].values,
    "days_birth": df.iloc[te]["DAYS_BIRTH"].values,
    "region_rating": df.iloc[te]["REGION_RATING_CLIENT"].values,
})
scored.to_csv(PROC + r"\scored_test_a.csv", index=False)

json.dump({"incumbent": m_inc, "custom": m_cust, "swap_set": swap},
          open(OUT + r"\module_a_metrics.json", "w"), indent=2)
print("\nSaved: scored_test_a.csv, swapset_frontier.csv, module_a_metrics.json")
