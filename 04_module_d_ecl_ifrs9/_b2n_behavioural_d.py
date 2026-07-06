"""
Behavioural Bounce-to-NPA (B2N) model - the way a real collections function builds it.

The origination-only B2N was weak (AUC 0.565): FICO/LTV/DTI describe the borrower at
booking, not how they are behaving while delinquent. A roll-rate model must use the
*current delinquency state*. From the Freddie panel we derive, at each Stage 2 loan-month:
  - dpd_bucket      : current depth (30 vs 60 days past due) - the single strongest signal
  - months_in_spell : how long the loan has been delinquent in this episode
  - prior_episodes  : how many times it has been delinquent before (and cured)
  - ever_modified   : has the loan been restructured
  - upb_ratio       : current balance / original balance (burn-down)
  - loan_age        : months on book
plus the origination features, for comparison. Label = rolls to Stage 3 next month.
"""
import warnings, json, zipfile, io; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os
from pathlib import Path
_HERE = Path(__file__).resolve().parent
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

DL = os.environ.get("FREDDIE_RAW_DIR", str(Path.home() / "Downloads"))
OUT = str(_HERE / "04_outputs")
YEARS = [2008, 2009, 2010, 2011]
ORIG_USE = {0:"fico",8:"cltv",9:"dti",10:"orig_upb",11:"ltv",12:"rate",19:"loan_id",21:"term"}
PERF_USE = {0:"loan_id",1:"period",2:"upb",3:"dq",4:"age",7:"modflag",8:"zb"}

def dq_num(s):
    s = str(s).strip().split(".")[0]
    return int(s) if s.isdigit() else np.nan

rows = []
for yr in YEARS:
    z = zipfile.ZipFile(rf"{DL}\sample_{yr}.zip")
    orig = pd.read_csv(io.BytesIO(z.read(f"sample_orig_{yr}.txt")), sep="|", header=None,
                       usecols=list(ORIG_USE), names=[ORIG_USE[i] for i in sorted(ORIG_USE)], low_memory=False)
    orig = orig.set_index("loan_id")
    perf = pd.read_csv(io.BytesIO(z.read(f"sample_svcg_{yr}.txt")), sep="|", header=None,
                       usecols=list(PERF_USE), names=[PERF_USE[i] for i in sorted(PERF_USE)], low_memory=False)
    perf["dq"] = perf["dq"].astype(str)
    perf["dqn"] = perf["dq"].map(dq_num)
    perf["stage"] = np.where(perf["dqn"]==0,1,np.where(perf["dqn"]<=2,2,3))
    assert (pd.Series(perf["stage"]).value_counts(normalize=True).get(1,0)) > 0.80, f"{yr} parse error"
    perf = perf.sort_values(["loan_id","period"])
    g = perf.groupby("loan_id")
    # delinquency spell construction
    perf["d"] = (perf["dqn"] >= 1).astype(int)
    perf["prevd"] = g["d"].shift(1).fillna(0)
    perf["spellstart"] = ((perf["d"]==1) & (perf["prevd"]==0)).astype(int)
    perf["cumspell"] = g["spellstart"].cumsum()
    perf["months_in_spell"] = perf.groupby(["loan_id","cumspell"]).cumcount() + 1
    perf.loc[perf["d"]==0, "months_in_spell"] = 0
    perf["prior_episodes"] = np.where(perf["d"]==1, perf["cumspell"]-1, perf["cumspell"])
    perf["mod"] = (perf["modflag"].astype(str).str.strip()=="Y").astype(int)
    perf["ever_modified"] = g["mod"].cummax()
    perf["nstage"] = g["stage"].shift(-1)
    # Stage 2 modelling rows
    s2 = perf[(perf["stage"]==2)].dropna(subset=["nstage"]).copy()
    s2 = s2.sample(min(80000, len(s2)), random_state=42)
    o = orig.reindex(s2["loan_id"]).reset_index(drop=True)
    upb = pd.to_numeric(s2["upb"].values, errors="coerce")
    rec = pd.DataFrame({
        "dpd_bucket": s2["dqn"].values,                # 1 = 30 DPD, 2 = 60 DPD
        "months_in_spell": s2["months_in_spell"].values,
        "prior_episodes": s2["prior_episodes"].values,
        "ever_modified": s2["ever_modified"].values,
        "loan_age": s2["age"].values,
        "upb_ratio": upb / pd.to_numeric(o["orig_upb"].values, errors="coerce"),
        "fico": o["fico"].values, "ltv": o["ltv"].values, "dti": o["dti"].values, "rate": o["rate"].values,
        "b2n": (s2["nstage"].values==3).astype(int),
    })
    rows.append(rec)
    print(f"{yr}: Stage-2 modelling rows {len(rec):,}  roll-to-NPA rate {rec['b2n'].mean():.2%}")

df = pd.concat(rows, ignore_index=True)
orig_feat = ["fico","ltv","dti","rate"]
beh_feat  = ["dpd_bucket","months_in_spell","prior_episodes","ever_modified","loan_age","upb_ratio"]

def fit(cols):
    X = df[cols]
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=1000, class_weight="balanced"))
    m.fit(X, df["b2n"])
    return m, roc_auc_score(df["b2n"], m.predict_proba(X)[:,1])

m_o, auc_o = fit(orig_feat)
m_b, auc_b = fit(beh_feat)
m_f, auc_f = fit(orig_feat + beh_feat)
print(f"\nB2N AUC - origination features only : {auc_o:.3f}")
print(f"B2N AUC - behavioural features only : {auc_b:.3f}")
print(f"B2N AUC - origination + behavioural : {auc_f:.3f}")

# standardised coefficients of the full model (driver importance)
coefs = sorted(zip(orig_feat+beh_feat, m_f.named_steps["logisticregression"].coef_[0]), key=lambda x:-abs(x[1]))
print("\nDrivers of roll-to-NPA (standardised logit coef):")
for n,c in coefs: print(f"  {n:16s}: {c:+.3f}")

# roll rate by current DPD bucket - the headline behavioural fact
by_bucket = df.groupby("dpd_bucket")["b2n"].agg(["mean","size"])
print("\nRoll-to-NPA rate by current delinquency depth:")
for b,r in by_bucket.iterrows():
    print(f"  {'30 DPD' if b==1 else '60 DPD'}: {r['mean']:.1%}  (n={int(r['size']):,})")

out = dict(b2n_auc_origination=round(auc_o,3), b2n_auc_behavioural=round(auc_b,3),
           b2n_auc_combined=round(auc_f,3),
           drivers={n:round(float(c),3) for n,c in coefs},
           roll_by_bucket={("30dpd" if b==1 else "60dpd"):round(float(r["mean"]),4) for b,r in by_bucket.iterrows()})
json.dump(out, open(OUT + r"\module_d_b2n_behavioural.json","w"), indent=2)
print("\nSaved module_d_b2n_behavioural.json")
