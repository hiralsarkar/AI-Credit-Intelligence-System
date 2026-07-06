"""
Module D - Collections / ECL-reduction engine on the Freddie Mac performance panel.
Builds the roll-rate and recovery models that turn 'calculate ECL' into 'reduce ECL':
  1. Current-to-Bounce  (C2B): P(Stage 1 -> Stage 2 next month) for performing loans
  2. Bounce-to-NPA      (B2N): P(Stage 2 -> Stage 3 next month) for delinquent loans
  3. NPA-to-Recovery    (cure): P(Stage 3 -> cure) vs roll to default
  4. Collection-strategy optimisation: rank accounts by roll risk, simulate intervention
     on a targeted population, and quantify the ECL reduction (Success Criteria 3).
Reads directly from the downloaded sample_YYYY.zip files. Origination features joined on.
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
ORIG_USE = {0:"fico",8:"cltv",9:"dti",11:"ltv",12:"rate",19:"loan_id",21:"term"}
PERF_USE = {0:"loan_id",1:"period",3:"dq",4:"age",8:"zb",21:"loss",26:"zb_upb"}

def dq_num(s):
    s = str(s).strip().split(".")[0]      # robust to '0', '00', '0.0', int/float inference
    return int(s) if s.isdigit() else np.nan

# accumulate transition events and per-loan-month feature rows (sampled to bound memory)
c2b_rows, b2n_rows = [], []        # (features..., label)
cure_rows = []                      # Stage 3 episodes: features + cured(0/1)
roll_counts = {"S1->S2":0,"S1->S1":0,"S2->S3":0,"S2->cure":0,"S2->S2":0,"S3->cure":0,"S3->S3":0,"S3->default":0}

for yr in YEARS:
    z = zipfile.ZipFile(rf"{DL}\sample_{yr}.zip")
    orig = pd.read_csv(io.BytesIO(z.read(f"sample_orig_{yr}.txt")), sep="|", header=None,
                       usecols=list(ORIG_USE), names=[ORIG_USE[i] for i in sorted(ORIG_USE)], low_memory=False)
    orig = orig.set_index("loan_id")
    perf = pd.read_csv(io.BytesIO(z.read(f"sample_svcg_{yr}.txt")), sep="|", header=None,
                       usecols=list(PERF_USE), names=[PERF_USE[i] for i in sorted(PERF_USE)], low_memory=False)
    perf = perf.sort_values(["loan_id", "period"])
    perf["dq"] = perf["dq"].astype(str)        # force string to avoid int/float inference drift
    perf["dqn"] = perf["dq"].map(dq_num)
    perf["stage"] = np.where(perf["dqn"]==0,1,np.where(perf["dqn"]<=2,2,3))
    sd = pd.Series(perf["stage"]).value_counts(normalize=True)
    print(f"  {yr} stage mix: S1 {sd.get(1,0):.1%} S2 {sd.get(2,0):.1%} S3 {sd.get(3,0):.1%}")
    assert sd.get(1,0) > 0.80, f"{yr}: implausible stage mix - parsing error"
    perf["nstage"] = perf.groupby("loan_id")["stage"].shift(-1)
    zb = pd.to_numeric(perf["zb"], errors="coerce")
    perf["terminal_default"] = zb.isin([3,9])

    # roll counts
    s1 = perf[perf["stage"]==1].dropna(subset=["nstage"])
    s2 = perf[perf["stage"]==2].dropna(subset=["nstage"])
    s3 = perf[perf["stage"]==3].dropna(subset=["nstage"])
    roll_counts["S1->S2"] += int((s1["nstage"]==2).sum()); roll_counts["S1->S1"] += int((s1["nstage"]==1).sum())
    roll_counts["S2->S3"] += int((s2["nstage"]==3).sum()); roll_counts["S2->cure"] += int((s2["nstage"]==1).sum()); roll_counts["S2->S2"] += int((s2["nstage"]==2).sum())
    roll_counts["S3->cure"] += int((s3["nstage"]<3).sum()); roll_counts["S3->S3"] += int((s3["nstage"]==3).sum())
    roll_counts["S3->default"] += int(perf[(perf["stage"]==3)&(perf["terminal_default"])].shape[0])

    # C2B sample: from Stage 1 rows, label = rolls to Stage 2 next month
    samp1 = s1.sample(min(120000, len(s1)), random_state=42)
    f1 = orig.reindex(samp1["loan_id"]).reset_index(drop=True)
    f1["age"] = samp1["age"].values; f1["c2b"] = (samp1["nstage"].values==2).astype(int)
    c2b_rows.append(f1)
    # B2N sample: from Stage 2 rows, label = rolls to Stage 3
    if len(s2):
        samp2 = s2.sample(min(60000, len(s2)), random_state=42)
        f2 = orig.reindex(samp2["loan_id"]).reset_index(drop=True)
        f2["age"] = samp2["age"].values; f2["b2n"] = (samp2["nstage"].values==3).astype(int)
        b2n_rows.append(f2)
    # cure: per Stage-3 loan, did it ever cure (reach stage<3 after) vs terminal default
    g3 = perf[perf["stage"]==3].groupby("loan_id")
    ever_cure = g3.apply(lambda d: (d["nstage"]<3).any()).rename("cured")
    if len(ever_cure):
        fc = orig.reindex(ever_cure.index).copy(); fc["cured"]=ever_cure.values
        cure_rows.append(fc.reset_index(drop=True))
    print(f"{yr}: perf rows {len(perf):,}  S1 {len(s1):,}  S2 {len(s2):,}  S3 episodes {ever_cure.shape[0]:,}")

feat = ["fico","cltv","dti","ltv","rate","term","age"]
def fit_score(rows, label):
    df = pd.concat(rows); X = df[[c for c in feat if c in df.columns]]
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=1000, class_weight="balanced"))
    m.fit(X, df[label]); auc = roc_auc_score(df[label], m.predict_proba(X)[:,1])
    rate = df[label].mean()
    return m, auc, rate, df

print("\n=== ROLL-RATE & RECOVERY MODELS ===")
m_c2b, auc_c2b, r_c2b, d_c2b = fit_score(c2b_rows, "c2b")
m_b2n, auc_b2n, r_b2n, d_b2n = fit_score(b2n_rows, "b2n")
fc = pd.concat(cure_rows); Xc = fc[[c for c in feat if c in fc.columns]]
m_cure = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
m_cure.fit(Xc, fc["cured"]); auc_cure = roc_auc_score(fc["cured"], m_cure.predict_proba(Xc)[:,1])
print(f"Current-to-Bounce  (C2B): AUC {auc_c2b:.3f}  monthly roll rate {r_c2b:.3%}")
print(f"Bounce-to-NPA      (B2N): AUC {auc_b2n:.3f}  monthly roll rate {r_b2n:.3%}")
print(f"NPA cure model          : AUC {auc_cure:.3f}  base cure rate {fc['cured'].mean():.3%}")

# monthly roll rates from counts
s1t = roll_counts["S1->S1"]+roll_counts["S1->S2"]
s2t = roll_counts["S2->cure"]+roll_counts["S2->S2"]+roll_counts["S2->S3"]
print("\nObserved monthly roll rates:")
print(f"  Stage1 -> Stage2 (bounce): {roll_counts['S1->S2']/s1t:.2%}")
print(f"  Stage2 -> Stage3 (worsen): {roll_counts['S2->S3']/s2t:.2%}   Stage2 -> cure: {roll_counts['S2->cure']/s2t:.2%}")

# ---- Collection-strategy optimisation (Success Criteria 3)
# Target Stage-2 accounts by B2N risk; intervene on top-X%, assume intervention lifts cure by `lift`.
print("\n=== COLLECTION STRATEGY OPTIMISATION (Stage 2 book) ===")
b2n_p = m_b2n.predict_proba(d_b2n[[c for c in feat if c in d_b2n.columns]])[:,1]
order = np.argsort(-b2n_p)            # highest roll-to-NPA risk first
base_roll = d_b2n["b2n"].mean()
LIFT = 0.30                            # assumed relative reduction in roll for contacted accounts
print(f"{'contact top':>12} {'NPA roll avoided':>17} {'accounts contacted':>19}")
strat = {}
for frac in [0.10, 0.20, 0.30, 1.00]:
    n = int(len(order)*frac); contacted = order[:n]
    rolls_base = d_b2n["b2n"].values.sum()
    rolls_new = rolls_base - d_b2n["b2n"].values[contacted].sum()*LIFT
    avoided = (rolls_base - rolls_new)/rolls_base
    strat[f"{int(frac*100)}pct"] = dict(contacted=round(frac,2), npa_roll_avoided=round(float(avoided),4))
    print(f"{frac*100:11.0f}% {avoided*100:16.1f}% {n:>19,}")

out = dict(
    roll_rates={"s1_to_s2": round(roll_counts['S1->S2']/s1t,4), "s2_to_s3": round(roll_counts['S2->S3']/s2t,4),
                "s2_to_cure": round(roll_counts['S2->cure']/s2t,4)},
    models={"c2b_auc": round(auc_c2b,3), "c2b_rate": round(r_c2b,4),
            "b2n_auc": round(auc_b2n,3), "b2n_rate": round(r_b2n,4),
            "cure_auc": round(auc_cure,3), "cure_rate": round(float(fc['cured'].mean()),4)},
    collection_strategy=strat, intervention_lift_assumed=LIFT)
json.dump(out, open(OUT + r"\module_d_collections.json","w"), indent=2)
print("\nSaved module_d_collections.json")
