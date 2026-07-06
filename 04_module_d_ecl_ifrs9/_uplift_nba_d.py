"""
Collections Uplift / Next-Best-Action model (Module D) - the recommendation-flavoured piece.

Targeting asks 'who will default?'. Uplift asks 'who will my intervention actually change?'.
You should spend collections effort on the *persuadables* - accounts whose outcome moves when
you act - not on the ones who cure (or default) regardless.

Treatment (observable in the panel): loan MODIFICATION (a real servicer intervention on a
delinquent account). Outcome: avoided terminal default. We estimate uplift with a two-model
(T-learner): outcome models fit separately on modified vs non-modified delinquent loans, then
uplift = P(avoid default | modify) - P(avoid default | no modify) per account.

Honest caveat: modification is NOT randomly assigned - servicers modify the accounts they think
are salvageable - so this is an OBSERVATIONAL estimate, confounded by selection into treatment.
A production version uses a randomised champion/challenger holdout or propensity adjustment.
This demonstrates the method and the ranking; it is not a causal claim.
"""
import warnings, json, zipfile, io; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os
from pathlib import Path
_HERE = Path(__file__).resolve().parent
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DL = os.environ.get("FREDDIE_RAW_DIR", str(Path.home() / "Downloads"))
OUT = str(_HERE / "04_outputs")
YEARS = [2008, 2009, 2010, 2011]
ORIG_USE = {0:"fico",8:"cltv",9:"dti",11:"ltv",12:"rate",19:"loan_id",21:"term"}
PERF_USE = {0:"loan_id",1:"period",3:"dq",4:"age",7:"modflag",8:"zb"}

def dq_num(s):
    s = str(s).strip().split(".")[0]
    return int(s) if s.isdigit() else np.nan

rows = []
for yr in YEARS:
    z = zipfile.ZipFile(rf"{DL}\sample_{yr}.zip")
    orig = pd.read_csv(io.BytesIO(z.read(f"sample_orig_{yr}.txt")), sep="|", header=None,
                       usecols=list(ORIG_USE), names=[ORIG_USE[i] for i in sorted(ORIG_USE)], low_memory=False).set_index("loan_id")
    perf = pd.read_csv(io.BytesIO(z.read(f"sample_svcg_{yr}.txt")), sep="|", header=None,
                       usecols=list(PERF_USE), names=[PERF_USE[i] for i in sorted(PERF_USE)], low_memory=False)
    perf["dqn"] = perf["dq"].astype(str).map(dq_num)
    perf["stage"] = np.where(perf["dqn"]==0,1,np.where(perf["dqn"]<=2,2,3))
    zb = pd.to_numeric(perf["zb"], errors="coerce")
    perf["mod"] = (perf["modflag"].astype(str).str.strip()=="Y").astype(int)
    perf["term_default"] = zb.isin([3,9]).astype(int)
    # collapse to loan level over the ever-delinquent population
    g = perf.groupby("loan_id")
    L = pd.DataFrame({
        "ever_delinq": g["dqn"].max().ge(1),
        "max_dpd": g["dqn"].max(),
        "delinq_months": (perf.assign(d=(perf["dqn"]>=1)).groupby("loan_id")["d"].sum()),
        "modified": g["mod"].max(),
        "defaulted": g["term_default"].max(),
        "resolved": g["zb"].apply(lambda s: pd.to_numeric(s,errors="coerce").notna().any()),
    })
    L = L[L["ever_delinq"] & L["resolved"]]          # delinquent loans that reached a terminal state
    L = L.join(orig, how="left")
    L["avoided_default"] = 1 - L["defaulted"]
    rows.append(L.reset_index()[["fico","ltv","dti","rate","term","max_dpd","delinq_months","modified","avoided_default"]])
    print(f"{yr}: ever-delinquent resolved loans {len(L):,}  modified {L['modified'].mean():.1%}  avoided-default {L['avoided_default'].mean():.1%}")

df = pd.concat(rows, ignore_index=True).dropna(subset=["fico"])
feat = ["fico","ltv","dti","rate","term","max_dpd","delinq_months"]
T = df["modified"].values; Y = df["avoided_default"].values
print(f"\nTotal delinquent resolved loans: {len(df):,}   treated (modified): {T.sum():,} ({T.mean():.1%})")

# naive ATE
ate = Y[T==1].mean() - Y[T==0].mean()
print(f"Naive avoided-default rate: modified {Y[T==1].mean():.1%}  vs  not-modified {Y[T==0].mean():.1%}   (naive ATE {ate:+.1%})")

# T-learner uplift
Xtr = df[feat]
m1 = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42).fit(Xtr[T==1], Y[T==1])
m0 = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42).fit(Xtr[T==0], Y[T==0])
uplift = m1.predict_proba(Xtr)[:,1] - m0.predict_proba(Xtr)[:,1]
df["uplift"] = uplift
print(f"\nUplift distribution: mean {uplift.mean():+.3f}  p10 {np.quantile(uplift,.1):+.3f}  p90 {np.quantile(uplift,.9):+.3f}")

# uplift-decile validation: within each predicted-uplift decile, observed treated-vs-control gap
df["udec"] = pd.qcut(df["uplift"], 10, labels=False, duplicates="drop")
print("\nUplift-decile validation (observed avoided-default gap, treated - control):")
val = []
for d in sorted(df["udec"].unique()):
    s = df[df["udec"]==d]
    t = s[s["modified"]==1]["avoided_default"]; c = s[s["modified"]==0]["avoided_default"]
    gap = (t.mean()-c.mean()) if (len(t)>20 and len(c)>20) else np.nan
    val.append((int(d)+1, round(float(gap),3) if gap==gap else None))
    print(f"  decile {int(d)+1:2d}: predicted uplift {s['uplift'].mean():+.3f}  observed gap {gap:+.3f}" if gap==gap else f"  decile {int(d)+1:2d}: predicted uplift {s['uplift'].mean():+.3f}  observed gap n/a")

# next-best-action: target the top-uplift accounts
top = df.nlargest(int(len(df)*0.20), "uplift")
print(f"\nNext-best-action: the top-20% by predicted uplift are the persuadables to prioritise.")
print(f"  Their mean predicted uplift: {top['uplift'].mean():+.3f}  vs whole book {uplift.mean():+.3f}")

out = dict(n_delinquent=int(len(df)), treated_share=round(float(T.mean()),4),
           naive_ate=round(float(ate),4),
           avoided_default_treated=round(float(Y[T==1].mean()),4),
           avoided_default_control=round(float(Y[T==0].mean()),4),
           uplift_mean=round(float(uplift.mean()),4),
           uplift_decile_gap=val, top20_uplift=round(float(top['uplift'].mean()),4))
json.dump(out, open(OUT + r"\module_d_uplift_nba.json","w"), indent=2)
print("\nSaved module_d_uplift_nba.json")
