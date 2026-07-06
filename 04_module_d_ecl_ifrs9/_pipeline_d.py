"""
Module D - IFRS 9 ECL & Staging on Freddie Mac Single-Family loan performance.
Vintages 2008-2011 (50k loans/year + full monthly performance).

Produces, all from real monthly DPD data:
  1. IFRS 9 stage allocation (Stage 1 / Stage 2 SICR / Stage 3 impaired) by month
  2. Stage transition matrix (monthly migration incl. cure, default, prepay)
  3. Lifetime PD by loan age + vintage curves (2008-2011)
  4. Real LGD from ACTUAL LOSS on disposed loans
  5. ECL by stage (12-month vs lifetime) and the provisioning uplift
  6. SICR driver analysis (what predicts Stage 1 -> Stage 2 migration)
Reads directly from the downloaded zips (no extraction).
"""
import warnings, json, zipfile, io; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import os
from pathlib import Path
_HERE = Path(__file__).resolve().parent
DL = os.environ.get("FREDDIE_RAW_DIR", str(Path.home() / "Downloads"))
OUT = str(_HERE / "04_outputs")
YEARS = [2008, 2009, 2010, 2011]

# origination: keep id + risk drivers (positions verified)
ORIG_USE = {0:"fico",2:"fthb",7:"occupancy",8:"cltv",9:"dti",10:"orig_upb",
            11:"ltv",12:"rate",16:"state",19:"loan_id",20:"purpose",21:"term"}
# performance: id, period, upb, delinquency status, age, zero-balance code, actual loss,
# zero-balance removal UPB (exposure at default). Positions align across the 32- and
# 35-column Freddie layouts for cols 0-26.
PERF_USE = {0:"loan_id",1:"period",2:"upb",3:"dq",4:"age",8:"zb",21:"loss",26:"zb_upb"}

def dq_to_stage(dq):
    # '0' current=Stage1; '1','2'=30-89 DPD=Stage2(SICR); '3'+=90+ DPD=Stage3; 'RA'=reperforming=Stage2
    s = str(dq).strip()
    if s == "RA": return 2
    if not s.isdigit(): return np.nan      # 'XX' / blank -> unknown
    n = int(s)
    return 1 if n == 0 else 2 if n <= 2 else 3

DEFAULT_ZB = {3, 9}   # short-sale/charge-off, REO disposition = credit default
PREPAY_ZB  = {1}      # prepaid / matured

trans_counts = {}                 # (from,to) -> count  among S1,S2,S3,DEFAULT,PREPAY
vintage_def_by_age = {}           # year -> {age -> [defaults, at_risk]}
lgd_losses, lgd_upb = [], []
sicr_rows = []                    # loan-level: origination features + reached_S2_24m label
stage_month_counts = {1:0, 2:0, 3:0}
loan_total = 0

for yr in YEARS:
    z = zipfile.ZipFile(rf"{DL}\sample_{yr}.zip")
    orig = pd.read_csv(io.BytesIO(z.read(f"sample_orig_{yr}.txt")), sep="|", header=None,
                       usecols=list(ORIG_USE), names=[ORIG_USE[i] for i in sorted(ORIG_USE)],
                       low_memory=False)
    orig["vintage"] = yr
    loan_total += len(orig)

    perf = pd.read_csv(io.BytesIO(z.read(f"sample_svcg_{yr}.txt")), sep="|", header=None,
                       usecols=list(PERF_USE), names=[PERF_USE[i] for i in sorted(PERF_USE)],
                       low_memory=False)
    perf["stage"] = perf["dq"].map(dq_to_stage)
    perf = perf.sort_values(["loan_id", "period"])

    # terminal state per row
    zb = pd.to_numeric(perf["zb"], errors="coerce")
    perf["state"] = perf["stage"]
    perf.loc[zb.isin(PREPAY_ZB), "state"] = 0      # 0 = prepaid (absorbing)
    perf.loc[zb.isin(DEFAULT_ZB), "state"] = 4     # 4 = default (absorbing)

    # monthly transitions (within loan)
    perf["prev"] = perf.groupby("loan_id")["state"].shift()
    tr = perf.dropna(subset=["prev", "state"])
    vc = tr.groupby(["prev", "state"]).size()
    for (a, b), c in vc.items():
        trans_counts[(int(a), int(b))] = trans_counts.get((int(a), int(b)), 0) + int(c)

    for s in (1, 2, 3):
        stage_month_counts[s] += int((perf["stage"] == s).sum())

    # vintage: first month the loan hits Stage 3 (90+ DPD) = default event; cumulative by age
    perf["is_def"] = (perf["stage"] == 3).astype(float)
    g = perf.groupby("loan_id")
    first_def_age = perf[perf["is_def"] == 1].groupby("loan_id")["age"].min()
    max_age = g["age"].max()
    by_age = vintage_def_by_age.setdefault(yr, {})
    for age in range(0, 121):
        at_risk = int((max_age >= age).sum())
        defaulted = int((first_def_age <= age).sum())
        if at_risk == 0: continue
        by_age[age] = [defaulted, len(orig)]

    # real LGD from credit-loss dispositions. ACTUAL LOSS is stored as a negative number;
    # exposure at default = ZERO BALANCE REMOVAL UPB.
    loss = pd.to_numeric(perf["loss"], errors="coerce").abs()
    ead_def = pd.to_numeric(perf["zb_upb"], errors="coerce")
    ok = loss.notna() & (loss > 0) & ead_def.notna() & (ead_def > 0)
    lgd_losses += list(loss[ok].values); lgd_upb += list(ead_def[ok].values)

    # SICR drivers: did loan reach Stage 2+ within first 24 months?
    early = perf[perf["age"] <= 24]
    reached_s2 = early.groupby("loan_id")["stage"].max().ge(2).rename("reached_s2")
    feat = orig.set_index("loan_id").join(reached_s2)
    feat["reached_s2"] = feat["reached_s2"].fillna(False)
    sicr_rows.append(feat[["fico","cltv","dti","ltv","rate","term","vintage","reached_s2"]])
    print(f"{yr}: loans {len(orig):,}  perf rows {len(perf):,}  S2+ within 24m {feat['reached_s2'].mean():.1%}")

# ---- transition matrix (probabilities) among S1,S2,S3 -> {S1,S2,S3,Default,Prepaid}
labels = {1:"Stage1",2:"Stage2",3:"Stage3",4:"Default",0:"Prepaid"}
print("\nMONTHLY TRANSITION MATRIX (row = from, %):")
TM = {}
for frm in (1,2,3):
    row_total = sum(trans_counts.get((frm,to),0) for to in (1,2,3,4,0))
    if row_total == 0: continue
    TM[labels[frm]] = {labels[to]: round(trans_counts.get((frm,to),0)/row_total,4) for to in (1,2,3,4,0)}
    print(f"  {labels[frm]:7s} -> " + "  ".join(f"{labels[to]}:{TM[labels[frm]][labels[to]]:.3f}" for to in (1,2,3,4,0)))

# ---- LGD
lgd = float(np.sum(lgd_losses)/np.sum(lgd_upb)) if lgd_upb else np.nan
print(f"\nReal LGD (sum actual loss / sum UPB at disposition): {lgd:.1%}  (n={len(lgd_upb):,} disposed)")

# ---- ECL by stage: absorbing Markov chain from the monthly transition counts
# transient states S1,S2,S3 ; absorbing Default(4), Prepaid(0)
trans, absor = [1, 2, 3], [4, 0]
Q = np.zeros((3, 3)); Rm = np.zeros((3, 2))
for ii, i in enumerate(trans):
    rt = sum(trans_counts.get((i, to), 0) for to in trans + absor)
    if rt == 0: continue
    for jj, j in enumerate(trans): Q[ii, jj] = trans_counts.get((i, j), 0) / rt
    for kk, k in enumerate(absor): Rm[ii, kk] = trans_counts.get((i, k), 0) / rt
Nfun = np.linalg.inv(np.eye(3) - Q)
absorb = Nfun @ Rm                       # P(absorption) into [Default, Prepaid] by start stage
life_pd = absorb[:, 0]                    # lifetime probability of terminal default
v = np.array([1.0, 0, 0]); pd12 = 0.0     # 12-month PD from Stage 1
for _ in range(12):
    pd12 += float(v @ Rm[:, 0]); v = v @ Q
ecl = dict(
    stage1_12m_pd=round(pd12, 4),
    stage1_lifetime_pd=round(float(life_pd[0]), 4),
    stage2_lifetime_pd=round(float(life_pd[1]), 4),
    stage3_lifetime_pd=round(float(life_pd[2]), 4),
    stage1_12m_ecl_rate=round(pd12 * lgd, 4),
    stage2_lifetime_ecl_rate=round(float(life_pd[1]) * lgd, 4),
    stage3_lifetime_ecl_rate=round(float(life_pd[2]) * lgd, 4),
    lifetime_uplift_s1_to_s2=round(float(life_pd[1]) / pd12, 1) if pd12 > 0 else None,
)
print("\nECL BY STAGE (LGD {:.1%}):".format(lgd))
print(f"  Stage 1  12-month PD {ecl['stage1_12m_pd']:.2%} -> 12m ECL rate {ecl['stage1_12m_ecl_rate']:.2%}")
print(f"  Stage 2  lifetime PD {ecl['stage2_lifetime_pd']:.2%} -> lifetime ECL rate {ecl['stage2_lifetime_ecl_rate']:.2%}")
print(f"  Stage 3  lifetime PD {ecl['stage3_lifetime_pd']:.2%} -> lifetime ECL rate {ecl['stage3_lifetime_ecl_rate']:.2%}")
print(f"  Provisioning uplift on Stage 1 -> Stage 2 migration: {ecl['lifetime_uplift_s1_to_s2']}x")

# ---- stage distribution across all loan-months
tot = sum(stage_month_counts.values())
print("\nStage distribution (all loan-months):")
for s in (1,2,3):
    print(f"  Stage {s}: {stage_month_counts[s]/tot:.2%}")

# ---- vintage cumulative default (Stage 3) by age
vint = {}
for yr, d in vintage_def_by_age.items():
    vint[yr] = {a: round(v[0]/v[1],4) for a,v in d.items()}
for yr in YEARS:
    a36 = vint[yr].get(36); a60 = vint[yr].get(60)
    print(f"  vintage {yr}: cum default @36m {a36:.2%}" + (f"  @60m {a60:.2%}" if a60 else ""))

# ---- SICR drivers
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
S = pd.concat(sicr_rows)
Xcols = ["fico","cltv","dti","ltv","rate","term"]
Xs = SimpleImputer(strategy="median").fit_transform(S[Xcols])
Xs = StandardScaler().fit_transform(Xs)
lr = LogisticRegression(max_iter=1000).fit(Xs, S["reached_s2"].astype(int))
drivers = sorted(zip(Xcols, lr.coef_[0]), key=lambda x: -abs(x[1]))
print("\nSICR drivers (standardized logit coef, + = raises Stage 2 risk):")
for n,c in drivers: print(f"  {n:6s}: {c:+.3f}")

out = dict(
    loans_total=loan_total, vintages=YEARS,
    transition_matrix=TM,
    lgd=round(lgd,4), lgd_n=len(lgd_upb),
    stage_distribution={f"stage{s}": round(stage_month_counts[s]/tot,4) for s in (1,2,3)},
    vintage_cum_default={str(yr): vint[yr] for yr in YEARS},
    sicr_drivers={n: round(c,4) for n,c in drivers},
    sicr_rate_24m=round(float(S["reached_s2"].mean()),4),
    ecl_by_stage=ecl,
)
json.dump(out, open(OUT + r"\module_d_ecl.json","w"), indent=2)
S.to_csv(OUT + r"\sicr_loanlevel.csv", index=False)
print("\nSaved module_d_ecl.json + sicr_loanlevel.csv")
