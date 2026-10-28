"""
Surgical Strike - fused collections targeting (Module D).

_collections_d.py answers "who is about to roll into NPA" (risk) and prices intervention
with an ASSUMED flat 30% lift for whoever gets contacted. _uplift_nba_d.py separately
answers "who would actually be saved by intervention" (persuadability) but never feeds
that back into a targeting decision. Neither, on its own, gives you the thing a CRO can
act on: which accounts, contacted with the same fixed collections capacity, produce the
most avoided loss.

This script fuses the two:
  1. Re-fits both models on a single, consistently engineered feature set (behavioural B2N
     risk + T-learner persuadability), scored on the LIVE book: loans currently open in
     Stage 2 whose performance record simply ends there (right-censored), not the
     already-resolved population the two source scripts trained on. Live-book features
     (max DPD so far, months delinquent so far) are running/point-in-time, not the
     lifetime versions used at training time, so scoring never looks ahead.
  2. Replaces the assumed 30% lift with the uplift model's own MEASURED per-decile
     avoided-default gap (treated vs control, observed in the training data) as the
     priced benefit of contacting an account.
  3. Prices every strategy the same way - EAD (current UPB) x LGD (41.8%, from
     _pipeline_d.py's realised-loss LGD) x benefit - so a risk-only queue and a
     risk x persuadability queue can be compared in real rupee/ECL terms at the same
     contact capacity, and the gain is decomposed into "better targeting" vs
     "honest pricing" so neither number is asserted without a mechanism.

Inherited limitation, unchanged from _uplift_nba_d.py: loan modification is an observed,
not randomised, treatment. The measured decile gaps are still a confounded, observational
estimate - servicers modify the loans they judge salvageable. This script prices strategies
using that estimate and states the caveat; it does not claim a causal effect. A production
deployment needs a randomised champion/challenger holdout before the priced numbers below
are relied on for capital planning.
"""
import warnings, json, zipfile, io; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os
from pathlib import Path
_HERE = Path(__file__).resolve().parent
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

DL = os.environ.get("FREDDIE_RAW_DIR", str(Path.home() / "Downloads"))
OUT = str(_HERE / "04_outputs")
YEARS = [2008, 2009, 2010, 2011]
ORIG_USE = {0: "fico", 8: "cltv", 9: "dti", 10: "orig_upb", 11: "ltv", 12: "rate", 19: "loan_id", 21: "term"}
PERF_USE = {0: "loan_id", 1: "period", 2: "upb", 3: "dq", 4: "age", 7: "modflag", 8: "zb"}
DEFAULT_ZB = {3, 9}
PREPAY_ZB = {1}
ASSUMED_LIFT = 0.30          # the flat lift _collections_d.py assumes - kept here only as a comparison baseline
CAPACITY_FRACS = [0.10, 0.20, 0.30]


def dq_num(s):
    s = str(s).strip().split(".")[0]
    return int(s) if s.isdigit() else np.nan


beh_train_rows, uplift_train_rows, live_rows = [], [], []

for yr in YEARS:
    z = zipfile.ZipFile(rf"{DL}\sample_{yr}.zip")
    orig = pd.read_csv(io.BytesIO(z.read(f"sample_orig_{yr}.txt")), sep="|", header=None,
                        usecols=list(ORIG_USE), names=[ORIG_USE[i] for i in sorted(ORIG_USE)], low_memory=False)
    orig["vintage"] = yr
    orig = orig.set_index("loan_id")

    perf = pd.read_csv(io.BytesIO(z.read(f"sample_svcg_{yr}.txt")), sep="|", header=None,
                        usecols=list(PERF_USE), names=[PERF_USE[i] for i in sorted(PERF_USE)], low_memory=False)
    perf["dq"] = perf["dq"].astype(str)
    perf["dqn"] = perf["dq"].map(dq_num)
    perf["stage"] = np.where(perf["dqn"] == 0, 1, np.where(perf["dqn"] <= 2, 2, 3))
    assert (pd.Series(perf["stage"]).value_counts(normalize=True).get(1, 0)) > 0.80, f"{yr}: implausible stage mix"
    perf = perf.sort_values(["loan_id", "period"])

    zbn = pd.to_numeric(perf["zb"], errors="coerce")
    perf["terminal_default"] = zbn.isin(DEFAULT_ZB)
    perf["terminal_prepay"] = zbn.isin(PREPAY_ZB)
    perf["resolved_row"] = perf["terminal_default"] | perf["terminal_prepay"]

    g = perf.groupby("loan_id")
    perf["d"] = (perf["dqn"] >= 1).astype(int)
    perf["prevd"] = g["d"].shift(1).fillna(0)
    perf["spellstart"] = ((perf["d"] == 1) & (perf["prevd"] == 0)).astype(int)
    perf["cumspell"] = g["spellstart"].cumsum()
    perf["months_in_spell"] = perf.groupby(["loan_id", "cumspell"]).cumcount() + 1
    perf.loc[perf["d"] == 0, "months_in_spell"] = 0
    perf["prior_episodes"] = np.where(perf["d"] == 1, perf["cumspell"] - 1, perf["cumspell"])
    perf["mod"] = (perf["modflag"].astype(str).str.strip() == "Y").astype(int)
    perf["ever_modified"] = g["mod"].cummax()
    perf["nstage"] = g["stage"].shift(-1)
    # point-in-time (no-lookahead) running versions of the uplift model's lifetime features
    perf["max_dpd_so_far"] = g["dqn"].cummax()
    perf["delinq_months_so_far"] = g["d"].cumsum()

    # 1) B2N training rows: Stage-2 loan-months with a known next-month stage
    s2 = perf[(perf["stage"] == 2)].dropna(subset=["nstage"]).copy()
    s2 = s2.sample(min(80000, len(s2)), random_state=42)
    o = orig.reindex(s2["loan_id"]).reset_index(drop=True)
    upb = pd.to_numeric(s2["upb"].values, errors="coerce")
    beh_train_rows.append(pd.DataFrame({
        "dpd_bucket": s2["dqn"].values, "months_in_spell": s2["months_in_spell"].values,
        "prior_episodes": s2["prior_episodes"].values, "ever_modified": s2["ever_modified"].values,
        "loan_age": s2["age"].values, "upb_ratio": upb / pd.to_numeric(o["orig_upb"].values, errors="coerce"),
        "fico": o["fico"].values, "ltv": o["ltv"].values, "dti": o["dti"].values, "rate": o["rate"].values,
        "b2n": (s2["nstage"].values == 3).astype(int),
    }))

    # 2) uplift training rows: ever-delinquent loans that reached a TERMINAL state (resolved)
    L = pd.DataFrame({
        "ever_delinq": g["dqn"].max().ge(1), "resolved": g["resolved_row"].any(),
        "max_dpd": g["dqn"].max(), "delinq_months": g["d"].sum(), "modified": g["mod"].max(),
        "defaulted": g["terminal_default"].any(),
    })
    Lr = L[L["ever_delinq"] & L["resolved"]].join(orig, how="left")
    Lr["avoided_default"] = (~Lr["defaulted"]).astype(int)
    uplift_train_rows.append(Lr.reset_index()[
        ["fico", "ltv", "dti", "rate", "term", "max_dpd", "delinq_months", "modified", "avoided_default"]
    ])

    # 3) live book: last row per loan, still Stage 2, panel ended before resolution
    last = perf.groupby("loan_id").tail(1)
    open_s2 = last[(last["stage"] == 2) & (~last["resolved_row"])].copy()
    lo = orig.reindex(open_s2["loan_id"]).reset_index(drop=True)
    live_rows.append(pd.DataFrame({
        "loan_id": open_s2["loan_id"].values, "vintage": yr,
        "dpd_bucket": open_s2["dqn"].values, "months_in_spell": open_s2["months_in_spell"].values,
        "prior_episodes": open_s2["prior_episodes"].values, "ever_modified": open_s2["ever_modified"].values,
        "loan_age": open_s2["age"].values,
        "upb_ratio": pd.to_numeric(open_s2["upb"].values, errors="coerce") / pd.to_numeric(lo["orig_upb"].values, errors="coerce"),
        "ead": pd.to_numeric(open_s2["upb"].values, errors="coerce"),
        "fico": lo["fico"].values, "ltv": lo["ltv"].values, "dti": lo["dti"].values,
        "rate": lo["rate"].values, "term": lo["term"].values,
        "max_dpd_so_far": open_s2["max_dpd_so_far"].values, "delinq_months_so_far": open_s2["delinq_months_so_far"].values,
    }))
    print(f"{yr}: B2N train rows {len(beh_train_rows[-1]):,}  uplift train loans {len(uplift_train_rows[-1]):,}  "
          f"live open Stage-2 {len(live_rows[-1]):,}")

beh = pd.concat(beh_train_rows, ignore_index=True)
upl = pd.concat(uplift_train_rows, ignore_index=True).dropna(subset=["fico"])
live = pd.concat(live_rows, ignore_index=True).dropna(subset=["fico", "ead"])
print(f"\nTotal: B2N training rows {len(beh):,}   uplift training loans {len(upl):,}   LIVE open Stage-2 book {len(live):,}")

# ---- combined behavioural B2N model
b2n_feat = ["fico", "ltv", "dti", "rate", "dpd_bucket", "months_in_spell", "prior_episodes",
            "ever_modified", "loan_age", "upb_ratio"]
m_b2n = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=1000, class_weight="balanced"))
m_b2n.fit(beh[b2n_feat], beh["b2n"])
auc_b2n = roc_auc_score(beh["b2n"], m_b2n.predict_proba(beh[b2n_feat])[:, 1])
print(f"\nB2N model (behavioural + origination) AUC: {auc_b2n:.3f}")

# ---- T-learner uplift model (same method as _uplift_nba_d.py)
up_feat = ["fico", "ltv", "dti", "rate", "term", "max_dpd", "delinq_months"]
T = upl["modified"].values
Y = upl["avoided_default"].values
m1 = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42).fit(upl[up_feat][T == 1], Y[T == 1])
m0 = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42).fit(upl[up_feat][T == 0], Y[T == 0])
upl["uplift"] = m1.predict_proba(upl[up_feat])[:, 1] - m0.predict_proba(upl[up_feat])[:, 1]

# uplift-decile validation on the TRAINING (resolved) population -> the MEASURED benefit table
upl["udec"], edges = pd.qcut(upl["uplift"], 10, labels=False, duplicates="drop", retbins=True)
n_bins = len(edges) - 1
decile_gap = {}
for d in range(n_bins):
    s = upl[upl["udec"] == d]
    t = s[s["modified"] == 1]["avoided_default"]
    c = s[s["modified"] == 0]["avoided_default"]
    gap = (t.mean() - c.mean()) if (len(t) > 20 and len(c) > 20) else np.nan
    decile_gap[d + 1] = round(float(gap), 4) if gap == gap else None
print(f"\nUplift-decile validation on resolved training population ({n_bins} bins, measured avoided-default gap):")
for d, gp in decile_gap.items():
    print(f"  decile {d:2d}: {'n/a (too few observations)' if gp is None else f'{gp:+.1%}'}")

# ---- score the LIVE, currently-open Stage-2 book
live["b2n_risk"] = m_b2n.predict_proba(live[b2n_feat])[:, 1]

live_up_X = live[["fico", "ltv", "dti", "rate", "term"]].copy()
live_up_X["max_dpd"] = live["max_dpd_so_far"]
live_up_X["delinq_months"] = live["delinq_months_so_far"]
live_up_X = live_up_X[up_feat]
live["uplift"] = m1.predict_proba(live_up_X)[:, 1] - m0.predict_proba(live_up_X)[:, 1]

live["uplift_clipped"] = live["uplift"].clip(edges[0], edges[-1])
live["udec"] = pd.cut(live["uplift_clipped"], bins=edges, labels=False, include_lowest=True) + 1
live["measured_gap"] = live["udec"].map(decile_gap).astype(float)
# unknown or non-positive deciles get zero assumed benefit - conservative, not an assumed harm
live["measured_gap"] = live["measured_gap"].fillna(0.0).clip(lower=0.0)

live["priority_score"] = live["b2n_risk"] * live["measured_gap"]

print(f"\nLive open Stage-2 book: {len(live):,} loans, total EAD {live['ead'].sum():,.0f}")
print(f"  mean B2N risk (next-month roll to NPA): {live['b2n_risk'].mean():.1%}")
print(f"  mean measured persuadability gap: {live['measured_gap'].mean():+.1%}  "
      f"(share with a measurable positive gap: {(live['measured_gap'] > 0).mean():.1%})")

# ---- pull LGD and Stage-2 lifetime ECL rate from the pipeline's own output (single source of truth)
with open(os.path.join(OUT, "module_d_ecl.json")) as f:
    ecl_ref = json.load(f)
LGD = ecl_ref["lgd"]
stage2_lifetime_ecl_rate = ecl_ref["ecl_by_stage"]["stage2_lifetime_ecl_rate"]
book_baseline_ecl = float(live["ead"].sum()) * stage2_lifetime_ecl_rate
print(f"\nLGD (from _pipeline_d.py, realised losses): {LGD:.1%}")
print(f"Live book baseline lifetime ECL (EAD x Stage-2 lifetime ECL rate {stage2_lifetime_ecl_rate:.2%}): {book_baseline_ecl:,.0f}")

# ---- four-way comparison: targeting order (risk-only vs fused) x pricing (assumed vs measured)
print("\n=== TARGETING COMPARISON (expected loss avoided, EAD x LGD x benefit) ===")
comparison = {}
for frac in CAPACITY_FRACS:
    n = max(1, int(len(live) * frac))
    risk_top = live.nlargest(n, "b2n_risk")
    fused_top = live.nlargest(n, "priority_score")

    A = float((risk_top["ead"] * LGD * ASSUMED_LIFT).sum())                 # risk-only  + assumed lift (status quo)
    B = float((risk_top["ead"] * LGD * risk_top["measured_gap"]).sum())     # risk-only  + measured gap
    C = float((fused_top["ead"] * LGD * ASSUMED_LIFT).sum())                # fused      + assumed lift
    D = float((fused_top["ead"] * LGD * fused_top["measured_gap"]).sum())   # fused      + measured gap (surgical strike)

    comparison[f"{int(frac*100)}pct"] = dict(
        accounts_contacted=n,
        risk_only_assumed_lift=round(A, 0),
        risk_only_measured_gap=round(B, 0),
        fused_assumed_lift=round(C, 0),
        fused_measured_gap=round(D, 0),
        targeting_effect=round(C - A, 0),      # better queue, same (old) pricing
        measurement_effect=round(B - A, 0),    # same queue, honest pricing
        combined_effect=round(D - A, 0),       # surgical strike vs status quo
        combined_pct_of_book_ecl=round((D - A) / book_baseline_ecl, 4) if book_baseline_ecl else None,
    )
    print(f"\nContact top {int(frac*100)}% ({n:,} accounts):")
    print(f"  A. risk-only,  assumed 30% lift (status quo)   : {A:>14,.0f}")
    print(f"  B. risk-only,  measured decile gap              : {B:>14,.0f}")
    print(f"  C. risk x persuadability, assumed 30% lift      : {C:>14,.0f}")
    print(f"  D. risk x persuadability, measured gap (surgical): {D:>14,.0f}")
    print(f"  -> targeting effect (C-A): {C-A:>12,.0f}   measurement effect (B-A): {B-A:>12,.0f}   "
          f"combined (D-A): {D-A:>12,.0f}  ({(D-A)/book_baseline_ecl:.2%} of book's baseline lifetime ECL)")

# ---- cohort segmentation: which recovery cohorts carry the highest priority?
print("\n=== COHORT SEGMENTATION (live open Stage-2 book) ===")
by_vintage = live.groupby("vintage").agg(
    n=("loan_id", "size"), ead=("ead", "sum"),
    mean_b2n_risk=("b2n_risk", "mean"), mean_measured_gap=("measured_gap", "mean"),
    mean_priority=("priority_score", "mean"),
).round(4)
print("\nBy vintage:")
print(by_vintage.to_string())

live["spell_bucket"] = pd.cut(live["months_in_spell"], bins=[0, 1, 3, 6, 1000],
                               labels=["1 month", "2-3 months", "4-6 months", "7+ months"])
by_spell = live.groupby("spell_bucket", observed=True).agg(
    n=("loan_id", "size"), ead=("ead", "sum"),
    mean_b2n_risk=("b2n_risk", "mean"), mean_measured_gap=("measured_gap", "mean"),
    mean_priority=("priority_score", "mean"),
).round(4)
print("\nBy current delinquency-spell length:")
print(by_spell.to_string())

out = dict(
    b2n_model_auc=round(float(auc_b2n), 3),
    uplift_decile_gap_measured=decile_gap,
    live_book=dict(
        n_loans=int(len(live)), total_ead=round(float(live["ead"].sum()), 0),
        mean_b2n_risk=round(float(live["b2n_risk"].mean()), 4),
        mean_measured_gap=round(float(live["measured_gap"].mean()), 4),
        share_with_positive_gap=round(float((live["measured_gap"] > 0).mean()), 4),
    ),
    lgd_used=LGD,
    stage2_lifetime_ecl_rate=stage2_lifetime_ecl_rate,
    book_baseline_lifetime_ecl=round(book_baseline_ecl, 0),
    targeting_comparison=comparison,
    cohort_by_vintage=json.loads(by_vintage.reset_index().to_json(orient="records")),
    cohort_by_spell_length=json.loads(by_spell.reset_index().to_json(orient="records")),
    assumed_lift_baseline=ASSUMED_LIFT,
    caveat=("Persuadability (measured_gap) comes from an OBSERVATIONAL T-learner: loan "
            "modification is a servicer decision, not a randomised treatment, so the gap is "
            "confounded by selection into treatment. These are priced targeting comparisons "
            "under that estimate, not a validated causal ECL forecast. A production rollout "
            "needs a randomised champion/challenger holdout on a live contact queue before "
            "the combined_effect numbers are used for capital or budget planning."),
)
json.dump(out, open(os.path.join(OUT, "module_d_surgical_strike.json"), "w"), indent=2)
print("\nSaved module_d_surgical_strike.json")
