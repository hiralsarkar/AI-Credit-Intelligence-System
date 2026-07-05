"""
Reject Inference - Module C.

A scorecard trained only on BOOKED loans is biased: it never observes how the applicants
the lender declined would have performed. The booked population is a censored, lower-risk
slice of everyone who applied. Reject inference folds the declined applicants back in so the
model reflects the THROUGH-THE-DOOR population - what a regulator and a CRO expect.

Method (parcelling / score-based augmentation):
  1. Fit a PD model on accepted loans using features shared with the rejected file
     (credit score, DTI, requested amount).
  2. Score the rejected applicants -> inferred PD.
  3. Assign inferred good/bad by the model probability (parcelling), weighted by the
     observed accept/reject mix, and compare the booked vs through-the-door bad rate.
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

BASE = str(_HERE)
REJ = os.environ.get("LC_REJECT_CSV", str(Path.home() / "Downloads" / "LendingClub Dataset" / "rejected_2007_to_2018q4.csv" / "rejected_2007_to_2018Q4.csv"))

# ---- accepted (booked) loans with known outcome
acc = pd.read_csv(BASE + r"\01_data\processed\clean_lendingclub.csv",
                  usecols=lambda c: c in ["fico_mid","dti","loan_amnt","DEFAULT"])
acc = acc.dropna(subset=["DEFAULT"])
common = ["fico_mid","dti","loan_amnt"]
print(f"Accepted (booked) loans: {len(acc):,}   booked bad rate: {acc['DEFAULT'].mean():.2%}")

# ---- rejected applicants (sample for speed)
rej = pd.read_csv(REJ, usecols=["Amount Requested","Risk_Score","Debt-To-Income Ratio"])
rej = rej.sample(min(400000, len(rej)), random_state=42)
rej["fico_mid"] = pd.to_numeric(rej["Risk_Score"], errors="coerce")
rej["loan_amnt"] = pd.to_numeric(rej["Amount Requested"], errors="coerce")
rej["dti"] = pd.to_numeric(rej["Debt-To-Income Ratio"].astype(str).str.replace("%","",regex=False), errors="coerce")
rej = rej.dropna(subset=["fico_mid"]).query("fico_mid > 300 and fico_mid < 900")
print(f"Rejected applicants (sample): {len(rej):,}")

# true acceptance rate from the rejection summary (full population counts, not the sample)
rs = pd.read_csv(BASE + r"\01_data\processed\rejection_summary.csv")
accept_rate = rs["approved_count"].sum() / rs["total_apps"].sum()
n_acc, n_rej = len(acc), len(rej)
print(f"True acceptance rate (rejection summary): {accept_rate:.1%}")

# ---- PD model on accepted, shared features (no class weighting -> calibrated PDs)
m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                  LogisticRegression(max_iter=1000))
m.fit(acc[common], acc["DEFAULT"])
auc = roc_auc_score(acc["DEFAULT"], m.predict_proba(acc[common])[:,1])
print(f"Accepted-only PD model AUC (shared features): {auc:.3f}")

# ---- score both populations
acc_pd = m.predict_proba(acc[common])[:,1]
rej_pd = m.predict_proba(rej[common])[:,1]
print(f"\nMean predicted PD - accepted: {acc_pd.mean():.2%}   rejected: {rej_pd.mean():.2%}")
print(f"Median credit score - accepted: {acc['fico_mid'].median():.0f}   rejected: {rej['fico_mid'].median():.0f}")

# ---- parcelling: infer rejected outcomes from model PD, build through-the-door book
rej_inferred_bad = rej_pd                      # expected bad rate per reject
booked_bad = acc["DEFAULT"].mean()
# weight by the true acceptance rate, not the sample mix
ttd_bad = accept_rate * booked_bad + (1 - accept_rate) * rej_inferred_bad.mean()
print(f"\nBooked bad rate (KGB):                {booked_bad:.2%}")
print(f"Inferred rejected bad rate:            {rej_inferred_bad.mean():.2%}")
print(f"Through-the-door bad rate (AGB):       {ttd_bad:.2%}")
print(f"Selection-bias understatement:         {(ttd_bad-booked_bad)/booked_bad:+.0%}")

# swap-set view: of rejects, how many the model would have scored better than the worst accepted decile?
acc_thr = np.quantile(acc_pd, 0.90)            # worst 10% of accepted by PD
good_rejects = (rej_pd <= acc_thr).mean()
print(f"\nShare of rejected applicants safer than the worst-accepted decile: {good_rejects:.1%}")
print("  -> reject inference surfaces good applicants the cutoff turned away (the swap-in set).")

out = dict(accepted=int(n_acc), rejected_sample=int(n_rej), accept_rate=round(accept_rate,4),
           model_auc=round(auc,3), acc_mean_pd=round(float(acc_pd.mean()),4),
           rej_mean_pd=round(float(rej_pd.mean()),4),
           booked_bad_rate=round(float(booked_bad),4), ttd_bad_rate=round(float(ttd_bad),4),
           selection_bias_pct=round(float((ttd_bad-booked_bad)/booked_bad),4),
           safe_rejects_share=round(float(good_rejects),4))
json.dump(out, open(BASE + r"\04_outputs\module_c_reject_inference.json","w"), indent=2)
print("\nSaved module_c_reject_inference.json")
