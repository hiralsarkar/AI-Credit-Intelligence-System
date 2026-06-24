"""
IFRS 9 staging, stress testing, and the explicit SC3 point on the custom-model book.
Same economic assumptions as _economics_a.py.
"""
import warnings, json; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

BASE = r"C:\Users\hiral\Downloads\AI-Credit-Intelligence-System\01_module_a_application_risk"
s = pd.read_csv(BASE + r"\01_data\processed\scored_test_a.csv")
LGD, CAP, REV, OPEX_R = 0.45, 0.105, 0.12, 0.03
RW_MAP = {1: .75, 2: .75, 3: 1.0, 4: 1.5, 5: 1.5}
pd_c = s["pd_custom"].values; y = s["actual_default"].values; ead = s["ead"].values
band = np.where(pd_c < .05, 1, np.where(pd_c < .10, 2, np.where(pd_c < .20, 3, np.where(pd_c < .40, 4, 5))))
rw = np.vectorize(RW_MAP.get)(band)

inc_default = 0.0369  # incumbent default at 50% (from economics run)
order = np.argsort(pd_c)

# explicit SC3 point: highest approval that still beats incumbent default AND > 50% approval
sc3 = None
for q in np.linspace(0.50, 0.80, 301):
    n = int(len(pd_c) * q); appr = order[:n]
    dr = y[appr].mean()
    if dr < inc_default - 0.0008:
        sc3 = (q, dr)
print(f"SC3 balanced point: approval {sc3[0]:.1%}, default {sc3[1]:.2%} "
      f"(incumbent 50.0% / {inc_default:.2%}) -> higher approval AND lower default")

# recommended book = SC3 point
n = int(len(pd_c) * sc3[0]); book = order[:n]
def econ_book(idx):
    el = pd_c[idx]*LGD*ead[idx]; cap = ead[idx]*rw[idx]*CAP
    ni = ead[idx]*REV - el - ead[idx]*OPEX_R
    return el, cap, ni
el, cap, ni = econ_book(book)
print(f"\nRecommended (SC3) book: {len(book):,} loans, approval {sc3[0]:.1%}")
print(f"  net income +Rs{ni.sum()/1e7:,.1f} cr | portfolio RAROC {ni.sum()/cap.sum():.1%} | "
      f"ECL rate {el.sum()/ead[book].sum():.2%}")

# IFRS 9 staging on the recommended book.
#   Stage 3 = credit-impaired (defaulted).
#   Stage 2 = significant increase in credit risk (SICR proxy: PD >= 12%, watchlist), not defaulted.
#   Stage 1 = performing, low risk (PD < 12%), not defaulted.
SICR = 0.20   # significant increase in credit risk proxy: PD >= 20% (bands 4-5)
def stage_table(idx, label):
    yy = y[idx]; pp = pd_c[idx]
    st = np.where(yy == 1, 3, np.where(pp >= SICR, 2, 1))
    print(f"\nIFRS 9 staging ({label}, SICR proxy PD>={SICR:.0%}):")
    out = {}
    for s in [1, 2, 3]:
        m = st == s
        if m.sum() == 0:
            out[s] = dict(loans=0, share=0.0, mean_pd=0.0, ecl_rate=0.0)
            print(f"  Stage {s}: 0 loans (0.0%)"); continue
        eclr = (pp[m]*LGD).mean()
        out[s] = dict(loans=int(m.sum()), share=round(m.mean(),4),
                      mean_pd=round(pp[m].mean(),4), ecl_rate=round(eclr,4))
        print(f"  Stage {s}: {m.sum():,} loans ({m.mean():.1%})  mean PD {pp[m].mean():.2%}  ECL rate {eclr:.2%}")
    return out

all_idx = np.arange(len(pd_c))
staging_pool = stage_table(all_idx, "full applicant pool")
staging = stage_table(book, "approved book")

# stress: PD multipliers on the recommended book
print("\nStress testing (recommended book):")
print(f"{'scenario':>10} {'PD x':>6} {'net income (cr)':>16} {'portfolio RAROC':>16}")
for name, mult in [("Baseline",1.0),("Mild",1.5),("Severe",2.5),("Extreme",3.6)]:
    pds = np.minimum(pd_c[book]*mult, 0.999)
    el_s = pds*LGD*ead[book]
    ni_s = ead[book]*REV - el_s - ead[book]*OPEX_R
    print(f"{name:>10} {mult:>6.1f} {ni_s.sum()/1e7:>16,.1f} {ni_s.sum()/cap.sum():>16.1%}")

out = dict(sc3_point=dict(approval=round(sc3[0],4), default=round(sc3[1],4)),
           recommended_book=dict(loans=int(len(book)), approval=round(sc3[0],4),
                                 net_income_cr=round(ni.sum()/1e7,1),
                                 portfolio_raroc=round(ni.sum()/cap.sum(),4),
                                 ecl_rate=round(el.sum()/ead[book].sum(),4)),
           ifrs9_staging_approved=staging, ifrs9_staging_pool=staging_pool)
json.dump(out, open(BASE + r"\04_outputs\module_a_stress_staging.json","w"), indent=2)
print("\nSaved module_a_stress_staging.json")
