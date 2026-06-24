"""
Module A economics layer: EL, economic capital, RAROC, ECL on the scored test set.
Replicates the project's established assumptions exactly:
  LGD 0.45 | capital ratio 0.105 | revenue 12% of EAD | opex 3% of EAD | hurdle 14%
  band risk weights B1/B2=0.75, B3=1.00, B4/B5=1.50
Compares the incumbent book vs the custom-model books at three operating points.
"""
import warnings, json; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

BASE = r"C:\Users\hiral\Downloads\AI-Credit-Intelligence-System\01_module_a_application_risk"
s = pd.read_csv(BASE + r"\01_data\processed\scored_test_a.csv")

LGD, CAP, REV, OPEX_R, HURDLE = 0.45, 0.105, 0.12, 0.03, 0.14
RW_MAP = {1: 0.75, 2: 0.75, 3: 1.00, 4: 1.50, 5: 1.50}

def band(p):
    return 1 if p < .05 else 2 if p < .10 else 3 if p < .20 else 4 if p < .40 else 5

def econ(pd_col):
    b = s[pd_col].apply(band)
    rw = b.map(RW_MAP)
    ead = s["ead"].values
    el = s[pd_col].values * LGD * ead
    econcap = ead * rw.values * CAP
    ni = ead * REV - el - ead * OPEX_R
    raroc = np.where(econcap > 0, ni / econcap, 0.0)
    return dict(band=b.values, ead=ead, el=el, econcap=econcap, ni=ni, raroc=raroc)

E_cust = econ("pd_custom")
y = s["actual_default"].values

def book(mask, e, pd_col):
    ead = e["ead"][mask]
    return dict(
        approval=round(mask.mean(), 4),
        actual_default=round(y[mask].mean(), 4),
        model_pd=round(s[pd_col].values[mask].mean(), 4),
        ecl_rate=round((e["el"][mask].sum()) / ead.sum(), 4),       # ECL / EAD
        total_ni_cr=round(e["ni"][mask].sum() / 1e7, 1),            # rupees -> crore
        total_ead_cr=round(ead.sum() / 1e7, 1),
        econ_capital_cr=round(e["econcap"][mask].sum() / 1e7, 1),
        portfolio_raroc=round(e["ni"][mask].sum() / e["econcap"][mask].sum(), 4),
        n_loans=int(mask.sum()),
    )

def approve_lowest(pd_vals, q):
    return pd_vals <= np.quantile(pd_vals, q)

pd_inc = s["pd_incumbent"].values
pd_cust = s["pd_custom"].values

# incumbent operating point: 50% approval by bureau score
inc_mask = approve_lowest(pd_inc, 0.50)
inc_default = y[inc_mask].mean()
# economics of the incumbent book (uses same economic model, incumbent selection)
E_inc = econ("pd_incumbent")

# custom, same volume (50%) -> SC1
cust_sv = approve_lowest(pd_cust, 0.50)
# custom, same default appetite as incumbent -> SC2  (find approval matching incumbent default)
best_q = 0.50
for q in np.linspace(0.50, 0.85, 351):
    if y[approve_lowest(pd_cust, q)].mean() <= inc_default:
        best_q = q
cust_sd = approve_lowest(pd_cust, best_q)
# custom, RAROC-gated point (loan RAROC >= hurdle)
cust_raroc = E_cust["raroc"] >= HURDLE

print("=== INCUMBENT book (bureau score, 50% approval) ===")
print(json.dumps(book(inc_mask, E_inc, "pd_incumbent"), indent=2))
print("\n=== CUSTOM, SAME VOLUME 50% (SC1: lower default) ===")
print(json.dumps(book(cust_sv, E_cust, "pd_custom"), indent=2))
print(f"\n=== CUSTOM, SAME DEFAULT APPETITE (SC2: more volume) approval={best_q:.3f} ===")
print(json.dumps(book(cust_sd, E_cust, "pd_custom"), indent=2))
print("\n=== CUSTOM, RAROC-GATED (>=14%) point ===")
print(json.dumps(book(cust_raroc, E_cust, "pd_custom"), indent=2))

# band economics on the custom approved book (RAROC-gated)
print("\n=== BAND ECONOMICS (custom model, full test population) ===")
rows = []
for bnd in [1, 2, 3, 4, 5]:
    m = E_cust["band"] == bnd
    if m.sum() == 0: continue
    rows.append(dict(band=bnd, loans=int(m.sum()),
                     mean_pd=round(pd_cust[m].mean(), 4),
                     actual_default=round(y[m].mean(), 4),
                     ecl_rate=round((E_cust["el"][m].sum())/E_cust["ead"][m].sum(), 4),
                     raroc=round(E_cust["ni"][m].sum()/E_cust["econcap"][m].sum(), 4)))
print(pd.DataFrame(rows).to_string(index=False))

out = dict(
    incumbent=book(inc_mask, E_inc, "pd_incumbent"),
    custom_same_volume=book(cust_sv, E_cust, "pd_custom"),
    custom_same_default=book(cust_sd, E_cust, "pd_custom"),
    custom_raroc_gated=book(cust_raroc, E_cust, "pd_custom"),
    band_economics=rows,
)
json.dump(out, open(BASE + r"\04_outputs\module_a_economics.json", "w"), indent=2)
print("\nSaved module_a_economics.json")
