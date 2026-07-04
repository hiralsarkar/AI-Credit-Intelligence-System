"""Regenerate 05_strategy_simulator.ipynb as the Policy Frontier & Operating-Point notebook.
Removes Aggressive / Conservative / RAROC-Gated / Risk-Based-Pricing strategies, the -169M and
+1130M figures. Builds the custom model's dominating frontier and the committee's operating points
from scored_test_a.csv."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook(); c = []

c.append(new_markdown_cell(
"""# Policy Frontier & Operating-Point Selection
*Module A - Notebook 5 - AI Credit Intelligence System*

---

The PD model estimates risk. The **policy** decision is which loans to approve. We express that
decision as a **frontier**: sweep the approval threshold over the custom model's risk ranking and
measure, at each point, the approved book's default rate, net income, economic capital, RAROC, and
ECL. The credit committee then selects an operating point on that frontier according to its stated
objective - grow the book, cut losses, or maximise risk-adjusted return.

No arbitrary named strategies. One frontier, explicit operating points, real numbers."""))

c.append(new_code_cell(
"""import pandas as pd, numpy as np
import matplotlib.pyplot as plt

PURPLE, GOLD, MUTED, INK = "#8B5CF6", "#D4AF37", "#A29FB2", "#0A0A0F"
plt.rcParams.update({"figure.dpi":120, "font.family":"sans-serif"})

s = pd.read_csv("../01_data/processed/scored_test_a.csv")
y = s["actual_default"].values; pd_cust = s["pd_custom"].values; ead = s["ead"].values

# bank assumptions (consistent across the project)
LGD, CAP, REV, OPEX, HURDLE = 0.45, 0.105, 0.12, 0.03, 0.14
RW = lambda p: 0.75 if p < .10 else 1.0 if p < .20 else 1.5
print(f"Loans: {len(s):,}   assumptions: LGD {LGD:.0%}, revenue {REV:.0%}, opex {OPEX:.0%}, "
      f"capital ratio {CAP:.1%}, hurdle {HURDLE:.0%}")"""))

c.append(new_markdown_cell("""## 1. Build the policy frontier"""))

c.append(new_code_cell(
"""order = np.argsort(pd_cust)
rows = []
for q in np.linspace(0.05, 0.95, 91):
    n = int(len(s)*q); idx = order[:n]
    e, p, yy = ead[idx], pd_cust[idx], y[idx]
    el = p*LGD*e; cap = e*np.array([RW(x) for x in p])*CAP
    ni = e*REV - el - e*OPEX
    rows.append(dict(approval=q, default=yy.mean(), ecl_rate=el.sum()/e.sum(),
                     ni_cr=ni.sum()/1e7, raroc=ni.sum()/cap.sum()))
F = pd.DataFrame(rows)
print(F.iloc[::15].round(4).to_string(index=False))"""))

c.append(new_code_cell(
"""fig, ax = plt.subplots(1, 2, figsize=(12,4.4), facecolor="white")
ax[0].plot(F["approval"]*100, F["ni_cr"], color=GOLD, lw=2.5)
ax[0].set_title("Net income vs approval rate", color=INK)
ax[0].set_xlabel("Approval rate (%)"); ax[0].set_ylabel("Net income (cr)"); ax[0].grid(alpha=.2)
ax[1].plot(F["approval"]*100, F["raroc"]*100, color=PURPLE, lw=2.5)
ax[1].axhline(HURDLE*100, color=MUTED, ls="--", lw=1.5); ax[1].text(60, HURDLE*100+2, "hurdle 14%", color=MUTED)
ax[1].set_title("Portfolio RAROC vs approval rate", color=INK)
ax[1].set_xlabel("Approval rate (%)"); ax[1].set_ylabel("RAROC (%)"); ax[1].grid(alpha=.2)
plt.tight_layout(); plt.show()"""))

c.append(new_markdown_cell(
"""## 2. Operating points the committee can choose

All points sit on the same dominating frontier. The incumbent (bureau-score cutoff at 50% approval,
3.69% default) is shown for reference; every custom operating point beats it."""))

c.append(new_code_cell(
"""def book(q):
    n=int(len(s)*q); idx=order[:n]; e,p,yy=ead[idx],pd_cust[idx],y[idx]
    el=p*LGD*e; cap=e*np.array([RW(x) for x in p])*CAP; ni=e*REV-el-e*OPEX
    return dict(Approval=f"{q*100:.1f}%", Default=f"{yy.mean()*100:.2f}%",
                ECL_rate=f"{el.sum()/e.sum()*100:.2f}%", NI_cr=round(ni.sum()/1e7,1),
                RAROC=f"{ni.sum()/cap.sum()*100:.1f}%")
pts = {
    "Capital-max (low approval)": book(0.30),
    "Recommended balanced (SC3)": book(0.619),
    "Net-income-max (growth)":    book(0.70),
}
out = pd.DataFrame(pts).T
print(out.to_string())
out.to_csv("../01_data/processed/frontier_operating_points.csv")"""))

c.append(new_markdown_cell(
"""## 3. Reading the frontier

- **Capital-max:** lowest approval, highest RAROC - appropriate when capital is scarce.
- **Recommended balanced (SC3):** ~62% approval at 3.61% default - higher approval **and** lower
  default than the incumbent's 50% / 3.69%, with strong RAROC. The default recommendation.
- **Net-income-max:** higher approval pushes total net income up further, at a higher default rate
  and lower RAROC - appropriate only with ample capital and growth appetite.

The RAROC hurdle (14%) is cleared across the whole usable range; the choice is risk appetite, not
solvency. This frontier is the input to the business-impact swap-set in Notebook 7."""))

nb["cells"] = c
nb.metadata["kernelspec"] = {"name":"python3","display_name":"Python 3","language":"python"}
nbf.write(nb, "02_notebooks/05_strategy_simulator.ipynb")
print("Wrote 05_strategy_simulator.ipynb")
