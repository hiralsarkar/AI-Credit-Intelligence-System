"""Regenerate 08_ready_reckoner.ipynb: a clean risk-band -> decision lookup table built from the
custom model's real band economics. Removes references to the old named strategies."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook(); c = []

c.append(new_markdown_cell(
"""# Ready Reckoner - Risk Band to Decision Lookup
*Module A - Notebook 8 - AI Credit Intelligence System*

---

A one-page operational lookup: for each risk band it gives the mean PD, RAROC, ECL rate, IFRS 9
stage, and the resulting underwriting decision. It lets a credit officer or pricing manager act on
the model without rerunning it. All figures are the custom model's real, calibrated band economics."""))

c.append(new_code_cell(
"""import pandas as pd, numpy as np
import matplotlib.pyplot as plt
PURPLE, GOLD, MUTED, RISK, INK = "#8B5CF6", "#D4AF37", "#A29FB2", "#C45B5B", "#0A0A0F"
plt.rcParams.update({"figure.dpi":120, "font.family":"sans-serif"})

s = pd.read_csv("../01_data/processed/scored_test_a.csv")
y = s["actual_default"].values; p = s["pd_custom"].values; ead = s["ead"].values
LGD, CAP, REV, OPEX, HURDLE = 0.45, 0.105, 0.12, 0.03, 0.14
RW = lambda x: 0.75 if x < .10 else 1.0 if x < .20 else 1.5
band = np.where(p<.05,1,np.where(p<.10,2,np.where(p<.20,3,np.where(p<.40,4,5))))"""))

c.append(new_markdown_cell("""## 1. Build the ready reckoner"""))

c.append(new_code_cell(
"""names={1:"B1 Very Low",2:"B2 Low",3:"B3 Medium",4:"B4 High",5:"B5 Very High"}
rows=[]
for b in [1,2,3,4,5]:
    m=band==b
    if m.sum()==0: continue
    e=ead[m]; pb=p[m]
    el=pb*LGD*e; cap=e*np.array([RW(x) for x in pb])*CAP; ni=e*REV-el-e*OPEX
    raroc=ni.sum()/cap.sum()
    stage = "Stage 1" if b<=3 else "Stage 2/3"
    if raroc>=HURDLE and b<=2: dec="APPROVE"
    elif raroc>=HURDLE and b==3: dec="APPROVE (review)"
    else: dec="DECLINE"
    rows.append(dict(Band=names[b], Loans=int(m.sum()), Mean_PD=f"{pb.mean()*100:.1f}%",
                     Actual_default=f"{y[m].mean()*100:.1f}%", RAROC=f"{raroc*100:.1f}%",
                     ECL_rate=f"{el.sum()/e.sum()*100:.2f}%", IFRS9=stage, Decision=dec))
RR=pd.DataFrame(rows); print(RR.to_string(index=False))"""))

c.append(new_markdown_cell("""## 2. Decision matrix"""))

c.append(new_code_cell(
"""fig,ax=plt.subplots(figsize=(9,3.6),facecolor="white"); ax.axis("off")
cells=[]; colors=[]
for _,r in RR.iterrows():
    cells.append([r["Band"],r["Mean_PD"],r["RAROC"],r["ECL_rate"],r["IFRS9"],r["Decision"]])
    colors.append(["#F4F1FB"]*5 + [("#EAD9A0" if "APPROVE" in r["Decision"] else "#EBC9C9")])
t=ax.table(cellText=cells, colLabels=["Band","Mean PD","RAROC","ECL rate","IFRS 9","Decision"],
           cellColours=colors, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,1.6)
ax.set_title("Underwriting decision by risk band", color=INK, pad=12); plt.tight_layout(); plt.show()"""))

c.append(new_code_cell(
"""RR.to_csv("../01_data/processed/ready_reckoner.csv", index=False)
print("Saved ready_reckoner.csv")"""))

c.append(new_markdown_cell(
"""## 3. How to use it

- **Credit operations:** read the Decision column. Bands 1-2 approve, Band 3 approves on review,
  Bands 4-5 decline (RAROC negative, capital-destructive).
- **Pricing:** RAROC by band shows headroom; higher-risk approved bands carry the smallest margin.
- **Risk committee:** the ECL rate and IFRS 9 column show the provisioning intensity each band adds.

Refresh whenever the PD model is recalibrated (at least annually, or on a PSI breach)."""))

nb["cells"]=c
nb.metadata["kernelspec"]={"name":"python3","display_name":"Python 3","language":"python"}
nbf.write(nb,"02_notebooks/08_ready_reckoner.ipynb"); print("Wrote 08_ready_reckoner.ipynb")
