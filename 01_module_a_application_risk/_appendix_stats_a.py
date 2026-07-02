"""Technical-appendix statistics for Module A: feature importance, calibration deciles,
PSI (train vs test), precision/recall at the recommended cutoff. Real numbers for the doc."""
import warnings, json; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os
from pathlib import Path
_HERE = Path(__file__).resolve().parent
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier

BASE = str(_HERE)
df = pd.read_csv(BASE + r"\01_data\raw\application_train.csv")
y = df["TARGET"].values
drop = {"TARGET", "SK_ID_CURR"}
num = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]
tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
ytr, yte = y[tr], y[te]

xgb = XGBClassifier(n_estimators=450, max_depth=4, learning_rate=0.05, subsample=0.9,
                    colsample_bytree=0.8, reg_lambda=1.0, eval_metric="auc", random_state=42, n_jobs=4)
xgb.fit(df.iloc[tr][num], ytr)
p_tr = xgb.predict_proba(df.iloc[tr][num])[:, 1]
p_te = xgb.predict_proba(df.iloc[te][num])[:, 1]

def ks(y_, s):
    o=np.argsort(s); yy=y_[o]; cb=np.cumsum(yy)/yy.sum(); cg=np.cumsum(1-yy)/(1-yy).sum(); return float(np.max(np.abs(cb-cg)))
auc=roc_auc_score(yte,p_te)
print(f"AUC {auc:.4f}  KS {ks(yte,p_te):.4f}  Gini {2*auc-1:.4f}")

# feature importance (gain), top 12
imp = pd.Series(xgb.get_booster().get_score(importance_type="gain"))
imp = imp.sort_values(ascending=False).head(12)
print("\nTop features (gain):")
for k,v in imp.items(): print(f"  {k}: {v:.0f}")

# decile calibration on test
dfc = pd.DataFrame({"pd":p_te,"y":yte}).sort_values("pd")
dfc["dec"] = pd.qcut(dfc["pd"], 10, labels=False, duplicates="drop")
cal = dfc.groupby("dec").agg(mean_pd=("pd","mean"), actual=("y","mean"), n=("y","size"))
print("\nDecile calibration (predicted PD vs actual):")
for d,r in cal.iterrows(): print(f"  D{int(d)+1}: pred {r.mean_pd:.3f}  actual {r.actual:.3f}  n {int(r.n)}")

# PSI train vs test on score deciles (expected=train, actual=test)
edges = np.quantile(p_tr, np.linspace(0,1,11)); edges[0],edges[-1]=-1,2
e = np.histogram(p_tr, edges)[0]/len(p_tr); a = np.histogram(p_te, edges)[0]/len(p_te)
psi = float(np.sum((a-e)*np.log((a+1e-6)/(e+1e-6))))
print(f"\nPSI (train vs test score distribution): {psi:.4f}")

# precision / recall at recommended cutoff (decline the highest-PD tail; approve lowest 61.9%)
thr = np.quantile(p_te, 0.619)        # approve pd <= thr
decline = (p_te > thr).astype(int)    # positive class = decline = predicted bad
prec = precision_score(yte, decline); rec = recall_score(yte, decline)
print(f"\nAt recommended cutoff (approve lowest 61.9% PD): of declined applicants {prec:.1%} are true defaults (precision); "
      f"{rec:.1%} of all defaults are declined (recall)")

out = dict(auc=round(auc,4), ks=round(ks(yte,p_te),4), gini=round(2*auc-1,4),
           feature_importance={k:round(float(v),1) for k,v in imp.items()},
           calibration=[{"decile":int(d)+1,"pred":round(float(r.mean_pd),4),"actual":round(float(r.actual),4)} for d,r in cal.iterrows()],
           psi=round(psi,4), precision_decline=round(float(prec),4), recall_decline=round(float(rec),4))
json.dump(out, open(BASE + r"\04_outputs\module_a_appendix_stats.json","w"), indent=2)
print("\nSaved module_a_appendix_stats.json")
