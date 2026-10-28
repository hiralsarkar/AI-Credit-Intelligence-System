# Module A - Application Risk

**Dataset:** Home Credit Default Risk  
**Signal output:** `pd_score`, `risk_band`, `credit_score`, `el`, `raroc`, `value_decision`

---

## What This Module Does

Module A estimates the **Probability of Default at the point of loan application** - the risk signal generated from information a borrower provides when they apply. It is the primary credit risk signal in the decisioning architecture.

This module goes beyond PD prediction to build a complete financial risk stack:
- PD estimation (Logistic Regression + XGBoost)
- Credit scorecard (WoE/IV, 300-900 points scale)
- Expected Loss, RWA, Economic Capital (Basel III)
- RAROC per borrower and risk band
- Strategy simulation (4 lending strategies)
- Stress testing (4 macro scenarios, break-even analysis)

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_preprocessing` | Data pipeline - missingness, imputation, encoding, train-test split |
| `02_credit_risk_model` | PD model - Logistic Regression + XGBoost, AUC/KS/Gini evaluation |
| `03_scorecard_analysis` | WoE scorecard - IV feature selection, points table, score-to-PD mapping |
| `04_expected_loss_capital_model` | EL, RWA, Economic Capital, RAROC per borrower |
| `05_strategy_simulator` | 4 lending strategies compared on risk-return frontier |
| `06_stress_testing` | 4 macro scenarios, RAROC matrix, break-even multipliers, monitoring triggers |
| `07_business_impact_analysis` | Custom model vs bureau-score benchmark on the business frontier (approval / default / net income) |
| `08_ready_reckoner` | Score-to-decision lookup for the credit committee |
| `09_vintage_analysis` | Cohort default curves by application vintage |
| `10_ifrs9_lifetime_pd_staging` | Origination-time PD-bucket proxy for provisioning (see note below) |

> **Note on staging:** the origination-time IFRS 9 view in `10_ifrs9_lifetime_pd_staging` is a **PD-bucket proxy**, not true SICR-based staging. A single application snapshot has no notion of an *increase* in credit risk, so it cannot produce a genuine Stage 2 population. Real, panel-based IFRS 9 staging lives in **Module D** (Freddie Mac monthly performance data). Use this notebook for a first-cut provisioning lens only.

## Signal Output Schema

```python
{
  "pd_score":       float,   # 0.0-1.0, XGBoost champion model
  "risk_band":      int,     # 1 (Very Low) - 5 (Very High)
  "credit_score":   int,     # 300-900, WoE scorecard
  "risk_grade":     str,     # "Very Low Risk" to "Very High Risk"
  "el":             float,   # Expected Loss (₹)
  "el_rate":        float,   # EL as % of EAD
  "econ_capital":   float,   # Economic Capital (₹)
  "raroc":          float,   # Risk-Adjusted Return on Capital
  "value_decision": str      # "Value Accretive" / "Marginal" / "Value Destructive"
}
```

## Dataset

**Home Credit Default Risk** - consumer loan applicants, application-time features, bureau scores.  
Download from: https://www.kaggle.com/c/home-credit-default-risk  
Place `application_train.csv` in `data/raw/`

## Robustness checks

`_robustness_a.py` asks how far the headline numbers can be trusted. It answers three questions a model validator would ask, and writes `04_outputs/module_a_robustness.json`.

| Check | What it does | Why it matters |
|---|---|---|
| **Ablation** | Fits a logistic regression on the same full feature set as XGBoost, then splits the AUC lift into a *data effect* (more features) and a *model effect* (XGBoost over logistic regression) | Shows whether the gain comes from the algorithm or from using richer data |
| **Calibration** | Reliability table, Brier score and expected calibration error, before and after isotonic calibration fitted on the validation set | RAROC, expected loss and the breakeven rate treat the PD as a true probability |
| **Honest operating point** | Chooses the SC3 approval point on a validation set, then applies its score threshold to the untouched test set | Removes the optimism of picking and reporting a cutoff on the same loans |
| **Uncertainty** | Paired, stratified bootstrap (1,000 resamples) gives 95% intervals on AUC, the ablation effects and every swap-set figure | Turns each headline into a defensible range |

The test set is the same 61,503 loans used by `_pipeline_a.py`; the remaining 80% is split 75/25 into train and validation, and nothing is fitted or tuned on test. The statistics live in `_robustness_lib.py` and are unit-tested on synthetic data in `tests/test_robustness.py`.

```bash
python 01_module_a_application_risk/_robustness_a.py          # or: python run_all.py --only a
BOOTSTRAP_N=200 python 01_module_a_application_risk/_robustness_a.py   # faster, wider intervals
```

## Status

✅ Complete
