# Module B — Behavioural Risk
*AI Credit Intelligence System · Second of Three Risk Modules*

**Dataset:** Give Me Some Credit (Kaggle)  
**Signal output:** `delinquency_prob`, `behavioural_risk_band`, `behavioural_risk_score`, `stress_flags`, `shap_top_drivers`

---

## What This Module Does

Module A (Application Risk) captures risk at the **point of application** — a snapshot of who the borrower appears to be.

Module B captures a fundamentally different signal: **how this borrower has actually behaved with credit over the past two years.** This is the difference between what a borrower claims and what they do.

A borrower may present a clean application — stable income, no recent bureau flags — and still carry serious behavioural risk: maxed-out credit cards, a history of 90-day late payments, or an escalating delinquency pattern. Module A cannot see these. Module B is specifically designed to surface them.

In the Decision Engine, the two signals are combined:
- **Agreement between A and B** → high-confidence decision
- **Divergence between A and B** → manual review (the interesting cases)

---

## Notebooks

| Notebook | Description | Key Outputs |
|----------|-------------|-------------|
| `01_data_preprocessing_b` | Raw data profiling, sentinel removal, outlier treatment, feature engineering, 80/20 split | `clean_behavioural.csv`, train/test splits |
| `02_delinquency_model_b` | LR + XGBoost PD models, AUC/KS/Gini evaluation, risk banding (1–5), BRS (0–100) | `scored_test_b.csv`, both model `.pkl` files |
| `03_behavioural_scorecard_b` | WoE/IV scorecard (0–100 scale), stress indicator flags, score-to-risk mapping table | `scorecard_output_b.csv`, `behavioural_scorecard.pkl` |
| `04_shap_explainability_b` | Global + local SHAP, reason code formatter, adverse action notice generator | `shap_explainer_b.pkl`, `shap_values_b.csv` |

---

## Dataset — Give Me Some Credit

| Property | Value |
|----------|-------|
| Source | Give Me Some Credit — Kaggle Competition |
| Population | US consumer credit borrowers |
| Observation window | 2-year look-forward |
| Target | `SeriousDlqin2yrs` = 1 (90+ day delinquency in 2 years) |
| Rows | 150,000 (after cleaning: ~149,471) |
| Default rate | 6.68% |
| Missing values | `MonthlyIncome` (19.8%), `NumberOfDependents` (2.6%) |

**Place `cs-training.csv` in `01_data/raw/` before running.**

---

## Key Data Findings

1. **Delinquency history is the dominant signal** — borrowers with any 90-day late event default at 33.7% vs 4.6% for clean borrowers. Lift of ~7×.
2. **Revolving utilisation is strongly monotonic** — default rate rises from 1.9% (0–20% utilisation) to 21.1% (80–100%). Every 20pp increase in utilisation approximately doubles default risk.
3. **Age has a strong protective effect** — 18–25 year olds default at 11.2%; 65+ at 2.4%. Reflects credit history maturity.
4. **Escalating severity matters more than frequency** — a single 90-day late is more predictive than three 30-day lates.

---

## Engineered Features

| Feature | Formula | Predictive Signal |
|---------|---------|-------------------|
| `DELINQUENCY_SCORE` | Sum of all DPD counts | Cumulative severity |
| `WORST_DELINQUENCY` | Max severity (0–3) | Whether 90DPD threshold was breached |
| `HAS_90DAY_LATE` | Binary: ever 90+ DPD | Single strongest predictor |
| `UTILIZATION_RISK_BAND` | Ordinal 0–4 from utilisation | WoE-friendly encoding |
| `DEBT_TO_INCOME` | DebtRatio × income proxy | Repayment capacity |
| `INCOME_PER_DEPENDENT` | Income / (dependents + 1) | Disposable income |
| `CREDIT_LINES_RISK` | Binary: < 3 open lines | Thin-file flag |
| `INCOME_MISSING` | Binary: income undisclosed | Data quality + risk marker |

---

## Signal Output Schema

```python
# Module B signal — feeds into Decision Engine (Module D)
{
  # Core risk signal
  "delinquency_prob":        float,  # 0.0–1.0   P(90DPD in 2yr) — XGBoost champion
  "behavioural_risk_band":   int,    # 1–5       (1=Very Low, 5=Very High)
  "behavioural_risk_score":  float,  # 0–100     higher = safer

  # Stress flags (hard overrides in Decision Engine)
  "stress_flag":             bool,   # Has 90DPD OR delinquency_score ≥ 3
  "critical_utilisation":    bool,   # Revolving utilisation ≥ 80%
  "escalating_delinquency":  bool,   # Has 60DPD or 90DPD events
  "high_delinquency_score":  bool,   # Delinquency score ≥ 5 (repeated pattern)

  # SHAP explanation (for adverse action notices)
  "top_risk_drivers": [
      {"reason_code": "History of 90+ day past-due payment",  "shap_value": 0.342},
      {"reason_code": "High revolving credit utilisation",     "shap_value": 0.187},
      {"reason_code": "High cumulative delinquency history",   "shap_value": 0.143},
  ]
}
```

---

## Regulatory Framework

| Requirement | How Module B Addresses It |
|-------------|--------------------------|
| SR 11-7 (conceptual soundness) | WoE monotonicity checks; feature direction validation vs domain knowledge |
| SR 11-7 (ongoing monitoring) | PSI computed in NB02; SHAP values stored for drift monitoring |
| RBI Fair Practices Code | Reason code formatter produces compliant adverse action notices |
| ECOA/FCRA (adverse action) | Top-3 SHAP reason codes formatted as plain-English explanations |
| Basel III (model use test) | Delinquency signal feeds capital calculation in Module A via composite PD |

---

## Status

✅ **Complete** — all 4 notebooks implemented and documented.

**Next:** Module C (Portfolio & Pricing Risk) — LendingClub dataset
