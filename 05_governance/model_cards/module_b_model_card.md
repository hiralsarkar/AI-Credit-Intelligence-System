# Model Card — Module B: Behavioural Risk (Delinquency Model)
*AI Credit Intelligence System · Governance Documentation*

---

## 1. Model Overview

| Field | Detail |
|-------|--------|
| **Model name** | Behavioural Risk Delinquency Model — Champion (XGBoost) |
| **Version** | v1.0 |
| **Type** | Binary classification — P(90-day delinquency within 2 years) |
| **Champion model** | XGBoost (`xgb_behavioural.pkl`) |
| **Challenger model** | Logistic Regression (`lr_behavioural.pkl`) |
| **Scorecard** | WoE-based behavioural scorecard, 0–100 scale (`behavioural_scorecard.pkl`) |
| **Explainer** | SHAP TreeExplainer (`shap_explainer_b.pkl`) |
| **Developed by** | AI Credit Intelligence System — Module B |
| **Last trained** | March 2026 |
| **Next review due** | September 2026 |

---

## 2. Intended Use

### Primary Use
Assess the behavioural risk of a borrower based on their observed payment history over the prior 2 years. Output `delinquency_prob` (0–1) feeds into:
- Decision Engine composite score (40% weight)
- Hard override flags (stress_flag, critical_utilisation, high_delinquency_score)
- Adverse action reason code generation via SHAP

### Why This Model Exists Alongside Module A
Module A captures risk at the **point of application** — a static snapshot. A borrower who looked safe at origination can deteriorate financially within months. Module B catches this:
- A clean application PD (Module A) + high delinquency probability (Module B) = deteriorating borrower, flag for review
- A moderate application PD + clean behavioural history = stable borrower, potentially approvable

### Out-of-Scope Uses
- **New-to-credit applicants** with no 2-year payment history — use Module A fallback
- **Fraud detection** — behavioural patterns here are credit risk signals, not fraud signals
- **Collections scoring** — while related, collections prioritisation requires a separate model

---

## 3. Training Data

| Property | Value |
|----------|-------|
| **Dataset** | Give Me Some Credit (Kaggle) |
| **Population** | US consumer credit borrowers |
| **Observation window** | 2-year forward-looking period |
| **Training rows** | ~119,577 (80% of cleaned dataset) |
| **Test rows** | ~29,894 (20%) |
| **Split** | 80/20 stratified by target |
| **Target definition** | `SeriousDlqin2yrs = 1` (90+ day delinquency in 2 years) |
| **Default rate** | 6.68% |
| **Imbalance ratio** | ~14:1 (non-delinquent to delinquent) |

### Key Features

| Feature | Type | IV | Notes |
|---------|------|----|-------|
| `HAS_90DAY_LATE` | Engineered | High | Binary: ever 90+ DPD |
| `NumberOfTimes90DaysLate` | Raw | High | Count of 90DPD events |
| `DELINQUENCY_SCORE` | Engineered | High | Sum of all DPD counts |
| `RevolvingUtilizationOfUnsecuredLines` | Raw | Medium | Capped at 1.0 |
| `age` | Raw | Medium | Negative relationship (older = safer) |
| `DEBT_TO_INCOME` | Engineered | Medium | DebtRatio × income proxy |
| `INCOME_MISSING` | Engineered | Low | Missingness indicator flag |

### Data Cleaning Decisions

| Issue | Treatment | Rationale |
|-------|-----------|-----------|
| Sentinel values (96/98 in DPD cols) | Removed 264 rows | Non-physical counts — data error codes |
| Age = 0 | Removed 1 row | Invalid record |
| Revolving utilisation > 1.0 | Capped at 1.0 | Values of 50,000 are data errors; >100% is over-limit (meaningful, kept) |
| Debt ratio > 99th pct | Capped | Extreme values reflect zero-income denominator distortion |
| MonthlyIncome missing (19.8%) | Median impute + binary flag | Absence of disclosure is itself a signal |
| NumberOfDependents missing (2.6%) | Median impute | Small sample, weak predictor |

### Data Limitations
- **US market basis**: Default patterns in US consumer credit differ from India (different macro environment, credit culture, bureau coverage)
- **No bureau integration**: Dataset relies on self-reported and derived features — in production, bureau pull would replace much of this
- **Snapshot data**: 2-year forward window fixed at one point in time — does not capture regime changes

---

## 4. Model Performance

### Confirmed Metrics (from notebook 02 outputs — test set, n=29,946, default rate 6.6%)

| Model | AUC | KS | Gini | Role |
|-------|-----|----|------|------|
| Logistic Regression | **0.8518** | **0.5474** | **0.7036** | Challenger |
| XGBoost | **0.8571** | **0.5630** | **0.7142** | Champion |

Both models clear the design thresholds (AUC > 0.82, KS > 0.45). XGBoost (`xgb_behavioural.pkl`) is the champion used for `delinquency_prob`; the WoE scorecard (`behavioural_scorecard.pkl`) is retained for interpretability and adverse action notices.

### Key Validation Findings (from EDA in NB01)

| Segment | Delinquency Rate | Lift vs Average |
|---------|-----------------|-----------------|
| Has 90-day late history | ~33.7% | 5.0× |
| No delinquency history | ~4.6% | 0.7× |
| Utilisation 80–100% | ~21.1% | 3.2× |
| Utilisation 0–20% | ~1.9% | 0.3× |
| Age 18–25 | ~11.2% | 1.7× |
| Age 65+ | ~2.4% | 0.4× |

### Stress Flags Validation

| Flag | % Portfolio Flagged | Expected DR Lift |
|------|---------------------|-----------------|
| `STRESS_FLAG` | ~15–20% | 3–4× |
| `CRITICAL_UTILISATION` | ~25% | 2–3× |
| `HIGH_DELINQUENCY_SCORE` | ~5–8% | 5–7× |
| `ESCALATING_DELINQUENCY` | ~8–12% | 4–6× |

---

## 5. WoE Scorecard (0–100 Scale)

The WoE scorecard produces a `behavioural_score` (0–100) where **higher = safer**.

| Score Band | Risk Grade | Expected Delinquency Prob | Decision Engine Action |
|------------|------------|--------------------------|------------------------|
| 80–100 | Very Low Risk | < 3% | No adverse action |
| 60–79 | Low Risk | 3–7% | No adverse action |
| 40–59 | Moderate Risk | 7–15% | Flag for monitoring |
| 20–39 | High Risk | 15–30% | Escalate to manual review |
| 0–19 | Critical Risk | > 30% | Override — decline recommended |

**Design note:** The 0–100 scale is intentionally distinct from Module A's 300–900 scale. They measure different concepts (behavioural vs application-time risk) and must not be summed or confused.

---

## 6. SHAP Explainability

Module B uses `shap.TreeExplainer` for exact attribution. For every decision:
- **Global**: Top features by mean |SHAP| across portfolio
- **Local**: For each borrower, ranked list of features with signed contribution
- **Reason codes**: SHAP values converted to regulatory-compliant plain-English adverse action reasons

Top 3 expected global SHAP drivers:
1. `HAS_90DAY_LATE` / `NumberOfTimes90DaysLate` — dominant signal
2. `RevolvingUtilizationOfUnsecuredLines` — second strongest continuous predictor
3. `DELINQUENCY_SCORE` — cumulative severity

---

## 7. Limitations

1. **Thin-file gap**: If a borrower has < 2 years of credit history, the model falls back to population average (`delinquency_prob = 0.067`). This is a known degraded-performance zone.
2. **US market calibration**: Model requires recalibration against Indian behavioural data before production deployment.
3. **Temporal decay**: Payment behaviour patterns can shift quickly (e.g. during COVID). PSI monitoring is the primary detection mechanism.
4. **No real-time update**: The model uses a 2-year lookback window. Rapid deterioration within the past 6 months may be underweighted.
5. **Dependent feature structure**: `DELINQUENCY_SCORE` is derived from the same DPD columns that feed `HAS_90DAY_LATE` — some correlation between features is by design, not a modelling error.

---

## 8. Monitoring Requirements

| Trigger | Metric | Amber | Red | Action |
|---------|--------|-------|-----|--------|
| Score drift | PSI | > 0.10 | > 0.25 | Recalibrate |
| Performance | AUC | < 0.80 | < 0.78 | Recalibrate |
| Flag rate | Stress flag % | > 30% | > 45% | Portfolio review |
| Override rate | Hard override % | > 15% | > 25% | Policy review |

---

*Model card version: 1.0 · Last updated: March 2026 · Next review: September 2026*
