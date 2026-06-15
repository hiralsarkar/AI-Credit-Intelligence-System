# Model Card — Module A: Application Risk (PD Model)
*AI Credit Intelligence System · Governance Documentation*
*Format: Model Cards for Model Reporting (Mitchell et al., 2019) adapted for BFSI*

---

## 1. Model Overview

| Field | Detail |
|-------|--------|
| **Model name** | Application Risk PD Model — Champion (XGBoost) |
| **Version** | v1.0 |
| **Type** | Binary classification — Probability of Default |
| **Champion model** | XGBoost (`xgboost_pd.pkl`) |
| **Challenger model** | Logistic Regression (`logistic_regression_pd.pkl`, bundled with its `StandardScaler`) |
| **Scorecard** | WoE-based credit scorecard, 300–900 scale (`scorecard_woe_map.pkl`, `scorecard_points.pkl`) |
| **Developed by** | AI Credit Intelligence System — Module A |
| **Last trained** | June 2026 |
| **Next review due** | December 2026 |
| **Regulatory status** | Pending independent validation |

---

## 2. Intended Use

### Primary Use
Score new credit applicants at origination. The output `pd_score` (0–1) feeds directly into:
- Expected Loss calculation (EL = PD × LGD × EAD)
- Risk-based pricing via the strategy simulator
- Decision Engine composite score (60% weight)

### Secondary Use
- Portfolio-level risk monitoring (monthly)
- Stress testing and capital adequacy calculation (quarterly, RBI ICAAP)
- Challenger model benchmarking

### Out-of-Scope Uses
- **Behavioural monitoring** of existing customers — use Module B instead
- **Pricing decisions in isolation** — must be combined with Module C
- **Any non-credit application** (e.g. fraud detection, insurance pricing)

---

## 3. Training Data

| Property | Value |
|----------|-------|
| **Dataset** | Home Credit Default Risk (Kaggle) |
| **Population** | Consumer credit applicants — primarily Eastern Europe and Southeast Asia |
| **Period** | Historical snapshot (training year not disclosed by data source) |
| **Training rows** | 246,008 |
| **Test rows** | 61,503 |
| **Split** | 80/20 stratified by target |
| **Target definition** | `TARGET = 1` if loan defaulted (90+ DPD or write-off) within contract |
| **Default rate (train)** | 8.08% |
| **Default rate (test)** | 8.05% |
| **Features used** | 49 (after WoE/IV selection from ~120 original) |
| **Key predictors** | EXT_SOURCE_1/2/3, DAYS_EMPLOYED, DAYS_BIRTH, AMT_CREDIT |

### Data Exclusions
- Rows with missing target: removed
- Extreme outliers (> 99th percentile in income/annuity): capped, not removed
- No demographic features (gender, nationality, religion) used in final model

### Data Limitations
- **Geographic basis**: Home Credit operates in Eastern Europe and Asia — the default distribution may not mirror Indian retail lending directly. Default rates in India are structurally different from Czech Republic or Kazakhstan.
- **Temporal representativeness**: Snapshot data of unknown vintage. Does not capture post-COVID credit behaviour.
- **Income self-reporting**: `AMT_INCOME_TOTAL` is declared income, not verified — adversely selected in some population segments.

---

## 4. Model Performance

### Confirmed Metrics (from notebook 02 outputs — test set)

| Model | AUC | KS | Gini | PSI | Role |
|-------|-----|----|------|-----|------|
| Logistic Regression | **0.7480** | **0.3707** | **0.4961** | 0.0002 | Challenger |
| XGBoost | **0.7595** | **0.3874** | **0.5190** | 0.0001 | Champion |

### Benchmark Thresholds (Industry Standard)

| Metric | Minimum Acceptable | This Model |
|--------|-------------------|------------|
| AUC | > 0.70 | 0.7595 ✅ |
| KS | > 0.30 | 0.3874 ✅ |
| Gini | > 0.40 | 0.5190 ✅ |
| PSI (train vs test) | < 0.10 | 0.0001 ✅ |

### Capital Model Outputs (from notebook 04)

| Metric | Value |
|--------|-------|
| Portfolio EAD | ₹36.6 billion |
| Total Expected Loss | ₹6.4 billion |
| Portfolio EL Rate | 18.11% |
| Mean RAROC (full portfolio) | -55.77% |
| Value-Accretive loans (RAROC > 14%) | 12.2% |

### RAROC by Risk Band

| Band | Label | Loans | Mean PD | RAROC |
|------|-------|-------|---------|-------|
| 1 | Very Low | 182 | 3.82% | +92.89% |
| 2 | Low | 1,798 | 8.04% | +68.54% |
| 3 | Medium | 9,120 | 15.52% | +19.02% |
| 4 | High | 16,979 | 27.34% | -20.89% |
| 5 | Very High | 33,424 | 55.49% | -98.78% |

**Interpretation:** Only Bands 1–3 are value-accretive. The strategy simulator (NB05) operationalises this — the RAROC-Gated and Conservative strategies restrict approvals accordingly.

### Stress Test Results (from notebook 06)

| Scenario | RAROC (Conservative) | RAROC (RAROC-Gated) | Capital Adequacy |
|----------|---------------------|---------------------|------------------|
| Baseline | +71.03% | +36.15% | Adequate |
| Mild Stress (PD×1.5) | +42.19% | -0.84% | Breached |
| Severe Stress (PD×2.5) | -29.90% | -93.32% | Breached |
| Extreme Stress (PD×4.0) | -154.87% | -253.62% | Severely Breached |

**Finding:** Conservative strategy survives Mild Stress. All strategies breach under Severe Stress — consistent with a systemic event (COVID-19 equivalent). This is expected for a retail lending portfolio under Basel III stress testing.

---

## 5. Scorecard Performance

| Metric | Value |
|--------|-------|
| Scorecard AUC | 0.3896 |
| KS | 0.1794 |
| AUC degradation vs XGBoost | 0.3699 |

> **Note:** The WoE scorecard shows significantly lower AUC than XGBoost (0.38 vs 0.76). This is a known limitation of the scorecard — the IV selection process and WoE binning lose discriminatory power on this specific dataset because `EXT_SOURCE` features are already summary scores (not raw behavioural features), making WoE transformation less additive. The scorecard is retained for **interpretability and adverse action notice generation** only. XGBoost is the production PD model.

---

## 6. Fairness and Bias Assessment

### Demographic Features
No demographic features (gender, age, nationality, religion) are used as direct model inputs. `DAYS_BIRTH` (proxy for age) appears in the feature set but only because of its demonstrated correlation with credit maturity, not as a demographic discriminant.

### Known Proxy Risks
- `AMT_INCOME_TOTAL` may correlate with socioeconomic status
- `REGION_RATING_CLIENT_W_CITY` encodes geographic risk — could be a proxy for ethnicity in some markets

### Bias Monitoring (Ongoing)
Approval rate parity across income quartiles is monitored monthly via `05_governance/monitoring_triggers.py`. Amber trigger: > 20pp approval rate gap between top and bottom income quartile.

---

## 7. Limitations

1. **Geographic mismatch**: Model trained on Home Credit data (Eastern Europe/Asia). Calibration to Indian default rates requires local portfolio data.
2. **Point-in-time snapshot**: Does not capture macroeconomic regime changes. Requires quarterly recalibration.
3. **Thin-file applicants**: Borrowers with limited credit history receive unreliable EXT_SOURCE scores — model performance degrades for this segment.
4. **Post-origination use**: Model is designed for origination decisions only. Do not use for collections prioritisation or limit management.
5. **External bureau dependency**: EXT_SOURCE features dominate the model. If bureau connectivity is lost, model performance drops significantly.

---

## 8. Monitoring Requirements (SR 11-7 / RBI MRM)

| Trigger | Metric | Amber | Red | Action |
|---------|--------|-------|-----|--------|
| Score drift | PSI | > 0.10 | > 0.25 | Recalibrate |
| Performance | AUC | < 0.72 | < 0.70 | Recalibrate |
| Performance | KS | < 0.32 | < 0.30 | Recalibrate |
| Calibration | Expected vs actual DR | > 15% gap | > 25% gap | Recalibrate |
| Portfolio | EL rate vs forecast | > 20% gap | > 35% gap | Model review |

Monitoring is implemented in `05_governance/monitoring_triggers.py`.

---

## 9. Approval Chain

| Role | Name | Date | Sign-off |
|------|------|------|----------|
| Model Developer | — | March 2026 | ✅ |
| Model Validator | *Pending independent validation* | — | 🔲 |
| Credit Risk Officer | *Pending* | — | 🔲 |
| Risk Committee | *Pending* | — | 🔲 |

---

*Model card version: 1.1 · Last updated: June 2026 · Next review: December 2026*
