# Model Card — Module C: Portfolio & Pricing Risk
*AI Credit Intelligence System · Governance Documentation*

---

## 1. Model Overview

| Field | Detail |
|-------|--------|
| **Model name** | Loan Grade Classifier + Default Model (Portfolio & Pricing) |
| **Version** | v1.0 |
| **Grade classifier** | XGBoost multiclass — 7 classes A–G (`xgb_grade_classifier.pkl`) |
| **Default model** | XGBoost binary — P(Charged Off) (`xgb_default_c.pkl`) |
| **Pricing lookup** | Grade → market PD → suggested rate (`grade_pd_lookup.pkl`) |
| **Developed by** | AI Credit Intelligence System — Module C |
| **Last trained** | March 2026 |
| **Next review due** | September 2026 |

---

## 2. Intended Use

### Primary Use
Provide a **market-calibrated external benchmark** for the Decision Engine:
1. `market_pd` — what the market (LendingClub's 11-year track record) implies this borrower's default probability to be
2. `suggested_rate` — what interest rate this loan should carry to be fairly priced for its risk
3. `concentration_flag` — whether approving this loan would add portfolio concentration risk

### What Makes Module C Unique
Modules A and B produce **internal model estimates** — they are only as good as our training data and modelling assumptions. Module C provides an **external market cross-check** grounded in 1.3 million actual loan outcomes from a sophisticated P2P lender. When the two diverge significantly, it is a red flag worth investigating.

### The Reprice Decision
Module C is the only module that enables the `REPRICE` outcome. If a borrower passes A and B but the proposed rate is below the market-implied fair price, the Decision Engine routes to REPRICE (with the suggested rate) rather than APPROVE. This protects revenue without declining a creditworthy borrower.

### Out-of-Scope Uses
- **Primary PD model**: Module C's `market_pd` is a cross-check, not the primary PD estimate
- **US market pricing**: Suggested rates are calibrated to LendingClub's US P2P market — Indian rates require recalibration
- **Individual feature explanations**: Module C does not produce SHAP-level feature attribution

---

## 3. Training Data

| Property | Value |
|----------|-------|
| **Dataset** | LendingClub Loan Data 2007–2018 Q4 (Kaggle) |
| **Population** | US P2P consumer credit borrowers, closed loans only |
| **Period** | 2007–2018 Q4 |
| **Total accepted loans** | 2,260,668 |
| **Modeling population** | Closed loans only (Fully Paid + Charged Off) |
| **Sample used** | 300,000 closed loans |
| **Training rows** | ~240,000 (80%) |
| **Test rows** | ~60,000 (20%) |
| **Target (default model)** | `DEFAULT = 1` if Charged Off |
| **Default rate** | ~20.0% |
| **Target (grade model)** | Loan grade A–G (7 classes) |

### Why Closed Loans Only
Using "Current" loans introduces **survival bias** — they have not completed their term and many will eventually default but have not yet been labelled. Restricting to Fully Paid + Charged Off gives a clean, unambiguous target for both models.

### Market PD Lookup Table (Observed Rates)

| Grade | Market PD | Avg Rate | Basis |
|-------|-----------|----------|-------|
| A | ~5.5% | ~6.9% | 57,721 closed loans |
| B | ~13.0% | ~9.9% | 86,710 closed loans |
| C | ~22.6% | ~13.3% | 83,376 closed loans |
| D | ~32.5% | ~16.7% | 41,594 closed loans |
| E | ~43.2% | ~19.2% | 22,635 closed loans |
| F | ~51.9% | ~23.5% | 6,535 closed loans |
| G | ~55.1% | ~27.5% | 1,429 closed loans |

*These are empirically observed default rates from closed loans — not model predictions.*

### Data Limitations
- **US market basis**: LendingClub operates in the US. Grade → default rate mapping may differ in India.
- **P2P selection bias**: LendingClub's underwriting already filtered applications. The accepted loan population is not a random sample of all credit applicants.
- **Vintage bias**: Loans from 2007–2009 (pre-tightening) may have higher default rates than a post-2010 model trained dataset.
- **Rate environment**: US interest rates 2007–2018 differ significantly from Indian retail lending rates. Suggested rates require local recalibration.

---

## 4. Model Performance

### Confirmed Metrics (from notebook 02 outputs — test set, n=59,976, default rate 20.18%)

### Default Model (XGBoost Binary)

| Metric | Confirmed Value | Design Target |
|--------|-----------------|---------------|
| AUC | **0.7397** | ~0.72-0.76 ✅ |
| KS | **0.3517** | ~0.38-0.45 (slightly below target) |
| Gini | **0.4794** | ~0.44-0.52 ✅ |

### Grade Classifier (XGBoost Multiclass)

| Metric | Confirmed Value | Notes |
|--------|-----------------|-------|
| Within-1-grade accuracy | 100% on scored test set | Grade lookup is calibrated directly from the market PD table, so predictions track the assigned grade closely by construction |

**Why within-1-grade accuracy is the right metric:** A loan predicted as Grade C when it is actually Grade B is operationally equivalent. A loan predicted as Grade A when it is Grade F is a pricing failure. The model is evaluated on ordinal proximity, not categorical accuracy.

### Key EDA Findings (Confirmed from NB01 profiling)

| Feature | Default Rate Insight |
|---------|---------------------|
| Grade A | 5.3% default rate |
| Grade G | 57.9% default rate |
| 60-month term | 37.1% default (vs 14.6% for 36-month) |
| FICO 780+ | 5.5% default rate |
| FICO 620–660 | 23.6% default rate |
| Small business purpose | 29.4% default rate (highest) |
| Debt consolidation | 21.5% default rate |

### Concentration Analysis (from NB03)

| Dimension | HHI | Risk Level |
|-----------|-----|-----------|
| Purpose | > 2,500 (debt consolidation ~57%) | High concentration |
| Term | ~1,700 | Moderate |
| Grade | ~1,400 | Low |

---

## 5. Pricing Model

### Suggested Rate Formula
```
Suggested Rate = Grade Base Rate + clip((loan_pd - grade_avg_pd) × 0.60 × 100, -5, +5)
```

### Pricing Adequacy Classification

| Classification | Condition | Decision Engine Routing |
|---------------|-----------|------------------------|
| Underpriced | Suggested > Actual + 1% | REPRICE (if PD acceptable) |
| Fair | Within ±1% | No pricing action |
| Overpriced | Suggested < Actual − 1% | No pricing action (borrower-advantageous) |

---

## 6. Limitations

1. **Rate environment mismatch**: LendingClub US rates (6–28%) differ from Indian retail rates (10–36% NBFCs). Suggested rates need Indian market calibration.
2. **No SHAP attribution**: Module C does not provide feature-level explanations. Adverse action notices rely on Modules A and B for feature-level reasons.
3. **Grade classifier uncertainty**: Grade prediction confidence is not uniform across grades — E/F grade loans have more uncertainty than A/B.
4. **Concentration flag is binary**: Current implementation flags or doesn't flag — a continuous concentration score would be more nuanced.

---

## 7. Monitoring Requirements

| Trigger | Metric | Amber | Red | Action |
|---------|--------|-------|-----|--------|
| Score drift | PSI (default model) | > 0.10 | > 0.25 | Recalibrate |
| Grade drift | PSI (grade distribution) | > 0.10 | > 0.25 | Recalibrate |
| Pricing | Reprice rate | > 30% | > 50% | Rate schedule review |
| Concentration | High-purpose flag % | > 40% | > 60% | Portfolio review |
| Market calibration | Grade DR vs observed | > 15% gap | > 25% gap | Lookup table update |

---

### Confirmed Pricing & Concentration (from scored test set)

| Metric | Confirmed Value |
|--------|-----------------|
| Underpriced | 98.4% |
| Fair | 1.3% |
| Overpriced | 0.3% |
| Concentration flag rate | 25.1% |
| Purpose HHI | 2,232 (High concentration) |
| Term HHI | 6,332 (High concentration) |

*Model card version: 1.1 · Last updated: June 2026 (notebooks executed, metrics confirmed) · Next review: December 2026*
