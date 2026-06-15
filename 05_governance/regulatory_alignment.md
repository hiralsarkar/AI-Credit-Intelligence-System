# Regulatory Alignment Map
*AI Credit Intelligence System · Governance Documentation*

---

## Overview

This document maps every component of the AI Credit Intelligence System to the
specific regulatory requirements it satisfies. It is structured as a checklist
that a model validator, internal auditor, or regulator can use to verify compliance.

---

## 1. SR 11-7 — Supervisory Guidance on Model Risk Management (Federal Reserve)

SR 11-7 establishes the gold standard for model risk management globally. While
it is a US Federal Reserve guideline, RBI's 2023 MRM Circular draws heavily from it.
Both require the same three pillars: **Development**, **Validation**, **Governance**.

### Pillar 1 — Model Development

| SR 11-7 Requirement | Implementation | Location |
|--------------------|----------------|----------|
| Clearly defined purpose and intended use | Section 2 of each model card | `05_governance/model_cards/` |
| Conceptual soundness — theory behind model | Business context sections in all notebooks | Each module NB01 |
| Data quality assessment | Data profiling cells with missingness, outlier, sentinel analysis | Module A NB01, B NB01, C NB01 |
| Feature selection with documented rationale | WoE/IV analysis with explicit IV thresholds (< 0.02 = drop) | Module A NB03, B NB03 |
| Appropriate modelling technique | LR (interpretable baseline) + XGBoost (champion) with explicit rationale | Module A NB02, B NB02, C NB02 |
| Performance testing on held-out data | 80/20 stratified split; AUC, KS, Gini on test set | All module NB02s |
| Sensitivity analysis | Stress testing across 4 scenarios | Module A NB06 |
| Champion/challenger framework | LR challenger + XGBoost champion, performance comparison table | Module A NB02, B NB02 |
| Limitations documented | Section 7 of each model card | `05_governance/model_cards/` |

### Pillar 2 — Model Validation

| SR 11-7 Requirement | Implementation | Location |
|--------------------|----------------|----------|
| Independent validation (conceptual soundness) | Coefficient direction checks — sign of each feature vs domain expectation | Module A NB02 (cell 4), B NB02 (cell 4) |
| Benchmarking | Module A vs Module B vs Module C cross-model PD comparison | Decision Engine — `market_pd_divergence` |
| Outcome analysis | PSI computed on train vs test score distributions | Module A NB02 (PSI=0.0001), B NB02 |
| Ongoing monitoring triggers | Amber/Red thresholds with automated checks | `05_governance/monitoring_triggers.py` |
| Model card — formal documentation | Complete model card per module | `05_governance/model_cards/` |

### Pillar 3 — Governance

| SR 11-7 Requirement | Implementation | Location |
|--------------------|----------------|----------|
| Audit trail — every decision logged | Full JSON + CSV with inputs, outputs, model versions | `04_decision_engine/04_audit_logger.py` |
| Decision explainability | SHAP attribution + signal-level % contribution + reason codes | `04_decision_engine/03_explainability.py` |
| Version control | Model version registry in audit log | `04_audit_logger.py` → `MODEL_VERSIONS` dict |
| Review schedule | Quarterly review, 6-month recalibration cycle documented | Model cards — Section 8 |
| Approval chain | Documented in model cards (pending sign-off) | `05_governance/model_cards/` |

---

## 2. RBI Model Risk Management Circular (2023)

RBI's MRM Circular for regulated entities mirrors SR 11-7 with India-specific additions.

| RBI MRM Requirement | Implementation | Status |
|--------------------|----------------|--------|
| Model inventory | 5 models across 3 modules — all documented | `05_governance/model_cards/` |
| Model risk categorisation | All models are Category B (significant business impact) | Noted in model cards |
| Conceptual soundness review | Feature direction checks, IV validation, business logic verification | All NB02s |
| Data governance | Source, vintage, representativeness documented per dataset | Model cards Section 3 |
| Performance metrics | AUC, KS, Gini reported — all above minimum thresholds (Module A confirmed) | Model cards Section 4 |
| Stress testing | 4 macro scenarios calibrated to Indian events (IL&FS, COVID-19) | Module A NB06 |
| Fair Practices Code | Adverse action notices with top-3 reason codes in plain English | `03_explainability.py` → `generate_adverse_action_reasons()` |
| Independent validation | Required before production deployment | Pending — noted in model cards |
| Model change policy | Version registry; recalibration triggers documented | `monitoring_triggers.py` |
| Board-level reporting | Portfolio RAROC, EL, strategy comparison available | Module A NB04, NB05 |

---

## 3. Basel III — Capital Adequacy (BIS / RBI)

| Basel III Requirement | Implementation | Parameter Value |
|----------------------|----------------|-----------------|
| Risk weights — standardised approach | Risk weight by band: 75% (Bands 1–2), 100% (Band 3), 150% (Bands 4–5) | `constants.py` → `RISK_WEIGHTS` |
| CET1 capital ratio | RBI minimum 4.5% + conservation buffer 2.5% + buffer = 10.5% | `constants.py` → `CAPITAL_RATIO = 0.105` |
| Expected Loss = PD × LGD × EAD | Implemented exactly | Module A NB04, Decision Engine |
| LGD — unsecured retail | 45% (Basel II IRB floor) | `constants.py` → `LGD_UNSECURED = 0.45` |
| ICAAP — Internal Capital Adequacy | Stress test confirms capital impact under 4 macro scenarios | Module A NB06 |
| Economic capital calculation | RWA × capital ratio per loan | Module A NB04 |
| RAROC hurdle rate | 14% (risk-adjusted return threshold) | `constants.py` → `HURDLE_RATE = 0.14` |

### Capital Adequacy — Stress Test Summary (Module A NB06)

| Scenario | PD Multiplier | LGD Add | Conservative Strategy RAROC | Capital Status |
|----------|--------------|---------|----------------------------|----------------|
| Baseline | 1.0× | +0% | +72.66% | Adequate |
| Mild Stress | 1.5× | +5% | +44.91% | Adequate |
| Severe Stress | 2.5× | +15% | -24.46% | Breached |
| Extreme Stress | 4.0× | +25% | -144.71% | Severely Breached |

*Conservative strategy maintains adequacy through Mild Stress. Severe/Extreme stress represent systemic events (COVID-19, IL&FS equivalent) where capital support from the institution is required — this is the expected regulatory outcome.*

---

## 4. RBI Fair Practices Code — Adverse Action Disclosure

| Requirement | Implementation |
|-------------|----------------|
| Borrower must be informed of reason for rejection | `generate_adverse_action_reasons()` produces top 3 plain-English reasons |
| Reasons must be specific, not generic | Each reason codes to a specific feature/metric (e.g. "History of 90+ day past-due payment") |
| Complaint redressal mechanism | Decision ID logged; full record retrievable via `get_decision_by_id()` |
| Non-discrimination | No demographic features used in any model |

---

## 5. Model Risk Tier Classification

Under RBI MRM Circular, models are classified by risk tier:

| Model | Risk Tier | Basis |
|-------|-----------|-------|
| Module A — XGBoost PD | **Tier 2 (High)** | Primary origination decision; material credit impact |
| Module A — WoE Scorecard | **Tier 2 (High)** | Adverse action notices; regulatory disclosure |
| Module B — XGBoost Delinquency | **Tier 2 (High)** | 40% weight in composite PD; hard override flags |
| Module C — Grade Classifier | **Tier 3 (Medium)** | Cross-check / pricing signal, not primary PD |
| Decision Engine | **Tier 1 (Critical)** | Integrates all signals; final lending decision |

Tier 1 models require: independent validation, board-level approval, quarterly monitoring, documented recalibration process.

---

## 6. Compliance Checklist — Production Readiness

| Item | Status | Notes |
|------|--------|-------|
| Model documentation complete | ✅ | All model cards drafted |
| Regulatory alignment documented | ✅ | This document |
| Performance above minimum thresholds | ✅ Module A confirmed | B/C pending execution |
| Stress testing complete | ✅ | Module A NB06 — 4 scenarios |
| Adverse action notices implemented | ✅ | `03_explainability.py` |
| Audit trail implemented | ✅ | `04_audit_logger.py` — JSON + CSV |
| No demographic features in models | ✅ | Verified across all 3 modules |
| Independent validation | 🔲 Pending | Required before production |
| Board/Risk Committee approval | 🔲 Pending | Required before production |
| Production monitoring deployed | 🔲 Planned | `monitoring_triggers.py` — prototype ready |
| Indian market recalibration | 🔲 Planned | Modules B and C trained on US data |

---

*Regulatory alignment map version: 1.0 · Last updated: March 2026*
*Author: AI Credit Intelligence System — Governance Layer*
