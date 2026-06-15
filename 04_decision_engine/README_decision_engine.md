# Decision Engine — Module D
*AI Credit Intelligence System · Integration Layer*

**Inputs:** Signal outputs from Module A, B, and C  
**Output:** Unified lending decision with full explainability and audit trail

---

## What This Does

The Decision Engine is the layer that makes this project a **system** rather than a
collection of models. It ingests the three risk signals produced by each module and
produces a single, explainable lending decision.

```
Module A Signal          Module B Signal          Module C Signal
(Application Risk)  +   (Behavioural Risk)   +   (Portfolio Risk)
      ↓                        ↓                        ↓
┌──────────────────────────────────────────────────────────────┐
│                    Signal Aggregator                         │
│   Weighted composite PD + market PD divergence check        │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    Decision Engine                           │
│   5-layer rules: overrides → PD limits → pricing →          │
│   RAROC gate → default routing                              │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    Explainability Layer                      │
│   Signal attribution + SHAP reason codes +                  │
│   plain-English adverse action notice                       │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    Audit Logger                              │
│   Full JSON + CSV decision record (SR 11-7 compliant)       │
└──────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Description |
|------|-------------|
| `constants.py` | Single source of truth — all thresholds, weights, regulatory parameters |
| `01_engine.py` | Core decision logic — 5-layer rules, produces APPROVE/DECLINE/REPRICE/MANUAL REVIEW |
| `02_signal_aggregator.py` | Loads and combines Module A/B/C signals, computes composite PD |
| `03_explainability.py` | Signal attribution, SHAP reason codes, adverse action narrative generator |
| `04_audit_logger.py` | Logs every decision as JSON + CSV with full model provenance |
| `05_demo.py` | Runnable demo — 4 preset applicant profiles + custom input + batch mode |

---

## Decision Logic

```
Layer 1: Hard Overrides (Module B stress flags)
  IF high_delinquency_score OR critical_utilisation OR escalating_delinquency
    → MANUAL REVIEW (regardless of PD or RAROC)

Layer 2: Composite PD hard limits
  IF composite_pd >= 35%  → DECLINE
  IF composite_pd < 10% AND RAROC > 14%  → APPROVE (tentative — check layers below)

Layer 3: Module C routing (pricing and concentration)
  IF pricing_adequacy == "Underpriced" AND pd < review band  → REPRICE
  IF concentration_flag AND market_pd > 25%  → MANUAL REVIEW

Layer 4: RAROC gate
  IF RAROC < 14% AND pd not in auto-approve zone  → DECLINE

Layer 5: Default routing
  Remaining cases → MANUAL REVIEW
```

### Composite PD Formula

```
Composite PD = 0.60 × Module_A_PD + 0.40 × Module_B_PD

Weights are configurable in constants.py.
Module C does not enter the composite PD — it modifies the routing path.
```

---

## Quick Start

```bash
cd 04_decision_engine

# Run all 4 preset applicant profiles
python 05_demo.py

# Run a single preset profile
python 05_demo.py --profile ANAND_MEHTA

# Custom applicant
python 05_demo.py \
    --pd_a 0.08 --credit_score 642 --ead 250000 \
    --pd_b 0.05 --brs 72 \
    --market_pd 0.12 --grade B --rate 12.5 --pricing Fair

# Batch mode (process a scored CSV from Module A)
python 05_demo.py --batch ../01_module_a_application_risk/01_data/processed/scorecard_output.csv

# Show audit log summary
python 05_demo.py --summary
```

---

## Preset Applicant Profiles

| Profile | Description | Expected Decision |
|---------|-------------|-------------------|
| `ANAND_MEHTA` | Salaried professional, clean history, FICO 724 | APPROVE |
| `PRIYA_SHARMA` | Self-employed, loan underpriced for risk level | REPRICE |
| `VIKRAM_NAIR` | Escalating delinquency, critical utilisation | MANUAL REVIEW |
| `RAHUL_VERMA` | Repeated delinquencies, PD 41%, value-destructive | DECLINE |

---

## Output Schema

Every decision produces:

```python
{
  # Core decision
  "decision":              str,   # "APPROVE" / "DECLINE" / "REPRICE" / "MANUAL REVIEW"
  "primary_reason":        str,   # Most important reason in one sentence
  "composite_pd":          float, # Weighted PD from Modules A + B
  "composite_risk_band":   int,   # 1-5

  # Capital impact
  "capital_impact": {
    "ead":              float,  # Exposure at Default (Rs)
    "el":               float,  # Expected Loss (Rs)
    "el_rate":          float,  # EL as % of EAD
    "economic_capital": float,  # Capital consumed (Rs)
    "raroc":            float,  # Risk-Adjusted Return on Capital
    "value_decision":   str,    # "Value Accretive" / "Marginal" / "Value Destructive"
  },

  # Pricing
  "pricing": {
    "suggested_rate":   float,  # Risk-appropriate rate (%) — populated for REPRICE
    "pricing_adequacy": str,    # "Underpriced" / "Fair" / "Overpriced"
    "loan_grade":       str,    # Market grade equivalent (A-G)
    "market_pd":        float,  # Module C market-implied PD
  },

  # Signal summary (one-line per module)
  "signal_summary": {
    "module_a": str,
    "module_b": str,
    "module_c": str,
  },

  # Flags and overrides
  "overrides_triggered":  list,  # Hard override flags that fired
  "market_pd_note":       str,   # Note if market PD diverges from composite

  # Full explanation
  "reason_codes":      list,   # Ordered adverse action reason codes
  "decision_basis":    str,    # Full plain-English narrative
}
```

---

## Regulatory Framework

| Requirement | Implementation |
|-------------|----------------|
| SR 11-7 — Model documentation | `constants.py` documents all parameters with regulatory basis |
| SR 11-7 — Audit trail | `audit_logger.py` logs every decision as immutable JSON record |
| RBI Fair Practices Code | Adverse action reason codes in plain English, top 3 per decision |
| ECOA/FCRA | Same reason code structure; `regulatory_summary` field in each record |
| Basel III — RAROC | RAROC computed using RBI CET1 + capital conservation buffer (10.5%) |
| Basel III — RWA | Risk weights by band: Band 1-2 = 75%, Band 3 = 100%, Band 4-5 = 150% |

---

## Status

✅ **Complete** — all 5 files implemented and documented.

**Next:** `05_governance/` — PSI monitoring, model cards, bias checks, regulatory alignment map.
