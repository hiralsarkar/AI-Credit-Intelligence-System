# Governance — AI Credit Intelligence System
*Cross-module model risk management layer*

---

## Contents

| File / Folder | Description | Status |
|---------------|-------------|--------|
| `model_cards/module_a_model_card.md` | Module A model card — confirmed AUC=0.7551, KS=0.3786, stress test results | ✅ Complete |
| `model_cards/module_b_model_card.md` | Module B model card — behavioural delinquency model, SHAP, stress flags | ✅ Complete |
| `model_cards/module_c_model_card.md` | Module C model card — grade classifier, market PD, pricing model | ✅ Complete |
| `regulatory_alignment.md` | SR 11-7 / RBI MRM / Basel III / Fair Practices Code — full requirement mapping | ✅ Complete |
| `monitoring_triggers.py` | Runnable PSI, AUC, portfolio health, fairness checks with GREEN/AMBER/RED | ✅ Complete |

---

## Running the Monitoring System

```bash
cd 05_governance

# Run all checks
python monitoring_triggers.py

# With actual scored files
python monitoring_triggers.py \
    --scored_a ../01_module_a_application_risk/01_data/processed/scorecard_output.csv \
    --scored_b ../02_module_b_behavioural_risk/01_data/processed/scorecard_output_b.csv \
    --scored_c ../03_module_c_portfolio_pricing_risk/01_data/processed/scored_test_c.csv

# Export JSON report
python monitoring_triggers.py --export
```

---

## Monitoring Thresholds

| Metric | Amber | Red | Action |
|--------|-------|-----|--------|
| PSI (any model) | > 0.10 | > 0.25 | Recalibrate |
| AUC (Module A) | < 0.72 | < 0.70 | Recalibrate |
| KS (Module A) | < 0.32 | < 0.30 | Recalibrate |
| Portfolio RAROC | < 10% | < 0% | Strategy review |
| Default rate | > 10% | > 15% | Credit policy review |
| Approval rate income gap | > 20pp | > 35pp | Bias investigation |

---

## Regulatory Framework

| Standard | Coverage |
|----------|----------|
| SR 11-7 | Model development, validation, audit trail |
| RBI MRM Circular (2023) | Model inventory, documentation, independent validation |
| Basel III / RBI ICAAP | Capital adequacy, stress testing, EL = PD × LGD × EAD |
| RBI Fair Practices Code | Adverse action notices, income parity monitoring |

Full mapping: `regulatory_alignment.md`
