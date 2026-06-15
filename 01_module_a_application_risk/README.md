# Module A — Application Risk

**Dataset:** Home Credit Default Risk  
**Signal output:** `pd_score`, `risk_band`, `credit_score`, `el`, `raroc`, `value_decision`

---

## What This Module Does

Module A estimates the **Probability of Default at the point of loan application** — the risk signal generated from information a borrower provides when they apply. It is the primary credit risk signal in the decisioning architecture.

This module goes beyond PD prediction to build a complete financial risk stack:
- PD estimation (Logistic Regression + XGBoost)
- Credit scorecard (WoE/IV, 300–900 points scale)
- Expected Loss, RWA, Economic Capital (Basel III)
- RAROC per borrower and risk band
- Strategy simulation (4 lending strategies)
- Stress testing (4 macro scenarios, break-even analysis)

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_preprocessing` | Data pipeline — missingness, imputation, encoding, train-test split |
| `02_credit_risk_model` | PD model — Logistic Regression + XGBoost, AUC/KS/Gini evaluation |
| `03_scorecard_analysis` | WoE scorecard — IV feature selection, points table, score-to-PD mapping |
| `04_expected_loss_capital_model` | EL, RWA, Economic Capital, RAROC per borrower |
| `05_strategy_simulator` | 4 lending strategies compared on risk-return frontier |
| `06_stress_testing` | 4 macro scenarios, RAROC matrix, break-even multipliers, monitoring triggers |

## Signal Output Schema

```python
{
  "pd_score":       float,   # 0.0 – 1.0, XGBoost champion model
  "risk_band":      int,     # 1 (Very Low) – 5 (Very High)
  "credit_score":   int,     # 300 – 900, WoE scorecard
  "risk_grade":     str,     # "Very Low Risk" to "Very High Risk"
  "el":             float,   # Expected Loss (₹)
  "el_rate":        float,   # EL as % of EAD
  "econ_capital":   float,   # Economic Capital (₹)
  "raroc":          float,   # Risk-Adjusted Return on Capital
  "value_decision": str      # "Value Accretive" / "Marginal" / "Value Destructive"
}
```

## Dataset

**Home Credit Default Risk** — consumer loan applicants, application-time features, bureau scores.  
Download from: https://www.kaggle.com/c/home-credit-default-risk  
Place `application_train.csv` in `data/raw/`

## Status

✅ Complete
