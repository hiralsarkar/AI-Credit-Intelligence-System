# Module C — Portfolio & Pricing Risk

**Dataset:** LendingClub Loan Data  
**Signal output:** `market_pd`, `suggested_rate`, `loan_grade_signal`, `concentration_flag`

---

## What This Module Does

Modules A and B assess individual borrower risk. Module C adds a third signal: **market-calibrated pricing and portfolio-level risk**.

LendingClub data contains actual loan grades, interest rates charged, and observed default outcomes — meaning it encodes the market's revealed pricing of credit risk. This module answers two questions that neither Module A nor B can:

1. **Is the proposed interest rate appropriate for this risk level?** (pricing signal)
2. **Does adding this loan increase dangerous concentration in the portfolio?** (portfolio signal)

Together these make the decision engine **portfolio-aware** — it doesn't just evaluate each loan in isolation but considers what approving it does to the overall risk-return profile of the book.

## Notebooks

| Notebook | Description | Key Outputs |
|----------|-------------|-------------|
| `01_data_preprocessing_c` | Loan grade encoding, rate bands, vintage features, 80/20 split on closed loans | `clean_lendingclub.csv`, train/test splits |
| `02_loan_grade_model_c` | XGBoost grade classifier + XGBoost default model, market PD lookup table | `scored_test_c.csv`, `xgb_grade_classifier.pkl`, `xgb_default_c.pkl`, `grade_pd_lookup.pkl` |
| `03_portfolio_analysis_c` | Concentration risk (HHI by grade/term/purpose), portfolio summary | `hhi_summary.csv`, `portfolio_summary_c.csv` |

## Signal Output Schema

```python
{
  "market_pd":           float,  # Market-implied PD from loan grade model
  "suggested_rate":      float,  # Risk-based interest rate recommendation
  "loan_grade_signal":   str,    # "A" to "G" — market grade equivalent
  "concentration_flag":  bool,   # True if loan increases portfolio concentration risk
  "pricing_adequacy":    str,    # "Underpriced" / "Fair" / "Overpriced"
  "shap_top_drivers":    list    # Top 3 features driving the pricing signal
}
```

## Dataset

**LendingClub Loan Data** — loan grade, interest rate, purpose, term, funded amount, actual default outcome.  
Download from: https://www.kaggle.com/datasets/wordsforthewise/lending-club  
Place `accepted_2007_to_2018Q4.csv` in `data/raw/`

## Status

✅ **Complete** — all 3 notebooks executed. Default model AUC=0.7397, Gini=0.4794, KS=0.3517 on a 59,976-loan held-out test set (20.18% default rate). See `05_governance/model_cards/module_c_model_card.md` for confirmed metrics.

**Next:** `04_decision_engine/` — combines Modules A, B and C into a unified lending decision.
