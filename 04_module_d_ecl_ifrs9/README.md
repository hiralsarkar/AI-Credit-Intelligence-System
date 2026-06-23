# Module D - IFRS 9 ECL & Staging

Expected Credit Loss (ECL) and IFRS 9 staging on a loan-performance panel. This module
moves the project from *calculating* ECL to *reducing* it, by modelling how accounts
migrate between stages over their life.

## Why this module exists

ECL under IFRS 9 is a staging concept and staging needs time:

- **Stage 1** - performing, no significant increase in credit risk -> 12-month ECL
- **Stage 2** - significant increase in credit risk (SICR) -> lifetime ECL
- **Stage 3** - credit-impaired (90+ days past due / default) -> lifetime ECL

An origination snapshot cannot show this. It requires a panel with delinquency status
tracked month by month, which is what this module uses.

## Data

Freddie Mac Single-Family Loan-Level Dataset (Sample), origination + monthly performance,
vintages 2008-2011. The raw files are downloaded separately and are not committed
(see `.gitignore`). Place the annual sample files under `01_data/raw/`.

## Planned outputs

- IFRS 9 stage allocation from monthly delinquency status
- Stage transition matrix (monthly migration, including cures and defaults)
- Lifetime PD term structure and vintage default curves
- LGD from realised losses on disposed loans
- ECL by stage (12-month vs lifetime) and the provisioning uplift
- SICR driver analysis (what pushes an account from Stage 1 to Stage 2)

## Structure

```
01_data/        raw (gitignored) and processed data
02_notebooks/   analysis notebooks
03_models/      saved models
04_outputs/     metrics, tables, charts
```
