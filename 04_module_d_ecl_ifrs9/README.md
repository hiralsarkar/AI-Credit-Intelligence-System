# Module D - IFRS 9 ECL & Staging

Expected Credit Loss (ECL) and IFRS 9 staging on a loan-performance panel. This module
moves the project from *calculating* ECL to *reducing* it, by modelling how accounts
migrate between stages over their life. It is the only module in the project with
credible, SICR-based IFRS 9 staging, because it is the only one built on time-series
(monthly) performance data rather than an origination snapshot.

## Why this module exists

ECL under IFRS 9 is a staging concept and staging needs time:

- **Stage 1** - performing, no significant increase in credit risk -> 12-month ECL
- **Stage 2** - significant increase in credit risk (SICR) -> lifetime ECL
- **Stage 3** - credit-impaired (90+ days past due / default) -> lifetime ECL

An origination snapshot cannot show this (a single point in time has no notion of an
*increase* in risk). It requires a panel with delinquency status tracked month by month,
which is what this module uses. Any "staging" in Module A is a PD-bucket proxy only;
the real staging lives here.

## Data

Freddie Mac Single-Family Loan-Level Dataset (Sample), origination + monthly performance,
vintages 2008-2011 (200,000 loans, ~12.6M loan-months). The raw files are downloaded
separately and are not committed (see `.gitignore`). Place the annual sample files under
`01_data/raw/`, then run `_pipeline_d.py`.

## Outputs (produced by the pipelines)

Run `_pipeline_d.py` (staging + ECL), `_collections_d.py` (roll rates, contact strategy),
`_b2n_behavioural_d.py` (behavioural bridge), `_uplift_nba_d.py` (targeting model),
`_surgical_strike_d.py` (fused priority score + ECL impact, run last - it reads
`module_d_ecl.json` for LGD).

- **IFRS 9 stage allocation** from monthly delinquency status: Stage 1 97.6% / Stage 2 1.4% / Stage 3 1.0%.
- **Stage transition matrix** (monthly migration): Stage 2 -> Stage 1 cure rate **28.5%**, Stage 2 -> Stage 3 impairment 8.0%.
- **Lifetime PD term structure and vintage curves**: 2008 vintage ~6.0% cumulative default vs 2011 ~1.1%.
- **LGD from realised losses** on disposed loans: **41.8%** (n = 3,301 dispositions).
- **ECL by stage**: a Stage 1 -> Stage 2 lifetime-ECL provisioning cliff of **~39x**.
- **SICR driver analysis**: FICO is the dominant protective driver of Stage 1 -> Stage 2 migration, then note rate, CLTV, DTI.

### Collections and next-best-action

- `_collections_d.py`: roll rates and a contact-prioritisation strategy (contact the top 30% of at-risk accounts to avoid ~11.6% of the NPA roll, under an assumed intervention lift of 0.3, which is stated as an assumption, not a measured effect).
- `_uplift_nba_d.py`: **exploratory only.** A two-model T-learner estimates the uplift of loan modification on avoiding default, over the ever-delinquent population (n = 20,615, 12.6% treated). Naive ATE is +4.2pp, but modification is an **observed, not randomised, treatment, so the effect is confounded**, and the uplift-decile validation is not yet monotonic (signal appears only in the top deciles). Treat this as a targeting hypothesis, not a causal claim. Do not quote the ATE as a treatment effect. Full write-up: [`06_docs/uplift_nba_analysis.html`](../06_docs/uplift_nba_analysis.html).

### Surgical-strike: fused priority targeting

`_collections_d.py` and `_uplift_nba_d.py` never talked to each other: one ranks accounts by
roll risk and applies an assumed flat lift to whoever is contacted; the other estimates who is
actually persuadable but never turns that into a contact decision. `_surgical_strike_d.py`
fuses them and prices the result:

- Re-fits a combined behavioural B2N model (AUC 0.945, matching `_b2n_behavioural_d.py`) and the
  T-learner uplift model, both scored on the **live, currently-open Stage-2 book**: 589 loans
  whose performance record simply ends mid-delinquency (right-censored), not the already-resolved
  loans the two source models trained on. Live-book features (running max DPD, running months
  delinquent) are point-in-time, so scoring never looks ahead.
- Priority score = B2N risk x the uplift model's own **measured** per-decile avoided-default gap
  (not the flat 0.3 assumption), so accounts get ranked by risk **and** persuadability together.
- At the same fixed contact capacity, pricing every strategy as EAD x LGD x benefit isolates two
  effects: **better targeting** (same flat-lift assumption, but choosing who to contact by
  persuadability instead of risk alone) and **honest pricing** (same risk-only queue, but crediting
  each account only the gap actually measured in its decile). At 20% contact capacity on the live
  book, better targeting alone raises avoided loss to **1.8x the status-quo level (+77%)**;
  combining it with honest pricing raises it to **2.7x (+171%)**, a gain equal to **65% of the
  live sub-book's own baseline lifetime ECL** (the 589-loan sub-book, not the full portfolio -
  see caveats below).
- The gain **plateaus past ~20% capacity**: contacting more than that mostly adds accounts with
  zero measured persuadability, and 21.4% of the live book is exactly the share carrying a
  measurable positive gap. That the plateau lines up with the measured persuadable share, rather
  than growing indefinitely with capacity the way the flat-lift assumption implies, is itself a
  sanity check that the fused score is doing something real rather than fitting noise. Full
  numbers: [`04_outputs/module_d_surgical_strike.json`](04_outputs/module_d_surgical_strike.json).

**Caveat, inherited from `_uplift_nba_d.py` and not resolved by fusing:** persuadability is still
an observational, confounded estimate. This is a priced *comparison of targeting strategies* under
that estimate, not a validated causal ECL forecast. Before the 1.8x/2.7x/65% figures inform capital
or collections-budget decisions, they need a randomised champion/challenger holdout on a live
contact queue.

## Structure

```
01_data/        raw (gitignored) and processed data
02_notebooks/   analysis notebooks
03_models/      saved models
04_outputs/     metrics, tables, charts
```
