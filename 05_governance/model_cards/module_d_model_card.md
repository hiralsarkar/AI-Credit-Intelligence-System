# Model Card - Module D: IFRS 9 ECL & Staging

*AI Credit Intelligence System · Governance Documentation*

---

## 1. Model Overview

| Field | Detail |
|-------|--------|
| **Model name** | IFRS 9 Staging, Lifetime PD, LGD & ECL Engine (+ Collections / Uplift) |
| **Version** | v1.0 |
| **Staging + ECL** | Monthly stage allocation, transition matrix, lifetime PD, LGD, ECL by stage (`_pipeline_d.py`) |
| **Collections** | Roll-rate models C2B / B2N / cure + contact-prioritisation strategy (`_collections_d.py`) |
| **Behavioural bridge** | Origination-to-narrative delinquency model (`_b2n_behavioural_d.py`) |
| **Uplift / NBA** | Two-model T-learner, modification as treatment (`_uplift_nba_d.py`) - **exploratory** |
| **Developed by** | AI Credit Intelligence System - Module D |
| **Last trained** | July 2026 |
| **Next review due** | January 2027 |

This is the only module in the system with **SICR-based IFRS 9 staging**, because it is the only one built on a monthly performance panel rather than an origination snapshot. Any "staging" in Module A is a PD-bucket proxy; the credible staging lives here.

---

## 2. Intended Use

### Primary Use
Turn a loan-performance panel into a **provisioning decision**:
1. Allocate every loan-month to Stage 1 / 2 / 3 from delinquency status.
2. Estimate lifetime PD from the stage transition matrix and vintage curves.
3. Measure LGD from realised losses on disposed loans.
4. Compute **ECL by stage** (12-month for Stage 1, lifetime for Stage 2/3) and the provisioning uplift at the Stage 1 -> Stage 2 boundary.
5. Prioritise collections on the accounts where contact most reduces the roll to NPA.

### What Makes Module D Unique
Modules A-C score risk at a point in time. Module D models **how risk evolves over the life of the book** - cures, re-defaults, prepayments - which is exactly what IFRS 9 staging and lifetime ECL require. The ECL-reduction thesis follows directly: better origination keeps loans in Stage 1, and collections cure Stage 2 back before the lifetime-ECL cliff.

### Out-of-Scope Uses
- **Origination decisioning**: Module D is a provisioning and collections engine, not an approve/decline model.
- **Indian retail ECL**: calibrated on US single-family mortgages (Freddie Mac); Indian retail PD/LGD require recalibration.
- **Causal treatment effects**: the uplift model is observational (see Limitations) and must not be read as a proven intervention effect.

---

## 3. Training Data

| Property | Value |
|----------|-------|
| **Dataset** | Freddie Mac Single-Family Loan-Level (Sample), origination + monthly performance |
| **Vintages** | 2008, 2009, 2010, 2011 |
| **Loans** | 200,000 (~50,000 per vintage) |
| **Observations** | ~12.6M loan-months |
| **Target (staging)** | Stage from monthly delinquency status (0 current, 1-2 = 30-89 DPD, 3+ = 90+ DPD, RA = reperforming) |
| **Target (LGD)** | Realised ACTUAL LOSS on disposed loans (n = 3,301) |
| **Default definition** | Zero-balance codes {3, 9} (short-sale / charge-off, REO) |

### Data-Handling Notes (verified in pipeline)
- Freddie ACTUAL LOSS is stored **negative**; the pipeline uses `abs()`. EAD is the zero-balance-removal UPB (column 26).
- Delinquency parse is defensive (`str(s).strip()`, non-digit sentinels dropped); a guard asserts Stage 1 share > 0.80.
- Read directly from the annual sample zips; no intermediate extraction committed.

### Data Limitations
- **US mortgage basis**: secured single-family loans; loss and cure dynamics differ from unsecured Indian retail.
- **Crisis-weighted vintages**: 2008-2011 spans the housing downturn, so absolute default levels are elevated versus a benign vintage.
- **Sample, not universe**: the Freddie sample is a stratified subset of the full loan universe.

---

## 4. Model Performance

### IFRS 9 Stage Distribution (loan-month basis)

| Stage | Share | Mean lifetime PD | Lifetime ECL rate |
|-------|-------|------------------|-------------------|
| Stage 1 | 97.6% | 1.54% | 0.06% (12-month) |
| Stage 2 | 1.4% | 5.87% | 2.45% |
| Stage 3 | 1.0% | 9.73% | 4.07% |

### Monthly Stage Transition Matrix

| From \ To | Stage 1 | Stage 2 | Stage 3 | Default | Prepaid |
|-----------|---------|---------|---------|---------|---------|
| **Stage 1** | 98.0% | 0.52% | 0.0% | 0.0% | 1.48% |
| **Stage 2** | **28.5% (cure)** | 61.5% | 7.96% | 1.04% | 0.95% |
| **Stage 3** | 5.08% | 3.58% | 89.9% | 0.70% | 0.76% |

The **28.5% monthly Stage 2 -> Stage 1 cure rate** is the lever behind the collections case.

### LGD, ECL and the Provisioning Cliff

| Metric | Value |
|--------|-------|
| LGD (realised, n=3,301) | **41.8%** |
| Stage 1 12-month ECL rate | 0.06% |
| Stage 2 lifetime ECL rate | 2.45% |
| **Stage 1 -> Stage 2 lifetime-ECL uplift** | **~39x** |
| 24-month SICR incidence | 5.99% |

### SICR Drivers (Stage 1 -> Stage 2 migration)

| Driver | Signed weight | Reading |
|--------|---------------|---------|
| FICO | -1.44 | dominant protective factor |
| Note rate | +0.52 | higher rate, higher SICR risk |
| CLTV | +0.10 | |
| DTI | +0.09 | |
| Original LTV | +0.07 | |
| Term | +0.06 | |

### Vintage Default Curves
2008 vintage reaches ~6.0% cumulative default; 2011 vintage ~1.1% - the clearest evidence that origination quality, not just servicing, drives lifetime loss.

### Collections & Behavioural Models

| Model | AUC | Note |
|-------|-----|------|
| Behavioural delinquency bridge (B2N) | **0.946** | vs 0.555 origination-only; DPD bucket dominates |
| Cure to Stage 1 (C2B) | 0.749 | |
| Roll to Stage 3 | 0.618 | |

Contact strategy: prioritising the **top 30%** of at-risk accounts avoids ~**11.6%** of the NPA roll, **under an assumed intervention lift of 0.3** (an assumption, not a measured effect).

---

## 5. ECL Methodology

```
Stage 1  -> 12-month ECL  = PD_12m  × LGD × EAD
Stage 2  -> lifetime ECL  = PD_life × LGD × EAD   (SICR triggered)
Stage 3  -> lifetime ECL  = PD_life × LGD × EAD   (credit-impaired)
```

The ~39x jump from the Stage 1 12-month ECL rate (0.06%) to the Stage 2 lifetime rate (2.45%) is the provisioning cliff: preventing a single SICR migration is far cheaper than curing it after the fact, which is the quantitative basis for both better origination and earlier collections.

---

## 6. Limitations

1. **Uplift model is observational, not causal.** The T-learner uses loan modification as an *observed* treatment (n=20,615 delinquent loans, 12.6% modified). Naive ATE is +4.2pp avoided default, but treatment is not randomly assigned, so the effect is **confounded by selection into modification**, and the uplift-decile validation is **not monotonic** (signal concentrates in the top deciles). Use it as a targeting hypothesis only; production needs a randomised champion/challenger holdout or propensity adjustment. Do not quote the ATE as a treatment effect.
2. **Assumed intervention lift (0.3).** The collections NPA-roll-avoided figures scale with an assumed contact effectiveness, stated explicitly rather than measured.
3. **US secured basis.** LGD 41.8% and cure dynamics are mortgage-specific; unsecured retail LGD is materially higher and cures rarer.
4. **Crisis vintages.** 2008-2011 inflate absolute PD/loss versus a normal cycle; the *relative* stage and vintage structure is the transferable insight, not the absolute levels.

---

## 7. Monitoring Requirements

| Trigger | Metric | Amber | Red | Action |
|---------|--------|-------|-----|--------|
| Staging drift | PSI (stage distribution) | > 0.10 | > 0.25 | Re-estimate transition matrix |
| Cure rate | Stage 2 -> Stage 1 monthly | < 25% | < 20% | Collections review |
| LGD | Realised vs 41.8% | > 10% gap | > 20% gap | Re-measure on recent dispositions |
| ECL coverage | Provision / gross book | drift vs plan | material breach | Provisioning committee |
| SICR incidence | 24-month rate vs 5.99% | > 15% gap | > 25% gap | Recalibrate SICR model |

---

*Model card version: 1.0 · July 2026 (pipelines executed, metrics confirmed from `04_outputs/module_d_*.json`) · Next review: January 2027*
