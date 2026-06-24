# AI Credit Intelligence System - Master Blueprint

A full-lifecycle credit risk system engineered to beat the incumbent process on approval rate and default rate at the same time, then price, provision, and collect on that book more efficiently than the incumbent does.

Classification: Board Restricted - CRC-2026-09 - Version 2.0

---

## 1. The Governing Principle

The entire system is judged by one sentence:

> Approve more good borrowers and fewer bad ones than the incumbent at the same time, then price, provision, and collect on that book more efficiently.

Gini, KS, and AUC are enablers, not goals. A higher Gini is only useful because it shifts the approval-versus-default frontier outward. The deliverable a Chief Risk Officer respects is a superior business frontier, not a superior metric.

This corrects the previous narrative. The old headline was "RAROC gate approves fewer, higher quality loans." That reads as a retreat. The new headline is "our custom model dominates the incumbent's frontier," and the RAROC gate becomes a second-order dial that selects where on that superior frontier the committee chooses to sit.

---

## 2. The Success Criteria Framework (the spine)

Every decision point in the system (origination, pricing, collections) is evaluated against the same three criteria. This is the analytical spine that runs through every module.

| Criterion | Definition | What it proves |
|-----------|------------|----------------|
| SC1 | Same cost or population, better outcome | The model is sharper at the same operating point |
| SC2 | Same outcome, lower cost | The model achieves the target with less exposure or effort |
| SC3 (Gold) | Less cost AND better outcome, simultaneously | Genuine value creation - the only result that gets a model promoted |

How SC3 expresses itself in each domain:

- Origination: higher approval rate AND lower default rate than the incumbent.
- Pricing: more risk-adjusted revenue at the same or lower realised loss.
- Collections: fewer interventions AND higher cure or recovery rate.

Every module must state which criterion it targets and prove it with a swap-set analysis or an intervention table. No module claims a win on Gini alone.

---

## 3. Champion versus Challenger (corrected definition)

In real model risk management the Champion is the incumbent and the Challenger is the candidate. A Challenger is promoted only if it beats the Champion on the business frontier.

- Champion: the incumbent process the lender runs today. For this system the defensible incumbent is a bureau-score-only cutoff, which is exactly what a lender uses before it builds a custom model. This ties directly to the Bureau Custom Model thesis: a generic bureau score is the baseline a custom model must beat.
- Challenger: our custom model (PD model plus behavioural and portfolio signals).
- Promotion rule: the Challenger wins on every material axis - approval, default, risk-adjusted return, capital efficiency, and provisioning. That is the entire point of the project. A model that cannot beat the incumbent on all relevant parameters does not get deployed and does not get its author hired.

---

## 4. Lifecycle Architecture (the structure)

The system follows the credit lifecycle end to end. Each stage has a clear objective, the success criterion it proves, its inputs, and its outputs.

### Stage 0 - Data Foundation and Design
- Objective: build the analytical base before any model is trained.
- Contains: fact tables, roll-rate analysis, vintage analysis, data-point selection (the four-metrics strategy), sample selection (train, test, and out-of-time), feature universe construction (flag, aggregate, and momentum features), reject inferencing for rejected or undisbursed applicants where the outcome label is unavailable, and New-To-Credit treatment.
- Proves: a representative, leakage-free, regulator-defensible data foundation.
- Output: modelling samples and a documented feature universe.

### Stage 1 - Acquisition and Application Risk (Module A) - HEADLINE
- Objective: decide who to approve at the point of application.
- Contains: PD model (Challenger) benchmarked against a bureau-score-only Champion, WoE or IV scorecard, calibration, and the swap-set analysis.
- Proves: SC3 at origination - higher approval AND lower default than the incumbent.
- Output: calibrated PD per applicant, the dominating frontier, and the swap set.

### Stage 2 - Behavioural Risk (Module B)
- Objective: assess how an existing borrower is actually behaving.
- Contains: delinquency model, behavioural scorecard, hard-override stress flags, SHAP explainability.
- Proves: early detection of deterioration that feeds Stage classification and collections triggers.
- Output: behavioural risk score and delinquency probability.

### Stage 3 - Portfolio and Pricing Risk (Module C)
- Objective: confirm the rate is right and the book is not over-concentrated.
- Contains: grade classifier, market-implied PD, suggested-rate model, HHI concentration analysis, defensible risk-based pricing on the approved RAROC-positive book with modest spreads.
- Proves: pricing adequacy and concentration control.
- Output: suggested rate, market PD, concentration flags.

### Stage 4 - Capital, Expected Loss, and RAROC
- Objective: convert risk estimates into economic value.
- Contains: EL equals PD times LGD times EAD, Basel risk-weighted assets, economic capital, loan-level and portfolio RAROC, the policy efficient frontier, and operating-point selection.
- Proves: the chosen operating point clears the institutional hurdle and is capital efficient.
- Output: the frontier and the two operating points (recommended and net-income-maximising) as a risk-appetite dial on the improved frontier.

### Stage 5 - ECL and IFRS 9 (calculate)
- Objective: compute the provision the regulator requires.
- Contains: Stage 1, 2, and 3 classification, lifetime PD from the survival curve, vintage analysis, ECL provision.
- Proves: provisioning is accurate and conservative where required.
- Output: ECL by stage and by band, expressed as a rate (ECL divided by EAD), never as an absolute figure compared across portfolios of different sizes.

### Stage 6 - ECL Reduction through Collections and Recovery (reduce) - NEW
- Objective: move from calculating ECL to reducing it. This is what separates an elite system from a textbook one.
- Contains: Current-to-Bounce prediction, Bounce-to-NPA prediction, NPA-to-Recovery prediction, Collection Strategy Optimisation (the intervention table), and recovery-curve or LGD modelling.
- Proves: SC3 in collections - intervene on a smaller, better-targeted population AND achieve a higher cure or recovery rate, which lowers LGD and the realised default population, and therefore lowers the ECL provision on the same book.
- Output: targeted intervention strategy and a quantified, rate-based ECL reduction.

### Stage 7 - Decision Engine (Module D)
- Objective: combine all signals into one explainable decision.
- Contains: composite signal aggregator, hard-override checks, RAROC gate, reprice pathway, SHAP explanation, JSON audit log with model-version provenance, adverse-action notices.
- Proves: every decision is reproducible, explainable, and auditable.
- Output: Approve, Reprice, Review, or Decline with a full reason trail.

### Stage 8 - Adoption and MLOps
- Objective: get the model into production and keep it healthy.
- Contains: back testing, shadow testing, A/B testing, API building and deployment, the Ready Reckoner, explainability, calibration, model monitoring (PSI, CSI, and GREEN, AMBER, RED triggers), and versioning.
- Proves: the model is deployable and monitored, not a notebook artefact.
- Output: deployed scoring service and a live monitoring dashboard.

### Stage 9 - Governance and Model Risk Management
- Objective: satisfy the regulator and the board.
- Contains: model cards per module, SR 11-7 alignment, RBI MRM, Basel III ICAAP, IFRS 9, Fair Practices Code, the recalibration schedule, and quarterly regulatory forecasting.
- Proves: institutional-grade control.
- Output: the governance pack.

---

## 5. The Headline Result (Stage 1 swap-set)

The headline is a single table that a credit committee cannot argue with. It holds one axis constant and shows the Challenger winning on the other, then shows both moving favourably.

Metrics to report once the scored data is in:
- At the incumbent approval rate: the Challenger default rate (target: lower) - this is SC1.
- At the incumbent default rate: the Challenger approval rate (target: higher) - this is SC2.
- The simultaneous point: higher approval AND lower default - this is SC3, the gold standard.
- The swap set itself: the count of loans the Challenger declines that the incumbent approved (the bad swaps removed) and the loans the Challenger approves that the incumbent declined (the good swaps added).

All figures pending the rerun scored CSVs. The magnitude target is realistic and practically possible, not a drastic or theoretical maximum. A model improving from a bureau-score-only benchmark to a full custom model typically shifts the frontier by a modest but real margin, and that margin is the entire commercial case.

---

## 6. Bureau Custom Model Methodology

When a lender launches a new product there is no historical data, no application scorecard, and no default model. The solution is to build from bureau accounts that resemble the target book, selected on three signatures:

1. Product signature: product type, loan amount, tenure, interest rate, and default rate.
2. Customer signature: age, demographics, home ownership, and past credit behaviour.
3. Location signature: a product launched only in specific states uses accounts from those states; a pan-India product uses pan-India data.

A parallel New-To-Credit model is built for applicants with no bureau history. A weak NTC strategy alone can sink an otherwise sound product launch, so it is treated as a first-class component, not an afterthought.

---

## 7. ECL Reduction Engine (Stage 6 detail)

The real interview question is not "can you calculate ECL" but "can you reduce ECL." Reducing ECL raises profitability, lowers credit losses, improves capital efficiency, and improves the lender's valuation. The reduction is delivered by a sequence of models across the delinquency lifecycle:

- Current-to-Bounce: predict which performing accounts will miss the next payment, so intervention happens before the account rolls.
- Bounce-to-NPA: predict which bounced accounts will progress to non-performing, so effort concentrates where it changes the outcome.
- NPA-to-Recovery: predict recovery likelihood on non-performing accounts to prioritise collections.
- Collection Strategy Optimisation: the intervention table. The Champion intervenes broadly. The Challenger achieves SC3 - it intervenes on a smaller population and achieves a higher cure or recovery rate, which is less cost and more recovered value at once.
- Recovery-curve and LGD modelling: a lower LGD flows directly into a lower ECL, since ECL equals PD times LGD times EAD.

The ECL reduction is always expressed as a rate and on the same book, never as an absolute figure compared across portfolios of different sizes.

---

## 8. Design System (visual language)

The palette is black, white, purple, and gold, with restrained neutrals. The previous palette is retired. Every colour used must be readable; no near-invisible greys.

Tokens:
- Background: #0A0A0F
- Surface: #131119
- Card: #1B1825
- Border: #2C2838
- Purple (primary, structure and headers): #8B5CF6
- Purple deep (accents and rules): #6D28D9
- Purple bright (links and highlights): #A78BFA
- Gold (value, wins, positive results): #D4AF37
- Gold bright (emphasis): #E6C260
- White (headlines): #F5F4F8
- Body text (readable): #C9C7D4
- Secondary text (readable, not faint): #A29FB2
- Risk or breach only, used sparingly: #C45B5B

Rules:
- Purple denotes structure and the system. Gold denotes value, wins, and positive outcomes. White is for headlines. Readable grey is for secondary text.
- Never use a text colour below #A29FB2 in luminance for content that must be read.
- No double dashes anywhere. No em dashes. Use a plain hyphen or rewrite.
- Typography: a clean grotesk for display, a readable sans for body, a mono for figures and code.

---

## 9. Data Contract (what the rerun must produce)

For the swap-set and the downstream economics to be real, each module's scored output CSV (test set, ideally also out-of-time) should contain, per loan:

| Column | Purpose |
|--------|---------|
| loan_id | join key |
| actual_default | realised 12-month outcome, 1 or 0 - required for every comparison |
| pd_model | the Challenger model calibrated PD |
| bureau_score or ext_source | a single bureau-score feature, used to construct the incumbent Champion benchmark so we can prove the custom model beats a bureau-score-only cutoff |
| ead or amt_credit | exposure, for ECL and RAROC |
| segment fields (optional) | product type, state or location, age band - enables the three-signature and NTC cuts |

With these columns I can construct the incumbent benchmark, the swap set, the dominating frontier, and then layer ECL and RAROC on top, with every figure traceable to the data.

---

## 10. Build Sequence

Unblocked now (no dependency on the rerun):
- This blueprint.
- The shared design system and palette.
- The methodology and design narrative: Bureau Custom Model, three signatures, NTC, reject inferencing, the Design to Modelling to Adoption pipeline, and the ECL reduction engine design.
- The structural rebuild of the front-door document: corrected KPIs, proper structure, positive and elite Business Impact framing, and How-To-Run moved to the end or removed.

Waiting on the rerun scored CSVs:
- The swap-set actual numbers and the frontier-dominance proof.
- Refreshed Module A and Module B metrics.
- The quantified, rate-based ECL reduction.

Every number that lands will be defensible, real, and realistic. The target is a practically achievable improvement, not a drastic or theoretical one.
