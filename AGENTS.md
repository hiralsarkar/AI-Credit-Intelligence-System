# AGENTS.md - Source of Truth

Working spec for the AI Credit Intelligence System. Read before editing anything.

## What this is
A CRO-defensible, end-to-end retail-credit-risk system. Every number is real (derived from
LendingClub + Freddie Mac + Home Credit data) and must survive a credit-committee challenge.
Portfolio piece for senior data-science / credit-risk roles.

## Non-negotiable constraints
- No double hyphens or em dashes (U+2014) / en dashes (U+2013) anywhere (docs, code comments, notebooks). Plain hyphen or rewrite. CI enforces the dash rule via a test.
- Every figure must be defensible and reproducible from a pipeline. No fabricated / placeholder numbers.
- ECL expressed as a rate, never a headline absolute.
- Palette: black/white/purple/gold. `--bg:#0A0A0F --purple:#8B5CF6 --gold:#D4AF37 --white:#F5F4F8`
  `--text:#C9C7D4 --muted:#A29FB2 --risk:#C45B5B --good:#5EEAD4`. Fonts: Space Grotesk / Inter / IBM Plex Mono.
- Data paths via `os.environ` (FREDDIE_RAW_DIR etc.), not hard-coded absolute paths.

## Governing thesis
Custom Challenger model beats the incumbent bureau-score Champion on ALL grounds (Success Criteria 3:
higher approval AND lower default), yielding more profit and lower ECL - realistic, not drastic.
Approval cutoff is a constrained optimisation: max net income s.t. approved-book default <= risk
appetite (3.69%). Binds at 63.7% (income-max). Recommended operating point 61.9% - one prudential step
inside, strictly beats Champion on approval and default. Both shown in every table; 61.9% recommended.

## Module status
| Module | What | Key real numbers | State |
|---|---|---|---|
| A Application Risk | Custom PD vs incumbent, RAROC frontier, swap-set, stress, calibration | AUC 0.756 vs 0.718; optimum 61.9%/63.7%; 61,503 held-out loans | done |
| B Behavioural Risk | Delinquency + collections capture | pipeline_b clean | done |
| C Portfolio/Pricing + Reject Inference | KGB/AGB correction, swap-in | booked bad 20.18% vs through-the-door 34.10% (+69% bias); 35.5% rejects safer than worst-accepted decile; AUC 0.640 (3 shared feats) | done |
| D ECL / IFRS 9 | Staging, transition matrix, LGD, ECL-by-stage | LGD 41.8%; 39x ECL cliff S1->S2 | done |
| D Collections | C2B / B2N / cure roll-rate + strategy | behavioural B2N AUC 0.565->0.946; DPD bucket dominant | done |
| D Uplift / NBA | T-learner uplift on modification=treatment | 20,615 delinquent loans, 12.6% modified; naive ATE +4.2pp; top-20% persuadables +52pp uplift; decile-10 predicted +0.667 vs observed +0.745 | done |

Reject-inference caveat: AUC modest because only 3 features (score, DTI, amount) shared across accepted/rejected files - honest, not hidden.
Uplift caveat (stated in code + docs): modification is not randomly assigned, so this is OBSERVATIONAL
(confounded by selection into treatment); production needs randomised champion/challenger holdout or propensity adjustment. Not a causal claim.

## Docs (06_docs/, all on new palette)
optimal_cutoff_analysis.html (convergence chart, 61.9 + 63.7), technical_appendix.html (Sec 5 precision/recall
reframed analytically), raroc_methodology_paper.html, README.html, executive_dashboard.html,
executive_presentation.html, architecture_diagram.html, system_blueprint.md (master spec).

## Pipelines (run with python; outputs are JSON in 0N_outputs/)
01: _pipeline_a, _economics_a, _stress_staging_a, _appendix_stats_a
02: _pipeline_b
03: _pipeline_c, _reject_inference_c
04: _pipeline_d, _collections_d, _b2n_behavioural_d, _uplift_nba_d

## Data parsing gotchas (already fixed - do not regress)
- Freddie ACTUAL LOSS stored NEGATIVE -> use abs(); EAD = col 26 (zero-balance-removal UPB).
- Incumbent PD: no class_weight="balanced" (it inflated PDs); swap-set is rank-based so unaffected.
- dq parse: `str(s).strip().split(".")[0]`; force `astype(str)`; assert Stage-1 share > 0.80 as guard.
- Reject inference: true acceptance rate 37.5% from rejection_summary.csv, not the sample-mix artifact.

## Ways of working
- Steady, small daily GitHub commits (push little and often). Repo structure/hygiene must read elite at first glance.
- Rationale must feel natural/organic - explain why each step exists and how it helps, like a real bank function.

## Current status (2026-07-09)
- Phase 0 done: framing reworked so "incumbent / Champion" everywhere reads as a bureau-score benchmark (not a mature production process); README has a Scope and limitations section; Module A snapshot staging relabelled a PD-bucket proxy (real IFRS 9 staging is Module D only); `run_all.py` runs the pipelines in order and skips steps whose data is absent.
- Phase 1 done: `tests/` (decision-engine contracts + committed-output guards + a no-dash hygiene test), `.github/workflows/ci.yml` (ruff errors-only + pytest + demo smoke run), `pyproject.toml`, `Makefile`.

## Planned / remaining (priority order)
- Fold reject-inference + uplift onto a docs/HTML surface (Module D README done; add a C page).
- Re-execute the 25 notebooks against current pipeline outputs so no notebook contradicts the docs.
- Wire Module D staging signal into the decision engine; calibrate the 60/40 composite weight.
- NTC thin-file model (Home Credit), Bureau Custom Model (Product/Customer/Location signatures).
- MLOps / monitoring (PSI drift job), decision-engine refresh; rewrite Module B/C notebooks to the swap-set framing.
