#!/usr/bin/env python3
"""
run_all.py - regenerate the project's figures from data, in order, on a fixed seed.

This runs the module pipelines that produce the metrics and outputs quoted in the
docs. It is deliberately forgiving: any step whose input data is not present is
skipped with a note on where to get it, so a partial checkout still runs what it can.

  python run_all.py            run every step whose inputs are present
  python run_all.py --check    only report which inputs are present / missing
  python run_all.py --only a   run one module group: a | b | c | d

The decision engine needs no data at all:
  python 04_decision_engine/demo.py
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
FREDDIE = Path(os.environ.get("FREDDIE_RAW_DIR", str(Path.home() / "Downloads")))

A = "01_module_a_application_risk"
B = "02_module_b_behavioural_risk"
C = "03_module_c_portfolio_pricing_risk"
D = "04_module_d_ecl_ifrs9"

# Each step: (group, label, script, [required inputs], hint if missing)
STEPS = [
    ("a", "Module A - PD model, swap-set, scored output", f"{A}/_pipeline_a.py",
        [ROOT / A / "01_data/raw/application_train.csv"],
        "Download Home Credit (application_train.csv) -> 01_module_a_application_risk/01_data/raw/"),
    ("a", "Module A - EL, economic capital, RAROC by band", f"{A}/_economics_a.py",
        [ROOT / A / "01_data/processed/scored_test_a.csv"],
        "Produced by the Module A PD step above; run that first."),
    ("a", "Module A - stress test + PD-bucket provisioning proxy", f"{A}/_stress_staging_a.py",
        [ROOT / A / "01_data/processed/scored_test_a.csv"],
        "Produced by the Module A PD step above; run that first."),
    ("b", "Module B - delinquency model, collections capture", f"{B}/_pipeline_b.py",
        [ROOT / B / "01_data/processed/X_train_b.csv", ROOT / B / "01_data/processed/X_test_b.csv"],
        "Run 02_module_b_behavioural_risk/02_notebooks/01_data_preprocessing_b.ipynb first "
        "(needs cs-training.csv from Give Me Some Credit)."),
    ("c", "Module C - grade / market-implied PD, pricing", f"{C}/_pipeline_c.py",
        [ROOT / C / "01_data/processed/X_train_c.csv", ROOT / C / "01_data/processed/X_test_c.csv"],
        "Run 03_module_c_portfolio_pricing_risk/02_notebooks/01_data_preprocessing_c.ipynb first "
        "(needs accepted_2007_to_2018Q4.csv from LendingClub)."),
    ("c", "Module C - reject inference / selection bias", f"{C}/_reject_inference_c.py",
        [Path(os.environ.get("LC_REJECT_CSV",
              str(Path.home() / "Downloads" / "LendingClub Dataset" /
                  "rejected_2007_to_2018q4.csv" / "rejected_2007_to_2018Q4.csv")))],
        "Set LC_REJECT_CSV to the LendingClub rejected-loans CSV."),
    ("d", "Module D - IFRS 9 staging, transition matrix, ECL", f"{D}/_pipeline_d.py",
        [FREDDIE / f"sample_{y}.zip" for y in (2008, 2009, 2010, 2011)],
        "Place Freddie Mac sample_2008..2011.zip in FREDDIE_RAW_DIR (default ~/Downloads)."),
    ("d", "Module D - collections cure / roll-rate model", f"{D}/_collections_d.py",
        [FREDDIE / f"sample_{y}.zip" for y in (2008, 2009, 2010, 2011)],
        "Needs the Freddie Mac sample zips (see Module D step above)."),
    ("d", "Module D - behavioural bridge", f"{D}/_b2n_behavioural_d.py",
        [FREDDIE / f"sample_{y}.zip" for y in (2008, 2009, 2010, 2011)],
        "Needs the Freddie Mac sample zips (see Module D step above)."),
    ("d", "Module D - uplift / next-best-action (exploratory)", f"{D}/_uplift_nba_d.py",
        [FREDDIE / f"sample_{y}.zip" for y in (2008, 2009, 2010, 2011)],
        "Needs the Freddie Mac sample zips (see Module D step above)."),
]


def missing(paths):
    return [p for p in paths if not Path(p).exists()]


def main():
    args = sys.argv[1:]
    check_only = "--check" in args
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = args[i + 1].lower() if i + 1 < len(args) else None

    ran, skipped, failed = [], [], []
    print(f"Python: {PY}")
    print(f"Freddie raw dir (FREDDIE_RAW_DIR): {FREDDIE}\n")

    for group, label, script, needs, hint in STEPS:
        if only and group != only:
            continue
        gaps = missing(needs)
        if gaps:
            print(f"[skip] {label}")
            print(f"       missing: {gaps[0]}")
            print(f"       {hint}\n")
            skipped.append(label)
            continue
        if check_only:
            print(f"[ready] {label}")
            ran.append(label)
            continue
        print(f"[run ] {label}")
        rc = subprocess.run([PY, str(ROOT / script)], cwd=str(ROOT)).returncode
        if rc == 0:
            ran.append(label)
        else:
            print(f"       FAILED (exit {rc})\n")
            failed.append(label)

    verb = "ready" if check_only else "ran"
    print("\n" + "=" * 60)
    print(f"{verb}: {len(ran)}   skipped (no data): {len(skipped)}   failed: {len(failed)}")
    if failed:
        for f in failed:
            print(f"  FAILED: {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
