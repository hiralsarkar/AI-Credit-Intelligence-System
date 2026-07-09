#!/usr/bin/env python3
"""
Re-execute the analysis notebooks in place so their outputs match the current
pipelines. Data-aware: a module is skipped (not failed) when its raw data is
absent, so a partial checkout still refreshes what it can.

  python tools/run_notebooks.py            execute every module whose data is present
  python tools/run_notebooks.py --check    only report what would run / be skipped
  python tools/run_notebooks.py --only a   one module: a | b | c

Needs the full stack (jupyter, nbconvert, xgboost, shap, ...): pip install -r requirements.txt
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# module -> (notebooks dir, raw file that must exist to run it)
GROUPS = {
    "a": ("01_module_a_application_risk", "01_data/raw/application_train.csv"),
    "b": ("02_module_b_behavioural_risk", "01_data/raw/cs-training.csv"),
    "c": ("03_module_c_portfolio_pricing_risk", "01_data/raw/accepted_2007_to_2018Q4.csv"),
}
CELL_TIMEOUT = 1800


def notebooks_for(mod_dir):
    nbs = sorted((ROOT / mod_dir / "02_notebooks").glob("*.ipynb"))
    return [n for n in nbs if ".ipynb_checkpoints" not in str(n)]


def main():
    args = sys.argv[1:]
    check = "--check" in args
    only = args[args.index("--only") + 1].lower() if "--only" in args else None

    plan = []
    for key, (mod_dir, raw) in GROUPS.items():
        if only and key != only:
            continue
        present = (ROOT / mod_dir / raw).exists()
        plan.append((key, mod_dir, raw, present, notebooks_for(mod_dir)))

    for key, mod_dir, raw, present, nbs in plan:
        if not present:
            print(f"[skip] module {key.upper()} ({len(nbs)} notebooks) - missing {raw}")

    to_run = [(k, d, nbs) for (k, d, r, present, nbs) in plan if present]
    if check:
        for k, d, nbs in to_run:
            print(f"[ready] module {k.upper()}: {len(nbs)} notebooks")
        return

    if not to_run:
        print("Nothing to run: no module has its raw data present.")
        return

    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError:
        sys.exit("Notebook stack missing. Run: pip install -r requirements.txt")

    failed = []
    for key, mod_dir, nbs in to_run:
        for nb_path in nbs:
            print(f"[run ] {nb_path.relative_to(ROOT)}")
            nb = nbformat.read(nb_path, as_version=4)
            ep = ExecutePreprocessor(timeout=CELL_TIMEOUT, kernel_name="python3")
            try:
                ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
                nbformat.write(nb, nb_path)
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"       FAILED: {type(e).__name__}: {str(e)[:200]}")
                failed.append(str(nb_path.relative_to(ROOT)))

    print("\n" + "=" * 60)
    print(f"failed notebooks: {len(failed)}")
    for f in failed:
        print(f"  {f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
