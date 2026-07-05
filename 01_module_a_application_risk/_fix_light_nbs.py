"""Light fixes to NB02, NB04, NB10: remove residual old-logic labels/strategies. Source edits;
clears stale outputs of edited code cells so no old labels remain in rendered output."""
import nbformat

def edit(path, repls, clear_if=()):
    nb = nbformat.read(path, as_version=4)
    changed = 0
    for cell in nb.cells:
        src = cell.source
        new = src
        for a, b in repls:
            if a in new:
                new = new.replace(a, b); changed += 1
        if new != src:
            cell.source = new
            if cell.cell_type == "code":
                cell.outputs = []; cell.execution_count = None
        # clear outputs of code cells that still render an old label
        if cell.cell_type == "code" and any(k in cell.source for k in clear_if):
            cell.outputs = []; cell.execution_count = None
    nbformat.write(nb, path)
    print(f"{path.split('/')[-1]}: {changed} replacement(s)")

# NB02 - rename illustrative threshold labels
edit("02_notebooks/02_credit_risk_model.ipynb", [
    ("Aggressive (45.6% approved)", "High approval (45.6%)"),
    ("RAROC-Gated (12.2% approved)", "Selective (12.2%)"),
    ("each strategy's target approval rate", "each policy's target approval rate"),
    ("XGBoost Champion", "XGBoost (discrimination benchmark)"),
])

# NB04 - reframe the flagged pricing header (RAROC methodology itself is fine)
edit("02_notebooks/04_expected_loss_capital_model.ipynb", [
    ("Risk-Based Pricing Adjustment", "Risk-Adjusted Pricing"),
])

# NB10 - replace the 4-strategy comparison with incumbent vs recommended; defer real staging to Module D
old_dict = """thresholds = {
    'Simulation Baseline (No Segmentation)': None,  # all loans
    'Aggressive (45.7%)': 0.457,
    'Legacy Hurdle (~12%)': 0.122,
    'Portfolio Economic Optimum (~17%)': 0.170
}"""
new_dict = """thresholds = {
    'Incumbent (bureau cutoff, 50%)': 0.50,
    'Recommended (SC3, 61.9%)': 0.619,
}"""
edit("02_notebooks/10_ifrs9_lifetime_pd_staging.ipynb", [
    (old_dict, new_dict),
    ("## 6. RAROC Gated Strategy: IFRS 9 ECL Reduction",
     "## 6. Approved-Book Stage Mix at Origination\\n\\n"
     "*Note: this is the origination-time stage mix of the approved book. Full IFRS 9 lifetime "
     "staging with real Stage 1 / Stage 2 / Stage 3 transitions and ECL is in Module D, built on "
     "the Freddie Mac loan-performance panel.*"),
])
print("done")
