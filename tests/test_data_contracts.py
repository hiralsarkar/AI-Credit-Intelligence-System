"""Guards on committed pipeline outputs and repo hygiene. Data-light."""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(rel):
    p = ROOT / rel
    if not p.exists():
        pytest.skip(f"{rel} not present in this checkout")
    return json.loads(p.read_text(encoding="utf-8"))


def test_module_a_custom_beats_benchmark_and_pd_is_valid():
    m = _load("01_module_a_application_risk/04_outputs/module_a_metrics.json")
    assert 0.5 < m["incumbent"]["auc"] <= 1.0
    assert 0.5 < m["custom"]["auc"] <= 1.0
    assert m["custom"]["auc"] > m["incumbent"]["auc"]
    assert m["swap_set"]["sc1_default_reduction_pp"] > 0


def test_module_b_custom_beats_benchmark():
    m = _load("02_module_b_behavioural_risk/04_outputs/module_b_metrics.json")
    assert m["custom"]["auc"] > m["incumbent"]["auc"]


def test_module_d_staging_and_lgd_are_plausible():
    d = _load("04_module_d_ecl_ifrs9/04_outputs/module_d_ecl.json")
    assert 0.0 < d["lgd"] < 1.0
    assert d["stage_distribution"]["stage1"] > 0.80   # panel guard
    for row in d["transition_matrix"].values():
        assert abs(sum(row.values()) - 1.0) < 0.02


def test_no_em_or_en_dashes_in_tracked_sources():
    import subprocess
    em, en = chr(0x2014), chr(0x2013)
    listed = subprocess.run(["git", "ls-files", "*.md", "*.py", "*.html"],
                            cwd=ROOT, capture_output=True, text=True).stdout.split()
    bad = [f for f in listed
           if any(ch in (ROOT / f).read_text(encoding="utf-8", errors="ignore") for ch in (em, en))]
    assert not bad, f"em/en dashes found in tracked files: {bad}"


def test_sc3_operating_point_agrees_across_outputs():
    """SC3 has one definition (_operating_point.py); every output must agree."""
    m = _load("01_module_a_application_risk/04_outputs/module_a_metrics.json")
    s = _load("01_module_a_application_risk/04_outputs/module_a_stress_staging.json")
    assert m["swap_set"]["sc3_custom_approval"] == s["sc3_point"]["approval"]
    assert m["swap_set"]["sc3_custom_default"] == s["sc3_point"]["default"]
    # SC3 must beat the benchmark on both axes
    assert s["sc3_point"]["approval"] > m["swap_set"]["anchor_incumbent_approval"]
    assert s["sc3_point"]["default"] < m["swap_set"]["anchor_incumbent_default"]


def test_pipelines_build_portable_paths():
    """Windows-only 'BASE + r"\\folder\\file"' paths write mangled names on Mac/Linux/CI."""
    import re
    pat = re.compile(r'\+\s*r"\\')
    bad = [str(p.relative_to(ROOT)) for p in ROOT.glob("0*/_*.py")
           if pat.search(p.read_text(encoding="utf-8"))]
    assert not bad, f"backslash path concatenation in: {bad}"
