"""Unit tests for the Module A robustness statistics, on synthetic data. No dataset required."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_module_a_application_risk"))
from _robustness_lib import (  # noqa: E402
    auc, ks, default_at_approval, approval_at_default, swap_set,
    reliability_table, expected_calibration_error, brier, bootstrap,
)


def _data(n=20000, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-2.5 + signal * z)))     # true PD
    y = (rng.random(n) < p).astype(int)
    return y, p, z


def test_auc_matches_sklearn_including_ties():
    y, p, _ = _data()
    assert abs(auc(y, p) - roc_auc_score(y, p)) < 1e-9
    tied = np.round(p, 2)
    assert abs(auc(y, tied) - roc_auc_score(y, tied)) < 1e-9


def test_ks_is_between_zero_and_one_and_zero_for_noise():
    y, p, _ = _data()
    assert 0 < ks(y, p) < 1
    noise = np.random.default_rng(1).random(len(y))
    assert ks(y, noise) < 0.05


def test_approval_curve_is_consistent():
    y, p, _ = _data()
    d50 = default_at_approval(y, p, 0.5)
    assert d50 < y.mean()                                   # approving the safest half beats average
    assert approval_at_default(y, p, d50) >= 0.5 - 1e-4     # inverse relationship holds
    assert default_at_approval(y, p, 0.3) <= d50            # tighter approval, lower default


def test_swap_set_is_zero_for_identical_models_and_positive_for_better_one():
    y, p, z = _data()
    same = swap_set(y, p, p)
    assert abs(same["sc1_default_reduction_rel"]) < 1e-12
    assert abs(same["sc2_approval_gain_rel"]) < 1e-3
    weak = p + np.random.default_rng(2).normal(scale=0.15, size=len(p))   # noisier benchmark
    better = swap_set(y, weak, p)
    assert better["sc1_default_reduction_rel"] > 0
    assert better["sc2_approval_gain_rel"] > 0


def test_calibration_metrics_reward_true_probabilities():
    y, p, _ = _data(n=50000)
    rows = reliability_table(y, p, bins=10)
    assert len(rows) == 10 and sum(r["n"] for r in rows) == len(y)
    assert expected_calibration_error(y, p) < 0.01          # true PDs are calibrated
    inflated = np.clip(p * 2, 0, 1)
    assert expected_calibration_error(y, inflated) > expected_calibration_error(y, p)
    assert brier(y, inflated) > brier(y, p)


def test_bootstrap_interval_contains_point_and_shrinks_with_more_data():
    def stat(yy, s):
        return {"auc": auc(yy, s["m"])}
    y1, p1, _ = _data(n=4000, seed=3)
    y2, p2, _ = _data(n=40000, seed=3)
    small = bootstrap(y1, {"m": p1}, stat, n_boot=200)["auc"]
    big = bootstrap(y2, {"m": p2}, stat, n_boot=200)["auc"]
    assert small["lo"] <= small["point"] <= small["hi"]
    assert (big["hi"] - big["lo"]) < (small["hi"] - small["lo"])


def test_bootstrap_is_paired_so_identical_models_have_zero_width_difference():
    y, p, _ = _data(n=5000)
    def stat(yy, s):
        return {"diff": auc(yy, s["a"]) - auc(yy, s["b"])}
    r = bootstrap(y, {"a": p, "b": p.copy()}, stat, n_boot=100)["diff"]
    assert r["lo"] == r["hi"] == 0.0
