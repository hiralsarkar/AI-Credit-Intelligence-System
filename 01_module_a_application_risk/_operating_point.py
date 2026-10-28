"""
_operating_point.py - one definition of the recommended operating point (SC3)

SC3 is the highest approval rate at which the custom model's approved book
still has a strictly lower realised default rate than the bureau-score
benchmark at its 50% anchor. Before this module existed, _pipeline_a.py and
_stress_staging_a.py each defined SC3 with a different margin and grid, so
module_a_metrics.json (62.6%) disagreed with the docs (61.9%). Both scripts
now call this function, so a rerun reproduces one number everywhere.
"""
import numpy as np

SC3_DEFAULT_MARGIN = 0.0008          # must beat the benchmark default by at least 8 bp
SC3_GRID = np.linspace(0.50, 0.80, 301)


def sc3_point(pd_scores, defaults, benchmark_default, margin=SC3_DEFAULT_MARGIN, grid=SC3_GRID):
    """Return (approval_rate, default_rate) for the SC3 point, or None if no point qualifies."""
    pd_scores = np.asarray(pd_scores)
    defaults = np.asarray(defaults)
    order = np.argsort(pd_scores)
    best = None
    for q in grid:
        n = int(len(pd_scores) * q)
        dr = defaults[order[:n]].mean()
        if dr < benchmark_default - margin:
            best = (float(q), float(dr))
    return best
