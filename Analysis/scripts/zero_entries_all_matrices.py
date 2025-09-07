"""
zero_entries_all_matrices.py
----------------------------
Analyze zero entries in *all generated matrices* from the Phase Transition experiments,
using `load_all_matrices()` and relate them to the (per-generation) Lital iteration.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from util.load_experiment import load_all_matrices

# --- Load --------------------------------------------------------------------
df = load_all_matrices()

# Expect columns: ['matrix', 'iteration', 'generation', 'distribution', 'mutation_type', 'generation_type', 'city_size', 'range', ...]
# Keep rows with a valid numeric matrix and a positive iteration value
def to_numpy(m):
    if isinstance(m, np.ndarray):
        return m
    try:
        return np.array(m, dtype=float)
    except Exception:
        return None

df['matrix'] = df['matrix'].apply(to_numpy)
df = df[df['matrix'].notnull()]

# Keep only rows with positive iteration (some early/fallback rows may not have it)
df = df[df['iteration'].notnull()]
df = df[df['iteration'] > 0]

# --- Metrics: zero counts excluding diagonal ---------------------------------
def zero_metrics_excl_diag(A: np.ndarray):
    """
    Count zeros off-diagonal and compute ratio over off-diagonal entries.
    """
    if A.ndim != 2:
        return np.nan, np.nan, np.nan  # count, ratio, n
    n = A.shape[0]
    # mask off-diagonal
    mask = ~np.eye(n, dtype=bool)
    off = A[mask]
    total_off = off.size if off is not None else 0
    if total_off == 0:
        return 0, np.nan, n
    count_zero = np.sum(off == 0)
    ratio_zero = count_zero / total_off
    return int(count_zero), float(ratio_zero), int(n)

metrics = df['matrix'].apply(zero_metrics_excl_diag)
df[['zero_count', 'zero_ratio', 'n']] = pd.DataFrame(metrics.tolist(), index=df.index)

# Log iteration
df['log_iteration'] = np.log(df['iteration'].astype(float))

# --- Plot (jointplot) --------------------------------------------------------
sns.set(style="whitegrid", context="talk")

g = sns.jointplot(
    data=df,
    x='log_iteration',
    y='zero_count',
    hue='distribution',
    kind='scatter',
    palette='CMRmap',
    marginal_kws=dict(fill=True, common_norm=False, alpha=0.4),
    height=10,
    ratio=5,
    marginal_ticks=True
)

# Optional tweak to match your previous style
g.ax_marg_y.set_visible(False)
g.set_axis_labels('Log Lital Iteration', 'Zero Count (off-diagonal)', fontsize=12)

plt.tight_layout()
plt.subplots_adjust(top=0.95)

# Save figure
output_dir = "./plot/zero_entries_all_matrices"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "zero_entries_vs_log_iteration_joint.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Plot saved to {output_path}")
plt.show()

# --- Summary stats -----------------------------------------------------------
print("========= Summary Statistics by Distribution =========")
# Threshold for "many zeros"
COUNT_MAX = 200

# Helper safely-mean
def _mean_safe(s):
    s = pd.Series(s).dropna()
    return float(s.mean()) if len(s) else float('nan')

for dist in sorted(df['distribution'].dropna().unique()):
    sub = df[df['distribution'] == dist]
    eq0 = (sub['zero_count'] == 0).sum()
    gtT = (sub['zero_count'] > COUNT_MAX).sum()
    avg_logit_eq0 = _mean_safe(sub.loc[sub['zero_count'] == 0, 'log_iteration'])
    avg_logit_gtT = _mean_safe(sub.loc[sub['zero_count'] > COUNT_MAX, 'log_iteration'])
    print(f"Distribution: {dist}")
    print(f"  Zero count == 0: {eq0} (avg log(iter)={avg_logit_eq0:.2f})")
    print(f"  Zero count > {COUNT_MAX}: {gtT} (avg log(iter)={avg_logit_gtT:.2f})\n")

# Optional: quick peek per city size
# print('========= Summary by City Size =========')
# for n in sorted(df['n'].dropna().unique()):
#     sub = df[df['n'] == n]
#     print(f"n={int(n)}: rows={len(sub)}, zeros>0={(sub['zero_count']>0).sum()}")