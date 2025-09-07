"""
tiq_all_matrices.py
-------------------
Triangle Inequality (TIQ) analysis over *all generated matrices* from the
Phase Transition experiments using `load_all_matrices()`.

Outputs:
  CSV:
    - ./plot/tiq_all/triangle_inequality_violations_all.csv
    - ./plot/tiq_all/summary_zero_vs_nonzero_by_config.csv

  Figures:
    - ./plot/tiq_all/scatter_violation_count_vs_log_iter.png
    - ./plot/tiq_all/scatter_violation_ratio_vs_log_iter.png
    - ./plot/tiq_all/scatter_avg_violation_mag_vs_log_iter.png
    - ./plot/tiq_all/scatter_violation_count_vs_log_iter_by_mutation.png
    - ./plot/tiq_all/scatter_violation_count_vs_log_iter_by_distribution.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from util.load_experiment import load_all_matrices

# === Styling (kept consistent with tiq.py look-and-feel) =====================
sns.set_theme(
    style="whitegrid",
    context="talk",
    palette="viridis",
    rc={
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "font.size": 14,
        "axes.titlesize": 22,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.weight": "bold",
    },
)

# === TIQ metric (same logic as in tiq.py for consistency) ====================
def triangle_inequality_violation(matrix):
    """
    Calculate triangle inequality violations for a distance matrix.
    For any i != j and any k not equal to i or j, check if d[i,j] > d[i,k] + d[k,j].
    Returns counts and magnitudes mirroring the original implementation.
    """
    M = np.array(matrix, dtype=float)
    n = M.shape[0]
    violation_count = 0
    violation_magnitude = 0.0
    checks = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                checks += 1
                s = M[i, k] + M[k, j]
                if M[i, j] > s:
                    violation_count += 1
                    violation_magnitude += (M[i, j] - s)

    avg_violation = violation_magnitude / violation_count if violation_count else 0.0
    violation_ratio = violation_count / checks if checks else 0.0
    return {
        "violation_count": int(violation_count),
        "total_violation_magnitude": float(violation_magnitude),
        "average_violation_magnitude": float(avg_violation),
        "violation_ratio": float(violation_ratio),
    }

# === Helpers =================================================================
OUTDIR = "./plot/tiq_all"
os.makedirs(OUTDIR, exist_ok=True)

def _coerce_matrix(x):
    import numpy as _np
    return x if isinstance(x, _np.ndarray) else _np.array(x, dtype=float)

def _prep_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize iteration column
    if "iterations" not in df.columns and "iteration" in df.columns:
        df = df.copy()
        df["iterations"] = df["iteration"]
    # Keep only valid iterations (positive)
    df = df[pd.to_numeric(df["iterations"], errors="coerce").notnull()]
    df = df[df["iterations"] > 0]
    # Coerce matrix
    df["matrix"] = df["matrix"].apply(_coerce_matrix)
    # Log iterations
    df["log_iterations"] = np.log(df["iterations"].astype(float))
    return df

def compute_tiq_over_all_matrices() -> pd.DataFrame:
    df = load_all_matrices()
    df = _prep_dataframe(df)
    # Compute TIQ metrics
    metrics = []
    for row in df.itertuples():
        tiq = triangle_inequality_violation(row.matrix)
        out = {
            **tiq,
            "generation": getattr(row, "generation", None),
            "iterations": getattr(row, "iterations", None),
            "log_iterations": getattr(row, "log_iterations", None),
            # configuration fields if present
            "distribution": getattr(row, "distribution", None),
            "generation_type": getattr(row, "generation_type", None),
            "mutation_type": getattr(row, "mutation_type", None),
            "city_size": getattr(row, "city_size", None),
            "range": getattr(row, "range", None),
        }
        metrics.append(out)
    res = pd.DataFrame(metrics)
    return res

def save_csv(df_all: pd.DataFrame):
    csv_path = os.path.join(OUTDIR, "triangle_inequality_violations_all.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"Saved TIQ metrics to {csv_path}")
    return csv_path

# === Plotting ================================================================
def _scatter(df, x, y, hue=None, title="", outname="plot.png"):
    plt.figure(figsize=(10, 6))
    if hue and hue in df.columns:
        sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.6)
        plt.legend(title=hue, loc="best")
    else:
        sns.scatterplot(data=df, x=x, y=y, alpha=0.6)
    plt.xlabel("Log Lital Iterations" if x == "log_iterations" else x)
    plt.ylabel(y.replace("_", " ").title())
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(OUTDIR, outname)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

def make_figures(df_all: pd.DataFrame):
    # Global scatters
    _scatter(df_all, "log_iterations", "violation_count",
             title="Triangle Inequality Violation Count vs Log Iterations",
             outname="scatter_violation_count_vs_log_iter.png")
    _scatter(df_all, "log_iterations", "violation_ratio",
             title="Triangle Inequality Violation Ratio vs Log Iterations",
             outname="scatter_violation_ratio_vs_log_iter.png")
    _scatter(df_all, "log_iterations", "average_violation_magnitude",
             title="Average Violation Magnitude vs Log Iterations",
             outname="scatter_avg_violation_mag_vs_log_iter.png")

    # By key configuration facets (if present)
    if "mutation_type" in df_all.columns and df_all["mutation_type"].notnull().any():
        _scatter(df_all, "log_iterations", "violation_count", hue="mutation_type",
                 title="Violation Count vs Log Iterations (by Mutation Type)",
                 outname="scatter_violation_count_vs_log_iter_by_mutation.png")
    if "distribution" in df_all.columns and df_all["distribution"].notnull().any():
        _scatter(df_all, "log_iterations", "violation_count", hue="distribution",
                 title="Violation Count vs Log Iterations (by Distribution)",
                 outname="scatter_violation_count_vs_log_iter_by_distribution.png")

def summary_zero_vs_nonzero(df_all: pd.DataFrame) -> pd.DataFrame:
    # Group by useful config columns if they exist
    group_cols = [c for c in ["generation_type", "distribution", "mutation_type", "city_size"] if c in df_all.columns]
    if not group_cols:
        group_cols = ["_all_"]
        df_all = df_all.assign(_all_="all")

    rows = []
    for keys, sub in df_all.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        zero = sub[sub["violation_count"] == 0]
        nonzero = sub[sub["violation_count"] > 0]
        row = {k: v for k, v in zip(group_cols, keys)}
        row.update({
            "count_zero": int(len(zero)),
            "count_nonzero": int(len(nonzero)),
            "avg_log_iter_zero": float(zero["log_iterations"].mean()) if len(zero) else np.nan,
            "avg_log_iter_nonzero": float(nonzero["log_iterations"].mean()) if len(nonzero) else np.nan,
        })
        rows.append(row)
    summary = pd.DataFrame(rows)
    out = os.path.join(OUTDIR, "summary_zero_vs_nonzero_by_config.csv")
    summary.to_csv(out, index=False)
    print(f"Saved summary to {out}")
    return summary

# === Main ====================================================================
def main():
    df_all = compute_tiq_over_all_matrices()
    save_csv(df_all)
    make_figures(df_all)
    summary_zero_vs_nonzero(df_all)

if __name__ == "__main__":
    main()
