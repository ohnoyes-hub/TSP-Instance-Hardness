"""
intercity_similarity_all_matrices.py
------------------------------------
Compute inter-city similarity metrics for *all generated matrices* in the Phase Transition experiments using `load_all_matrices()`.

This mirrors `intercity_similarity.py` (which analyzed hard instances) but scales to the full per-generation dataset.
Artifacts:
    - ./plot/intercity_similarity_all_matrices/jointplot_normed_unique_vs_log_iter.png
    - ./plot/intercity_similarity_all_matrices/jointplot_nearest_neighbor_vs_log_iter.png
    - ./plot/intercity_similarity_all_matrices/distinct_distances_<config>_<value>.png
    - ./plot/intercity_similarity_all_matrices/distinct_distances_<config>_<value>.csv
    - ./plot/intercity_similarity_all_matrices/normed_unique_summary.csv
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import seaborn as sns

# === Robust import of load_all_matrices() ====================================
from util.load_experiment import load_all_matrices

# === Core helpers (copied/adapted from intercity_similarity.py) ==============

def is_symmetric(matrix) -> bool:
    """Quick check for matrix symmetry."""
    arr = np.array(matrix, dtype=float)
    return np.allclose(arr, arr.T, atol=1e-8)

def _offdiag_values(arr: np.ndarray, symmetric: bool) -> np.ndarray:
    """Return off-diagonal values (finite only), half-matrix if symmetric."""
    if symmetric:
        vals = arr[np.triu_indices_from(arr, k=1)]
    else:
        vals = arr[~np.eye(arr.shape[0], dtype=bool)]
    vals = vals[np.isfinite(vals)]
    return vals

def count_distinct_distances(matrix, symmetric: bool = True) -> int:
    """Count distinct off-diagonal distances."""
    arr = np.array(matrix, dtype=float)
    vals = _offdiag_values(arr, symmetric)
    return np.unique(vals).size

def normed_unique(matrix, symmetric: bool = True) -> float:
    """#distinct off-diagonal distances normalized by total possible pairs."""
    arr = np.array(matrix, dtype=float)
    n = arr.shape[0]
    total_possible = (n * (n - 1)) // 2 if symmetric else n * (n - 1)
    if total_possible == 0:
        return np.nan
    count = count_distinct_distances(arr, symmetric)
    return count / total_possible

def normalized_nearest_neighbor_distance(matrix) -> float:
    """Mean(nearest_i / mean(nearest)) across nodes; diagonal treated as inf."""
    arr = np.array(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.nan
    np.fill_diagonal(arr, np.inf)
    nearest = np.min(arr, axis=1)
    mu = np.mean(nearest)
    return float(np.mean(nearest / mu)) if mu and np.isfinite(mu) else np.nan

# === Plotting helpers ========================================================

OUTDIR = "./plot/intercity_similarity_all_matrices"

def plot_joint_normed_distinct_vs_log_iter(df: pd.DataFrame, hue_col: str = "distribution"):
    os.makedirs(OUTDIR, exist_ok=True)
    # Expect 'normed_unique_distances' and 'log_iterations' present
    sns.set(style="whitegrid", context="talk")
    jp = sns.jointplot(
        data=df,
        x="log_iterations",
        y="normed_unique_distances",
        hue=hue_col if hue_col in df.columns else None,
        kind="scatter",
        marginal_kws=dict(common_norm=False, fill=True),
        alpha=0.4,
        height=8,
        palette="CMRmap",
        marginal_ticks=True
    )
    jp.set_axis_labels("Log Lital Iteration", "Normalized Distinct Distances", fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    out = os.path.join(OUTDIR, "jointplot_normed_unique_vs_log_iter.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

def plot_joint_nearest_neighbor_vs_log_iter(df: pd.DataFrame, hue_col: str = "distribution"):
    os.makedirs(OUTDIR, exist_ok=True)
    sns.set(style="whitegrid", context="talk")
    jp = sns.jointplot(
        data=df,
        x="log_iterations",
        y="nearest_neighbor_norm",
        hue=hue_col if hue_col in df.columns else None,
        kind="scatter",
        marginal_kws=dict(common_norm=False, fill=True),
        alpha=0.4,
        height=8,
        palette="CMRmap",
        marginal_ticks=True
    )
    jp.set_axis_labels("Log Lital Iteration", "Normalized Nearest Neighbor Distance", fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    out = os.path.join(OUTDIR, "jointplot_nearest_neighbor_vs_log_iter.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

def analyze_by_config(df: pd.DataFrame, config_col: str, output_prefix: str):
    """Scatter + correlation tables by a single config column (e.g., distribution)."""
    os.makedirs(OUTDIR, exist_ok=True)
    summary = []
    for cfg, sub in df.groupby(config_col):
        sub = sub.copy()
        # Correlations
        pearson_raw = pearsonr(sub['normed_unique_distances'], sub['iterations'])[0] if len(sub) > 1 else np.nan
        spearman_raw = spearmanr(sub['normed_unique_distances'], sub['iterations'])[0] if len(sub) > 1 else np.nan
        pearson_log = pearsonr(sub['normed_unique_distances'], sub['log_iterations'])[0] if len(sub) > 1 else np.nan
        spearman_log = spearmanr(sub['normed_unique_distances'], sub['log_iterations'])[0] if len(sub) > 1 else np.nan
        print(
            f"[{config_col}={cfg}]\n"
            f"Raw: Pearson={pearson_raw:.3f}, Spearman={spearman_raw:.3f}\n"
            f"Log: Pearson={pearson_log:.3f}, Spearman={spearman_log:.3f}"
        )
        # Scatter plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.scatter(sub['iterations'], sub['normed_unique_distances'], alpha=0.7)
        ax1.set_xlabel('Iterations (raw)')
        ax1.set_ylabel('Normalized Distinct Distances')
        ax1.set_title(f"Raw Iterations - {cfg}")
        ax1.grid(True)
        ax2.scatter(sub['log_iterations'], sub['normed_unique_distances'], alpha=0.7)
        ax2.set_xlabel('Log Lital Iterations')
        ax2.set_ylabel('Normalized Distinct Distances')
        ax2.set_title(f"Log-scaled Iterations - {cfg}")
        ax2.grid(True)
        plt.tight_layout()
        png_path = os.path.join(OUTDIR, f"{output_prefix}_{config_col}_{cfg}.png")
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()
        # Save CSV without matrix column
        to_save = sub.drop(columns=['matrix'], errors='ignore')
        csv_path = os.path.join(OUTDIR, f"{output_prefix}_{config_col}_{cfg}.csv")
        to_save.to_csv(csv_path, index=False)
        print(f"Saved {png_path} and {csv_path}\n")
        summary.append({
            config_col: cfg,
            "pearson_raw": pearson_raw,
            "spearman_raw": spearman_raw,
            "pearson_log": pearson_log,
            "spearman_log": spearman_log,
            "N": len(sub)
        })
    return pd.DataFrame(summary)

def print_normed_unique_summary(df: pd.DataFrame, threshold: float = 0.6):
    rows = []
    group_cols = [c for c in ['generation_type', 'distribution'] if c in df.columns]
    if not group_cols:
        group_cols = ['_all_']
        df = df.assign(_all_='all')
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        count_leq = (sub['normed_unique_distances'] <= threshold).sum()
        count_gt = (sub['normed_unique_distances'] > threshold).sum()
        avg_log_leq = sub.loc[sub['normed_unique_distances'] <= threshold, 'log_iterations'].mean()
        avg_log_gt = sub.loc[sub['normed_unique_distances'] > threshold, 'log_iterations'].mean()
        row = {k: v for k, v in zip(group_cols, keys)}
        row.update({
            f'count_≤{threshold}': int(count_leq),
            f'avg_log_iter_≤{threshold}': float(avg_log_leq) if pd.notnull(avg_log_leq) else np.nan,
            f'count_>{threshold}': int(count_gt),
            f'avg_log_iter_>{threshold}': float(avg_log_gt) if pd.notnull(avg_log_gt) else np.nan,
        })
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    print(summary_df)
    out = os.path.join(OUTDIR, "normed_unique_summary.csv")
    summary_df.to_csv(out, index=False)
    print(f"Saved summary to {out}")

# === Main ====================================================================

def main():
    # Load full per-generation dataset
    df = load_all_matrices()

    # Keep rows with actual matrices and valid, positive iteration counts
    df = df.copy()
    # Standardize column name to 'iterations' for downstream reuse
    if 'iterations' not in df.columns and 'iteration' in df.columns:
        df['iterations'] = df['iteration']
    # Drop missing/non-positive iterations
    df = df[pd.to_numeric(df['iterations'], errors='coerce').notnull()]
    df = df[df['iterations'] > 0]
    # Coerce matrices to numpy arrays (in case loader left lists for some entries)
    df['matrix'] = df['matrix'].apply(lambda m: np.array(m, dtype=float) if not isinstance(m, np.ndarray) else m)

    # Compute metrics (row-wise symmetry detection)
    df['symmetric'] = df['matrix'].apply(is_symmetric)
    df['normed_unique_distances'] = df.apply(
        lambda row: normed_unique(row['matrix'], row['symmetric']), axis=1
    )
    df['nearest_neighbor_norm'] = df['matrix'].apply(normalized_nearest_neighbor_distance)
    df['log_iterations'] = np.log(df['iterations'].astype(float))

    # Summary and plots
    print_normed_unique_summary(df)
    # Joint plots colored by distribution if available, else uncolored
    hue_col = 'distribution' if 'distribution' in df.columns else None
    plot_joint_normed_distinct_vs_log_iter(df, hue_col or 'distribution')
    plot_joint_nearest_neighbor_vs_log_iter(df, hue_col or 'distribution')

    # Per-config analysis
    for config_col in ['distribution', 'generation_type']:
        if config_col in df.columns:
            print(f"Analyzing by {config_col}...")
            summary = analyze_by_config(df, config_col=config_col, output_prefix='distinct_distances')
            # Optionally print summary table
            print(summary)

if __name__ == "__main__":
    main()
