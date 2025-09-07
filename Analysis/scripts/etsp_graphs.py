# --- ETSP graph export utilities -------------------------------------------------
# Drop this whole block at the end of load_experiment.py (after load_all_matrices)
# and run the module. It will:
#   1) Load all matrices via load_all_matrices().
#   2) Filter to 'generation_type' that means Euclidean TSP (ETSP).
#   3) Pick the top 1% hardest (highest 'iteration').
#   4) Reconstruct 2D coordinates from the distance matrix via Classical MDS.
#   5) Plot a clean node graph (optionally k-NN edges) and save into a folder.

import os
import re
import math
from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.load_experiment import load_all_matrices


# ---------- Coordinate reconstruction (Classical MDS) ---------------------------

def _classical_mds_from_distance(D: np.ndarray, output_dim: int = 2) -> np.ndarray:
    """
    Classical MDS (a.k.a. Torgerson–Gower) from a full distance matrix.
    Returns coordinates in R^output_dim that best (in least-squares sense)
    reproduce the given pairwise distances.
    Notes:
        - Expects a symmetric, non-negative matrix with zeros on the diagonal.
        - Negative eigenvalues can appear due to noise/rounding; we clamp them.
    """
    n = D.shape[0]
    if n == 0:
        return np.zeros((0, output_dim))

    # Ensure symmetry and zero diagonal for numerical stability
    D = np.minimum(D, D.T)
    D = D.copy()
    np.fill_diagonal(D, 0.0)

    # Double-centering: B = -1/2 * J * (D^2) * J
    J = np.eye(n) - np.ones((n, n)) / n
    D2 = D ** 2
    B = -0.5 * J @ D2 @ J

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]  # descending
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Take the first 'output_dim' components, clamp negatives to zero
    lam = np.clip(eigvals[:output_dim], a_min=0.0, a_max=None)
    # If all non-positive, return zeros to avoid NaNs
    if not np.any(lam > 0):
        return np.zeros((n, output_dim))

    L_sqrt = np.diag(np.sqrt(lam))
    V = eigvecs[:, :output_dim]
    X = V @ L_sqrt
    return X


# ---------- Lightweight graph drawing helpers ----------------------------------

def _knn_undirected_edges(D: np.ndarray, k: int = 3) -> List[Tuple[int, int]]:
    """
    Build an undirected k-NN edge list from a distance matrix D.
    Each node connects to its k nearest neighbors (excluding self),
    edges are deduplicated.
    """
    n = D.shape[0]
    edges = set()
    for i in range(n):
        row = D[i].copy()
        row[i] = np.inf  # exclude self
        # ignore inf/0 (0 only happens on the diagonal after symmetrization)
        nbrs = np.argsort(row)
        added = 0
        for j in nbrs:
            if math.isfinite(row[j]):
                a, b = (i, j) if i < j else (j, i)
                edges.add((a, b))
                added += 1
                if added >= k:
                    break
    return sorted(edges)


def _safe_filename_from_row(row: pd.Series) -> str:
    parts = []
    for key in [
        'distribution', 'generation_type', 'city_size', 'range',
        'mutation_type', 'generation', 'iteration'
    ]:
        val = row.get(key, None)
        if pd.isna(val) if isinstance(val, (float, pd._libs.missing.NAType)) else val is None:
            continue
        s = str(val)
        s = re.sub(r'[^A-Za-z0-9_.-]+', '-', s)
        parts.append(f"{key}-{s}")
    return "_".join(parts) or "etsp_graph"


def _plot_etsp_graph(coords: np.ndarray,
                      D: np.ndarray,
                      title: str = "",
                      k_edges: int = 3,
                      annotate: bool = True,
                      figsize: Tuple[int, int] = (6, 6),
                      show_edge_lengths: bool = False,
                      edge_length_fmt: str = ".1f",
                      annotate_xy: bool = False,
                      coord_fmt: str = ".1f",
                      show_axes: bool = False,
                      show_grid: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    
    drawn_edges = []
    # Optional k-NN edges for readability (complete graph would be too busy)
    if k_edges and k_edges > 0:
        drawn_edges = _knn_undirected_edges(D, k=k_edges)
        for i, j in drawn_edges:
            x1, y1 = coords[i, 0], coords[i, 1]
            x2, y2 = coords[j, 0], coords[j, 1]
            ax.plot([x1, x2], [y1, y2], lw=0.6, alpha=0.35)

            if show_edge_lengths:
                xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                ax.text(
                    xm, ym, format(D[i, j], edge_length_fmt),
                    fontsize=6, ha="center", va="center", alpha=0.85,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6),
                    zorder=3
                )

    ax.scatter(coords[:, 0], coords[:, 1], s=32, zorder=2)
    if annotate:
        for i, (x, y) in enumerate(coords):
            ax.text(x, y, str(i + 1), fontsize=8, ha='center', va='center', zorder=4)
        
    # tick coordinates
    if show_axes:
        ax.axis('on')
        if show_grid:
            ax.grid(True, linewidth=0.3, alpha=0.3)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    ax.set_aspect('equal', adjustable='datalim')
    if title:
        ax.set_title(title)
    return fig


# ---------- Public API ----------------------------------------------------------

def save_hardest_etsp_figures(
    df: pd.DataFrame,
    out_dir: str = "../figures/ETSP-Top1Pct",
    top_fraction: float = 0.01,
    k_edges: int = 3,
    annotate: bool = True,
) -> int:
    """
    From the full matrices DataFrame produced by load_all_matrices(),
    select the top `top_fraction` hardest ETSP matrices (by 'iteration'),
    reconstruct 2D coordinates from the distance matrix, and save figures.

    Args:
        df: DataFrame from load_all_matrices(); requires columns 'generation_type',
            'iteration', and 'matrix'.
        out_dir: Folder to save PNG figures.
        top_fraction: Fraction to keep (e.g., 0.01 for 1%).
        k_edges: Draw k-NN edges per node for readability (None/0 to disable).
        annotate: Whether to label nodes 1..n.

    Returns:
        Number of figures written.
    """
    if df is None or len(df) == 0:
        print("No data to process.")
        return 0

    gtype = df.get('generation_type')
    if gtype is None:
        print("DataFrame lacks 'generation_type'.")
        return 0

    # Normalize the generation type for matching ETSP
    etsp_aliases = {
        'etsp', 'euclidean', 'euclidean_tsp', 'euclidean-tsp', 'euclidean tsp'
    }
    df = df.copy()
    df['__gtype_norm'] = df['generation_type'].astype(str).str.lower()
    etsp_df = df[df['__gtype_norm'].isin(etsp_aliases)]

    # Keep only rows that have iteration values and a matrix
    etsp_df = etsp_df[(etsp_df['iteration'].notna()) & (etsp_df['matrix'].notna())]

    if etsp_df.empty:
        print("No ETSP rows with iterations & matrices found.")
        return 0

    # Determine threshold for top X% hardest (by highest iteration)
    q = max(0.0, min(1.0, 1.0 - top_fraction))
    thr = etsp_df['iteration'].quantile(q)
    hardest = etsp_df[etsp_df['iteration'] >= thr]

    # Guarantee at least one selection
    if hardest.empty:
        take_n = max(1, int(math.ceil(len(etsp_df) * top_fraction)))
        hardest = etsp_df.nlargest(take_n, 'iteration')

    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for _, row in hardest.sort_values('iteration', ascending=False).iterrows():
        M = row['matrix']
        if isinstance(M, list):
            M = np.array(M, dtype=float)
        if not isinstance(M, np.ndarray):
            continue

        # Symmetrize & clean for MDS
        M = np.minimum(M, M.T)
        M = M.copy()
        np.fill_diagonal(M, 0.0)

        coords = _classical_mds_from_distance(M, output_dim=2)

        title = (
            f"ETSP—gen={row.get('generation', 'NA')}, "
            f"Lital Iter={row.get('iteration', 'NA')}, "
            f"size={row.get('city_size', 'NA')}, {row.get('distribution', '')}, control={row.get('range', 'NA')}"
        )
        #fig = _plot_etsp_graph(coords, M, title=title, k_edges=k_edges, annotate=annotate)
        fig = _plot_etsp_graph(coords, M, title=title, k_edges=2, annotate=True, show_edge_lengths=True)

        fname = _safe_filename_from_row(row) + ".png"
        save_path = os.path.join(out_dir, fname)
        fig.savefig(save_path, dpi=220, bbox_inches='tight')
        plt.close(fig)
        saved += 1

    print(f"Saved {saved} figure(s) to: {os.path.abspath(out_dir)}")
    return saved


def generate_and_save_hardest_etsp(
    out_dir: str = "../figures/ETSP-Top1Pct",
    top_fraction: float = 0.01,
    k_edges: int = 3,
    annotate: bool = True,
) -> int:
    """
    Convenience wrapper that calls load_all_matrices() and then
    save_hardest_etsp_figures(...).
    """
    df = load_all_matrices()
    return save_hardest_etsp_figures(
        df,
        out_dir=out_dir,
        top_fraction=top_fraction,
        k_edges=k_edges,
        annotate=annotate,
    )

def save_easiest_etsp_figures(
    df: pd.DataFrame,
    out_dir: str = "../figures/ETSP-Bottom5Pct",
    bottom_fraction: float = 0.05,
    k_edges: int = 3,
    annotate: bool = True,
) -> int:
    """
    From the full matrices DataFrame produced by load_all_matrices(),
    select the bottom `bottom_fraction` easiest ETSP matrices (by lowest 'iteration'),
    reconstruct 2D coordinates, and save figures.

    Args:
        df: DataFrame from load_all_matrices(); needs 'generation_type', 'iteration', 'matrix'.
        out_dir: Folder to save PNG figures.
        bottom_fraction: Fraction to keep from the *lowest* iterations (e.g., 0.05 for 5%).
        k_edges: Draw k-NN edges per node (None/0 to disable).
        annotate: Whether to label nodes 1..n.

    Returns:
        Number of figures written.
    """
    if df is None or len(df) == 0:
        print("No data to process.")
        return 0

    if 'generation_type' not in df.columns:
        print("DataFrame lacks 'generation_type'.")
        return 0

    # Normalize generation type to match ETSP
    etsp_aliases = {
        'etsp', 'euclidean', 'euclidean_tsp', 'euclidean-tsp', 'euclidean tsp'
    }
    df = df.copy()
    df['__gtype_norm'] = df['generation_type'].astype(str).str.lower()
    # Be safe if iteration accidentally loaded as string
    df['iteration'] = pd.to_numeric(df['iteration'], errors='coerce')

    etsp_df = df[
        df['__gtype_norm'].isin(etsp_aliases)
        & df['iteration'].notna()
        & df['matrix'].notna()
    ]

    if etsp_df.empty:
        print("No ETSP rows with iterations & matrices found.")
        return 0

    # Threshold for bottom X% easiest (by lowest iteration)
    bottom_fraction = max(0.0, min(1.0, float(bottom_fraction)))
    thr = etsp_df['iteration'].quantile(bottom_fraction)
    easiest = etsp_df[etsp_df['iteration'] <= thr]

    # Guarantee at least one selection (handles tiny datasets / quantile edge cases)
    if easiest.empty:
        take_n = max(1, int(math.ceil(len(etsp_df) * bottom_fraction)))
        easiest = etsp_df.nsmallest(take_n, 'iteration')

    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for _, row in easiest.sort_values('iteration', ascending=True).iterrows():
        M = row['matrix']
        if isinstance(M, list):
            M = np.array(M, dtype=float)
        if not isinstance(M, np.ndarray):
            continue

        # Symmetrize & clean for MDS (ETSP should be symmetric; this guards noise)
        M = np.minimum(M, M.T)
        M = M.copy()
        np.fill_diagonal(M, 0.0)

        coords = _classical_mds_from_distance(M, output_dim=2)

        title = (
            f"ETSP—gen={row.get('generation', 'NA')}, "
            f"Lital Iter={row.get('iteration', 'NA')}, "
            f"size={row.get('city_size', 'NA')}, {row.get('distribution', '')}, control={row.get('range', 'NA')}"
        )
        fig = _plot_etsp_graph(coords, M, title=title, k_edges=k_edges, annotate=annotate, show_edge_lengths=True)

        fname = _safe_filename_from_row(row) + ".png"
        save_path = os.path.join(out_dir, fname)
        fig.savefig(save_path, dpi=220, bbox_inches='tight')
        plt.close(fig)
        saved += 1

    print(f"Saved {saved} figure(s) to: {os.path.abspath(out_dir)}")
    return saved


def generate_and_save_easiest_etsp(
    out_dir: str = "../figures/ETSP-Bottom5Pct",
    bottom_fraction: float = 0.05,
    k_edges: int = 3,
    annotate: bool = True,
) -> int:
    """
    Convenience wrapper that calls load_all_matrices() and then
    save_easiest_etsp_figures(...).
    """
    df = load_all_matrices()
    return save_easiest_etsp_figures(
        df,
        out_dir=out_dir,
        bottom_fraction=bottom_fraction,
        k_edges=k_edges,
        annotate=annotate,
    )

# ---------- CLI ---------------------------------------------------------
if __name__ == "__main__":
    # generate_and_save_hardest_etsp(
    #     out_dir="../figures/ETSP-Top5Pct",
    #     top_fraction=0.05,   # top 5%
    #     k_edges=3,           # 3-NN edges for legibility; set 0 to disable
    #     annotate=True,
    # )
    generate_and_save_easiest_etsp(
        out_dir="../figures/ETSP-Bottom5Pct",
        bottom_fraction=0.05,   # 5% easiest
        k_edges=3,
        annotate=True,
    )