<<<<<<< HEAD
# --- Enhanced ETSP graph export utilities -------------------------------------------------
=======
# --- ETSP graph export utilities -------------------------------------------------
# Drop this whole block at the end of load_experiment.py (after load_all_matrices)
# and run the module. It will:
#   1) Load all matrices via load_all_matrices().
#   2) Filter to 'generation_type' that means Euclidean TSP (ETSP).
#   3) Pick the top 1% hardest (highest 'iteration').
#   4) Reconstruct 2D coordinates from the distance matrix via Classical MDS.
#   5) Plot a clean node graph (optionally k-NN edges) and save into a folder.
>>>>>>> a254e8852e74aa2d4f9baa764837b285f9ad6b3a

import os
import re
import math
<<<<<<< HEAD
import json
from typing import Iterable, Tuple, List, Optional, Dict, Any
from sklearn.manifold import MDS, TSNE
from scipy.spatial import ConvexHull, Voronoi
from matplotlib.patches import Polygon
import warnings
=======
from typing import Iterable, Tuple, List
>>>>>>> a254e8852e74aa2d4f9baa764837b285f9ad6b3a

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.load_experiment import load_all_matrices

<<<<<<< HEAD
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------- Coordinate reconstruction methods ---------------------------
=======

# ---------- Coordinate reconstruction (Classical MDS) ---------------------------
>>>>>>> a254e8852e74aa2d4f9baa764837b285f9ad6b3a

def _classical_mds_from_distance(D: np.ndarray, output_dim: int = 2) -> np.ndarray:
    """
    Classical MDS (a.k.a. Torgerson–Gower) from a full distance matrix.
    Returns coordinates in R^output_dim that best (in least-squares sense)
    reproduce the given pairwise distances.
<<<<<<< HEAD
=======
    Notes:
        - Expects a symmetric, non-negative matrix with zeros on the diagonal.
        - Negative eigenvalues can appear due to noise/rounding; we clamp them.
>>>>>>> a254e8852e74aa2d4f9baa764837b285f9ad6b3a
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


<<<<<<< HEAD
def _nonmetric_mds_from_distance(D: np.ndarray, output_dim: int = 2,
                                 n_init: int = 4, max_iter: int = 300) -> np.ndarray:
    """
    Non-metric MDS that preserves rank order of distances.
    Better for cases where exact distances are less important than relationships.
    """
    mds = MDS(n_components=output_dim, metric=False, dissimilarity='precomputed',
              n_init=n_init, max_iter=max_iter, random_state=42)
    return mds.fit_transform(D)


def _tsne_from_distance(D: np.ndarray, perplexity: float = None) -> np.ndarray:
    """
    t-SNE embedding for visualizing complex distance structures.
    Good for revealing clusters in TSP instances.
    """
    n = D.shape[0]
    # Auto-adjust perplexity based on dataset size
    if perplexity is None:
        perplexity = min(30.0, max(5.0, n / 3.0))
    
    tsne = TSNE(n_components=2, metric='precomputed', perplexity=perplexity,
                random_state=42, n_iter=1000)
    return tsne.fit_transform(D)


def reconstruct_coordinates(D: np.ndarray, method: str = 'classical_mds', **kwargs) -> np.ndarray:
    """
    Unified interface for coordinate reconstruction from distance matrix.
    
    Args:
        D: Distance matrix
        method: One of 'classical_mds', 'nonmetric_mds', 'tsne'
        **kwargs: Additional arguments for the specific method
    """
    methods = {
        'classical_mds': _classical_mds_from_distance,
        'nonmetric_mds': _nonmetric_mds_from_distance,
        'tsne': _tsne_from_distance
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")
    
    return methods[method](D, **kwargs)


# ---------- Graph analysis and metrics ----------------------------------
=======
# ---------- Lightweight graph drawing helpers ----------------------------------
>>>>>>> a254e8852e74aa2d4f9baa764837b285f9ad6b3a

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
<<<<<<< HEAD
=======
        # ignore inf/0 (0 only happens on the diagonal after symmetrization)
>>>>>>> a254e8852e74aa2d4f9baa764837b285f9ad6b3a
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


<<<<<<< HEAD
def calculate_clustering_coefficient(D: np.ndarray, k: int = 5) -> float:
    """Calculate clustering coefficient on k-NN graph."""
    n = D.shape[0]
    if n < 3:
        return 0.0
    
    edges = _knn_undirected_edges(D, k=k)
    
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    # Calculate clustering coefficient
    coefficients = []
    for node in range(n):
        neighbors = list(adj[node])
        if len(neighbors) < 2:
            continue
        
        # Count edges between neighbors
        edge_count = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in adj[neighbors[i]]:
                    edge_count += 1
        
        max_edges = len(neighbors) * (len(neighbors) - 1) / 2
        if max_edges > 0:
            coefficients.append(edge_count / max_edges)
    
    return np.mean(coefficients) if coefficients else 0.0


def calculate_hull_ratio(coords: np.ndarray) -> Optional[float]:
    """
    Calculate the ratio of cities on convex hull to total cities.
    This can indicate instance structure (higher ratio = more spread out).
    """
    n = len(coords)
    if n < 4:  # Need at least 4 points for meaningful hull
        return None
    
    try:
        hull = ConvexHull(coords)
        return len(hull.vertices) / n
    except Exception:
        return None


def calculate_instance_metrics(D: np.ndarray, coords: Optional[np.ndarray] = None) -> dict:
    """Calculate various metrics that correlate with TSP instance difficulty."""
    n = D.shape[0]
    
    if n < 2:
        return {}
    
    # Distance statistics
    triu_indices = np.triu_indices(n, k=1)
    distances = D[triu_indices]
    
    metrics = {
        'n_cities': n,
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'cv_distance': float(np.std(distances) / np.mean(distances)) if np.mean(distances) > 0 else 0,
        'distance_range': float(np.max(distances) - np.min(distances)),
        'distance_skewness': float(_calculate_skewness(distances)),
        'distance_kurtosis': float(_calculate_kurtosis(distances)),
    }
    
    # Nearest neighbor statistics
    nn_distances = []
    for i in range(n):
        non_zero = D[i, D[i] > 0]
        if len(non_zero) > 0:
            nn_distances.append(np.min(non_zero))
    
    if nn_distances:
        metrics.update({
            'mean_nn_distance': float(np.mean(nn_distances)),
            'std_nn_distance': float(np.std(nn_distances)),
            'cv_nn_distance': float(np.std(nn_distances) / np.mean(nn_distances)) if np.mean(nn_distances) > 0 else 0,
        })
    
    # Graph connectivity metrics
    metrics['clustering_coef_3nn'] = float(calculate_clustering_coefficient(D, k=3))
    metrics['clustering_coef_5nn'] = float(calculate_clustering_coefficient(D, k=5))
    
    # Spatial metrics if coordinates provided
    if coords is not None and len(coords) > 3:
        hull_ratio = calculate_hull_ratio(coords)
        if hull_ratio is not None:
            metrics['hull_ratio'] = float(hull_ratio)
        
        # Centroid deviation
        centroid = np.mean(coords, axis=0)
        distances_to_centroid = np.linalg.norm(coords - centroid, axis=1)
        metrics['mean_centroid_distance'] = float(np.mean(distances_to_centroid))
        metrics['std_centroid_distance'] = float(np.std(distances_to_centroid))
    
    return metrics


def _calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness of a distribution."""
    n = len(data)
    if n < 3:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return n * np.sum(((data - mean) / std) ** 3) / ((n - 1) * (n - 2))


def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate excess kurtosis of a distribution."""
    n = len(data)
    if n < 4:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return n * (n + 1) * np.sum(((data - mean) / std) ** 4) / ((n - 1) * (n - 2) * (n - 3)) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))


# ---------- Visualization functions --------------------------------------

def _safe_filename_from_row(row: pd.Series) -> str:
    """Generate safe filename from DataFrame row."""
=======
def _safe_filename_from_row(row: pd.Series) -> str:
>>>>>>> a254e8852e74aa2d4f9baa764837b285f9ad6b3a
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
<<<<<<< HEAD
                      show_axes: bool = False,
                      show_grid: bool = True,
                      show_heatmap: bool = False,
                      tour: Optional[List[int]] = None,
                      tour_color: str = 'red',
                      tour_alpha: float = 0.7) -> plt.Figure:
    """
    Enhanced plotting function with multiple visualization options.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Optional distance heatmap
    if show_heatmap and len(coords) > 3:
        try:
            _add_distance_heatmap(ax, coords, D)
        except Exception as e:
            print(f"Warning: Could not add heatmap: {e}")
    
    # Draw k-NN edges
    if k_edges and k_edges > 0 and tour is None:  # Don't show k-NN if showing tour
        edges = _knn_undirected_edges(D, k=k_edges)
        for i, j in edges:
            x1, y1 = coords[i, 0], coords[i, 1]
            x2, y2 = coords[j, 0], coords[j, 1]
            ax.plot([x1, x2], [y1, y2], 'gray', lw=0.6, alpha=0.35, zorder=1)
=======
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
>>>>>>> a254e8852e74aa2d4f9baa764837b285f9ad6b3a

            if show_edge_lengths:
                xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                ax.text(
                    xm, ym, format(D[i, j], edge_length_fmt),
                    fontsize=6, ha="center", va="center", alpha=0.85,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6),
                    zorder=3
                )
<<<<<<< HEAD
    
    # Draw tour if provided
    if tour is not None:
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            x1, y1 = coords[tour[i], 0], coords[tour[i], 1]
            x2, y2 = coords[tour[j], 0], coords[tour[j], 1]
            ax.plot([x1, x2], [y1, y2], color=tour_color, lw=2, alpha=tour_alpha, zorder=5)
    
    # Draw nodes
    ax.scatter(coords[:, 0], coords[:, 1], s=50, c='lightblue', 
              edgecolors='darkblue', linewidths=1, zorder=10)
    
    # Annotate nodes
    if annotate:
        for i, (x, y) in enumerate(coords):
            ax.text(x, y, str(i + 1), fontsize=8, ha='center', va='center', 
                   zorder=11, fontweight='bold')
    
    # Axes configuration
=======

    ax.scatter(coords[:, 0], coords[:, 1], s=32, zorder=2)
    if annotate:
        for i, (x, y) in enumerate(coords):
            ax.text(x, y, str(i + 1), fontsize=8, ha='center', va='center', zorder=4)
        
    # tick coordinates
>>>>>>> a254e8852e74aa2d4f9baa764837b285f9ad6b3a
    if show_axes:
        ax.axis('on')
        if show_grid:
            ax.grid(True, linewidth=0.3, alpha=0.3)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
<<<<<<< HEAD
    
    ax.set_aspect('equal', adjustable='datalim')
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')
    
    return fig


def _add_distance_heatmap(ax: plt.Axes, coords: np.ndarray, D: np.ndarray,
                          cmap: str = 'YlOrRd', alpha: float = 0.2):
    """Overlay a heatmap showing distance relationships using Voronoi cells."""
    if len(coords) < 4:
        return
    
    try:
        # Add boundary points for better Voronoi diagram
        x_min, y_min = coords.min(axis=0) - 1
        x_max, y_max = coords.max(axis=0) + 1
        boundary_points = np.array([
            [x_min, y_min], [x_min, y_max],
            [x_max, y_min], [x_max, y_max]
        ])
        extended_coords = np.vstack([coords, boundary_points])
        
        vor = Voronoi(extended_coords)
        
        # Color only the original points' regions
        for i in range(len(coords)):
            region_idx = vor.point_region[i]
            if region_idx == -1:
                continue
            region = vor.regions[region_idx]
            if not region or -1 in region:
                continue
            
            polygon = [vor.vertices[j] for j in region]
            if len(polygon) >= 3:
                # Calculate average distance for this city
                avg_dist = np.mean(D[i, :])
                normalized_dist = (avg_dist - D.min()) / (D.max() - D.min() + 1e-10)
                color = plt.cm.get_cmap(cmap)(normalized_dist)
                poly = Polygon(polygon, facecolor=color, alpha=alpha, edgecolor='none')
                ax.add_patch(poly)
    except Exception as e:
        print(f"Voronoi heatmap failed: {e}")


def _plot_on_axis(ax: plt.Axes, coords: np.ndarray, D: np.ndarray,
                  title: str = "", k_edges: int = 3, annotate: bool = True):
    """Helper function to plot on a specific axis (for grid plots)."""
    # Draw k-NN edges
    if k_edges and k_edges > 0:
        edges = _knn_undirected_edges(D, k=k_edges)
        for i, j in edges:
            x1, y1 = coords[i, 0], coords[i, 1]
            x2, y2 = coords[j, 0], coords[j, 1]
            ax.plot([x1, x2], [y1, y2], 'gray', lw=0.5, alpha=0.3)
    
    # Draw nodes
    ax.scatter(coords[:, 0], coords[:, 1], s=30, c='lightblue',
              edgecolors='darkblue', linewidths=0.5)
    
    # Annotate
    if annotate:
        for i, (x, y) in enumerate(coords):
            ax.text(x, y, str(i + 1), fontsize=6, ha='center', va='center')
    
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def create_comparison_grid(df: pd.DataFrame,
                          selection_criteria: List[dict],
                          out_file: str = "comparison_grid.png",
                          cols: int = 3,
                          embedding_method: str = 'classical_mds') -> plt.Figure:
    """
    Create a grid comparing different types of instances.
    """
    # Filter to ETSP instances
    etsp_aliases = {'etsp', 'euclidean', 'euclidean_tsp', 'euclidean-tsp', 'euclidean tsp'}
    df = df.copy()
    df['__gtype_norm'] = df['generation_type'].astype(str).str.lower()
    etsp_df = df[df['__gtype_norm'].isin(etsp_aliases)]
    etsp_df = etsp_df[(etsp_df['iteration'].notna()) & (etsp_df['matrix'].notna())]
    
    if etsp_df.empty:
        print("No ETSP instances found for comparison grid")
        return None
    
    rows = math.ceil(len(selection_criteria) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for idx, criteria in enumerate(selection_criteria):
        subset = criteria['filter'](etsp_df)
        if not subset.empty:
            row = subset.iloc[0]
            M = row['matrix']
            if isinstance(M, list):
                M = np.array(M)
            
            # Symmetrize
            M = np.minimum(M, M.T)
            np.fill_diagonal(M, 0.0)
            
            coords = reconstruct_coordinates(M, method=embedding_method)
            
            ax = axes[idx]
            _plot_on_axis(ax, coords, M, title=criteria['label'], k_edges=3)
    
    # Hide unused subplots
    for idx in range(len(selection_criteria), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if out_file:
        fig.savefig(out_file, dpi=150, bbox_inches='tight')
        print(f"Saved comparison grid to {out_file}")
    
    return fig


# ---------- Export functions ---------------------------------------------

def export_instance_data(row: pd.Series, coords: np.ndarray,
                        D: np.ndarray, output_dir: str = ".",
                        formats: List[str] = ['json', 'tsplib']) -> dict:
    """Export instance in multiple formats for different tools."""
    outputs = {}
    base_name = _safe_filename_from_row(row)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if 'json' in formats:
        # JSON format for web visualization or further processing
        data = {
            'metadata': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                        for k, v in row.to_dict().items() if k != 'matrix'},
            'coordinates': coords.tolist(),
            'distance_matrix': D.tolist(),
            'metrics': calculate_instance_metrics(D, coords)
        }
        
        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        outputs['json'] = json_path
    
    if 'tsplib' in formats:
        # TSPLIB format for TSP solvers
        tsp_content = generate_tsplib_format(coords, base_name, row)
        tsp_path = os.path.join(output_dir, f"{base_name}.tsp")
        with open(tsp_path, 'w') as f:
            f.write(tsp_content)
        outputs['tsplib'] = tsp_path
    
    return outputs


def generate_tsplib_format(coords: np.ndarray, name: str, metadata: pd.Series = None) -> str:
    """Generate TSPLIB95 format string."""
    lines = [
        f"NAME: {name}",
        f"TYPE: TSP",
        f"DIMENSION: {len(coords)}",
        f"EDGE_WEIGHT_TYPE: EUC_2D",
    ]
    
    if metadata is not None:
        lines.append(f"COMMENT: Generation {metadata.get('generation', 'NA')}, "
                    f"Iteration {metadata.get('iteration', 'NA')}")
    
    lines.append("NODE_COORD_SECTION")
    
    for i, (x, y) in enumerate(coords, 1):
        lines.append(f"{i} {x:.6f} {y:.6f}")
    
    lines.append("EOF")
    return "\n".join(lines)


# ---------- Enhanced main processing functions ---------------------------

def save_etsp_figures_enhanced(
    df: pd.DataFrame,
    out_dir: str = "../figures/ETSP-Enhanced",
    selection_type: str = 'hardest',  # 'hardest', 'easiest', or 'both'
    fraction: float = 0.01,
    k_edges: int = 3,
    annotate: bool = True,
    embedding_method: str = 'classical_mds',
    show_heatmap: bool = False,
    export_formats: List[str] = ['json'],
    compute_metrics: bool = True,
    create_interactive: bool = True
) -> int:
    """
    Enhanced version with multiple embedding methods and export options.
=======

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
>>>>>>> a254e8852e74aa2d4f9baa764837b285f9ad6b3a
    """
    if df is None or len(df) == 0:
        print("No data to process.")
        return 0

<<<<<<< HEAD
    # Filter ETSP instances
    etsp_aliases = {'etsp', 'euclidean', 'euclidean_tsp', 'euclidean-tsp', 'euclidean tsp'}
    df = df.copy()
    df['__gtype_norm'] = df['generation_type'].astype(str).str.lower()
    df['iteration'] = pd.to_numeric(df['iteration'], errors='coerce')
    
=======
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

>>>>>>> a254e8852e74aa2d4f9baa764837b285f9ad6b3a
    etsp_df = df[
        df['__gtype_norm'].isin(etsp_aliases)
        & df['iteration'].notna()
        & df['matrix'].notna()
    ]

    if etsp_df.empty:
        print("No ETSP rows with iterations & matrices found.")
        return 0

<<<<<<< HEAD
    # Select instances based on type
    if selection_type == 'hardest':
        q = max(0.0, min(1.0, 1.0 - fraction))
        thr = etsp_df['iteration'].quantile(q)
        selected = etsp_df[etsp_df['iteration'] >= thr]
        if selected.empty:
            take_n = max(1, int(math.ceil(len(etsp_df) * fraction)))
            selected = etsp_df.nlargest(take_n, 'iteration')
    elif selection_type == 'easiest':
        thr = etsp_df['iteration'].quantile(fraction)
        selected = etsp_df[etsp_df['iteration'] <= thr]
        if selected.empty:
            take_n = max(1, int(math.ceil(len(etsp_df) * fraction)))
            selected = etsp_df.nsmallest(take_n, 'iteration')
    else:  # both
        # Get both hardest and easiest
        q_hard = max(0.0, min(1.0, 1.0 - fraction))
        thr_hard = etsp_df['iteration'].quantile(q_hard)
        hardest = etsp_df[etsp_df['iteration'] >= thr_hard]
        
        thr_easy = etsp_df['iteration'].quantile(fraction)
        easiest = etsp_df[etsp_df['iteration'] <= thr_easy]
        
        selected = pd.concat([hardest, easiest])

    os.makedirs(out_dir, exist_ok=True)
    
    # Store metrics for analysis
    all_metrics = []
    
    saved = 0
    for _, row in selected.iterrows():
=======
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
>>>>>>> a254e8852e74aa2d4f9baa764837b285f9ad6b3a
        M = row['matrix']
        if isinstance(M, list):
            M = np.array(M, dtype=float)
        if not isinstance(M, np.ndarray):
            continue

<<<<<<< HEAD
        # Symmetrize & clean
=======
        # Symmetrize & clean for MDS (ETSP should be symmetric; this guards noise)
>>>>>>> a254e8852e74aa2d4f9baa764837b285f9ad6b3a
        M = np.minimum(M, M.T)
        M = M.copy()
        np.fill_diagonal(M, 0.0)

<<<<<<< HEAD
        # Reconstruct coordinates using specified method
        try:
            coords = reconstruct_coordinates(M, method=embedding_method)
        except Exception as e:
            print(f"Failed to reconstruct coordinates: {e}")
            coords = _classical_mds_from_distance(M)  # Fallback

        # Create title
        title = (
            f"ETSP–gen={row.get('generation', 'NA')}, "
            f"Iter={row.get('iteration', 'NA')}, "
            f"n={row.get('city_size', 'NA')}, "
            f"{embedding_method.replace('_', ' ').title()}"
        )
        
        # Plot figure
        fig = _plot_etsp_graph(
            coords, M, title=title, 
            k_edges=k_edges, 
            annotate=annotate,
            show_edge_lengths=False,
            show_heatmap=show_heatmap
        )

        # Save figure
        fname = _safe_filename_from_row(row) + f"_{embedding_method}.png"
        save_path = os.path.join(out_dir, fname)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Export additional formats
        if export_formats:
            export_dir = os.path.join(out_dir, 'exports')
            export_instance_data(row, coords, M, export_dir, export_formats)
        
        # Compute and store metrics
        if compute_metrics:
            metrics = calculate_instance_metrics(M, coords)
            metrics.update({
                'generation': row.get('generation'),
                'iteration': row.get('iteration'),
                'city_size': row.get('city_size'),
                'distribution': row.get('distribution'),
                'embedding_method': embedding_method
            })
            all_metrics.append(metrics)
        
        # Create interactive HTML visualization
        if create_interactive:
            html_path = os.path.join(out_dir, _safe_filename_from_row(row) + f"_{embedding_method}.html")
            create_interactive_html(coords, M, row, html_path)
        
        saved += 1


    # Save metrics to CSV
    if compute_metrics and all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(out_dir, 'instance_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics to {metrics_path}")

=======
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

>>>>>>> a254e8852e74aa2d4f9baa764837b285f9ad6b3a
    print(f"Saved {saved} figure(s) to: {os.path.abspath(out_dir)}")
    return saved


<<<<<<< HEAD
# ---------- Interactive visualization --------------------------------------

def create_interactive_html(coords: np.ndarray, D: np.ndarray,
                           metadata: dict, output_file: str,
                           save_png: bool = True):
    """Create an interactive HTML visualization using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Run: pip install plotly")
        return
    
    # Create interactive plot
    fig = go.Figure()
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode='markers+text',
        text=[str(i+1) for i in range(len(coords))],
        textposition="middle center",
        marker=dict(size=15, color='lightblue', line=dict(width=1, color='darkblue')),
        hovertemplate='City %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
    ))
    
    # Add k-NN edges
    edges = _knn_undirected_edges(D, k=3)
    edge_x = []
    edge_y = []
    for i, j in edges:
        edge_x.extend([coords[i, 0], coords[j, 0], None])
        edge_y.extend([coords[i, 1], coords[j, 1], None])
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='gray'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Calculate metrics for display
    metrics = calculate_instance_metrics(D, coords)
    
    # Create info text
    info_text = f"""
    <b>Instance Information:</b><br>
    Generation: {metadata.get('generation', 'Unknown')}<br>
    Iteration: {metadata.get('iteration', 'Unknown')}<br>
    Cities: {metadata.get('city_size', len(coords))}<br>
    <br>
    <b>Metrics:</b><br>
    Mean Distance: {metrics.get('mean_distance', 0):.2f}<br>
    CV Distance: {metrics.get('cv_distance', 0):.3f}<br>
    Clustering (3-NN): {metrics.get('clustering_coef_3nn', 0):.3f}<br>
    Hull Ratio: {metrics.get('hull_ratio', 0):.3f}
    """
    
    fig.add_annotation(
        text=info_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
        align="left",
        font=dict(size=10)
    )
    
    fig.update_layout(
        title=f"Interactive TSP Instance - {metadata.get('generation', 'Unknown')} Generation",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=True, zeroline=False, title="X"),
        yaxis=dict(showgrid=True, zeroline=False, scaleanchor='x', scaleratio=1, title="Y"),
        width=800,
        height=800
    )

    # Optionally save static PNG
    if save_png:
        png_file = output_file.replace('.html', '.png')
        fig.write_image(png_file, scale=2)
        print(f"Saved static PNG to {png_file}")    
    else:
        fig.write_html(output_file)
        print(f"Saved interactive visualization to {output_file}")


# ---------- Analysis and comparison functions -----------------------------

def analyze_difficulty_correlation(df: pd.DataFrame, output_dir: str = "../analysis", embedding_method: str = 'classical_mds'):
    """
    Analyze correlation between instance metrics and difficulty (iterations).
    """
    # Filter ETSP instances
    etsp_aliases = {'etsp', 'euclidean', 'euclidean_tsp', 'euclidean-tsp', 'euclidean tsp'}
    df = df.copy()
    df['__gtype_norm'] = df['generation_type'].astype(str).str.lower()
    df['iteration'] = pd.to_numeric(df['iteration'], errors='coerce')
    
    etsp_df = df[
        df['__gtype_norm'].isin(etsp_aliases)
        & df['iteration'].notna()
        & df['matrix'].notna()
    ]
    
    if etsp_df.empty:
        print("No ETSP instances found for analysis")
        return None
    
    print(f"Analyzing {len(etsp_df)} ETSP instances...")
    
    # Calculate metrics for all instances
    all_metrics = []
    for idx, (_, row) in enumerate(etsp_df.iterrows()):
        if idx % 10 == 0:
            print(f"Processing instance {idx+1}/{len(etsp_df)}")
        
        M = row['matrix']
        if isinstance(M, list):
            M = np.array(M, dtype=float)
        if not isinstance(M, np.ndarray):
            continue
        
        # Symmetrize
        M = np.minimum(M, M.T)
        np.fill_diagonal(M, 0.0)
        
        # Get coordinates
        try:
            coords = reconstruct_coordinates(M, method=embedding_method)
        except:
            coords = None
        
        # Calculate metrics
        metrics = calculate_instance_metrics(M, coords)
        metrics['iteration'] = row['iteration']
        metrics['generation'] = row.get('generation')
        metrics['city_size'] = row.get('city_size')
        all_metrics.append(metrics)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Calculate correlations with iteration (difficulty)
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    correlations = metrics_df[numeric_cols].corr()['iteration'].sort_values(ascending=False)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full metrics
    metrics_df.to_csv(os.path.join(output_dir, 'instance_metrics_full.csv'), index=False)
    
    # Save correlations
    correlations.to_csv(os.path.join(output_dir, 'difficulty_correlations.csv'))
    
    # Create correlation plot
    fig, ax = plt.subplots(figsize=(10, 6))
    correlations_plot = correlations[correlations.index != 'iteration']
    colors = ['green' if x > 0 else 'red' for x in correlations_plot.values]
    correlations_plot.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Correlation with Difficulty (Iterations)')
    ax.set_title('Feature Correlation with TSP Instance Difficulty')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'correlation_plot.png'), dpi=150)
    plt.close()
    
    print(f"\nTop correlations with difficulty:")
    print(correlations.head(10))
    print(f"\nAnalysis saved to {output_dir}")
    
    return metrics_df


def create_difficulty_comparison(df: pd.DataFrame,
                                 output_file: str = "difficulty_comparison.png",
                                 n_samples: int = 3) -> plt.Figure:
    """
    Create visual comparison of easy vs hard instances.
    """
    # Prepare selection criteria
    selection_criteria = [
        {'label': f'Easiest {n_samples}', 
         'filter': lambda df: df.nsmallest(n_samples, 'iteration')},
        {'label': f'Hardest {n_samples}', 
         'filter': lambda df: df.nlargest(n_samples, 'iteration')},
        {'label': f'Median {n_samples}',
         'filter': lambda df: df.iloc[len(df)//2 - n_samples//2:len(df)//2 + n_samples//2 + 1][:n_samples]}
    ]
    
    # Create comparison grid
    fig = create_comparison_grid(
        df, 
        selection_criteria, 
        output_file,
        cols=n_samples,
        embedding_method='classical_mds'
    )
    
    return fig


# ---------- Main execution functions --------------------------------------

def generate_comprehensive_analysis(
    out_dir: str = "../figures/ETSP-Analysis",
    top_fraction: float = 0.05,
    embedding_methods: List[str] = ['classical_mds', 'nonmetric_mds', 'tsne'],
    export_formats: List[str] = ['json', 'tsplib']
) -> None:
    """
    Run comprehensive analysis with multiple embedding methods and exports.
    """
    print("Loading data...")
    df = load_all_matrices()
    
    if df is None or df.empty:
        print("No data loaded")
        return
    
    print(f"Loaded {len(df)} instances")
    
    # Create main output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Generate visualizations for each embedding method
    for method in embedding_methods:
        print(f"\n{'='*60}")
        print(f"Processing with {method}...")
        
        method_dir = os.path.join(out_dir, method)
        
        # Save hardest instances
        print(f"Saving top {top_fraction*100}% hardest instances...")
        save_etsp_figures_enhanced(
            df,
            out_dir=os.path.join(method_dir, f"top_{int(top_fraction*100)}pct"),
            selection_type='hardest',
            fraction=top_fraction,
            embedding_method=method,
            show_heatmap=(method == 'classical_mds'),  # Only show heatmap for one method
            export_formats=export_formats if method == 'classical_mds' else [],
            compute_metrics=(method == 'classical_mds')
        )
        
        # Save easiest instances
        print(f"Saving bottom {top_fraction*100}% easiest instances...")
        save_etsp_figures_enhanced(
            df,
            out_dir=os.path.join(method_dir, f"bottom_{int(top_fraction*100)}pct"),
            selection_type='easiest',
            fraction=top_fraction,
            embedding_method=method,
            show_heatmap=False,
            export_formats=[],
            compute_metrics=False
        )
    
    # 2. Create comparison visualizations
    print(f"\n{'='*60}")
    print("Creating comparison visualizations...")
    
    comparison_dir = os.path.join(out_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Difficulty comparison
    create_difficulty_comparison(
        df,
        output_file=os.path.join(comparison_dir, "difficulty_comparison.png"),
        n_samples=4
    )
    
    # Method comparison (same instance, different embeddings)
    print("Creating embedding method comparison...")
    fig = create_embedding_comparison(df, comparison_dir)
    
    # 3. Run correlation analysis
    print(f"\n{'='*60}")
    print("Running correlation analysis...")
    analyze_difficulty_correlation(
        df,
        output_dir=os.path.join(out_dir, "analysis"),
        embedding_method='classical_mds'
    )
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to {os.path.abspath(out_dir)}")


def create_embedding_comparison(df: pd.DataFrame, output_dir: str) -> plt.Figure:
    """
    Compare different embedding methods on the same instance.
    """
    # Filter ETSP instances
    etsp_aliases = {'etsp', 'euclidean', 'euclidean_tsp', 'euclidean-tsp', 'euclidean tsp'}
    df = df.copy()
    df['__gtype_norm'] = df['generation_type'].astype(str).str.lower()
    
    etsp_df = df[
        df['__gtype_norm'].isin(etsp_aliases)
        & df['iteration'].notna()
        & df['matrix'].notna()
    ]
    
    if etsp_df.empty:
        return None
    
    # Pick a representative hard instance
    hard_instance = etsp_df.nlargest(1, 'iteration').iloc[0]
    M = hard_instance['matrix']
    if isinstance(M, list):
        M = np.array(M, dtype=float)
    
    # Symmetrize
    M = np.minimum(M, M.T)
    np.fill_diagonal(M, 0.0)
    
    # Create comparison
    methods = ['classical_mds', 'nonmetric_mds', 'tsne']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, method in enumerate(methods):
        try:
            coords = reconstruct_coordinates(M, method=method)
        except Exception as e:
            print(f"Failed {method}: {e}")
            coords = _classical_mds_from_distance(M)
        
        ax = axes[idx]
        _plot_on_axis(ax, coords, M, title=method.replace('_', ' ').title(), k_edges=3)
    
    fig.suptitle(f"Embedding Method Comparison - Instance with {hard_instance['iteration']:.0f} iterations",
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "embedding_method_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved embedding comparison to {save_path}")
    
    return fig


# ---------- CLI entry points ---------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        # Run comprehensive analysis
        generate_comprehensive_analysis()
    else:
        # Run basic analysis (backward compatible)
        df = load_all_matrices()
        
        # Generate enhanced visualizations
        save_etsp_figures_enhanced(
            df,
            out_dir="../figures/ETSP-Enhanced",
            selection_type='both',
            fraction=0.05,
            embedding_method='classical_mds',
            show_heatmap=True,
            export_formats=['json', 'tsplib'],
            compute_metrics=True
        )
        
        # Create comparison
        create_difficulty_comparison(df)
=======
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
>>>>>>> a254e8852e74aa2d4f9baa764837b285f9ad6b3a
