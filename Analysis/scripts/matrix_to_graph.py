import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from util.load_experiment import load_all_matrices
import pandas as pd
import os
import math
import matplotlib
matplotlib.use("Agg")

def is_symmetric_matrix(M, atol=1e-9):
    if isinstance(M, list):
        M = np.array(M, dtype=float)
    # treat inf on diagonal as symmetric by ignoring diag
    if M.shape[0] != M.shape[1]:
        return False
    A = M.copy().astype(float)
    np.fill_diagonal(A, 0.0)
    B = A.T.copy()
    return np.allclose(A, B, atol=atol, equal_nan=True)

def _draw_curve(ax, p0, p1, rad=0.20, arrow=True, lw=1.8, alpha=1.0, linestyle='-'):
    style = f"arc3,rad={rad}"
    arrowstyle = '-|>' if arrow else '-'
    patch = FancyArrowPatch(
        p0, p1,
        connectionstyle=style,
        arrowstyle=arrowstyle,
        mutation_scale=14 if arrow else 1,
        linewidth=lw,
        alpha=alpha,
        color='k',
        shrinkA=12, shrinkB=12
    )
    if not arrow and linestyle != '-':
        patch.set_linestyle(linestyle)
    ax.add_patch(patch)

def build_graph_from_matrix(M, symmetric=None):
    """
    Returns (G, labels) where:
      - G is nx.Graph (if symmetric) or nx.DiGraph (if asymmetric)
      - node labels are 0..n-1
    """
    M = np.array(M, dtype=float)
    n = M.shape[0]
    labels = list(range(n))
    finite = np.isfinite

    if symmetric is None:
        symmetric = is_symmetric_matrix(M)

    if symmetric:
        G = nx.Graph()
        G.add_nodes_from(labels)
        for i in range(n):
            for j in range(i+1, n):
                if finite(M[i, j]) and finite(M[j, i]):
                    # if asymmetric noise on a symmetric matrix, choose the average
                    w = float((M[i, j] + M[j, i]) / 2.0)
                    G.add_edge(i, j, weight=w)
    else:
        G = nx.DiGraph()
        G.add_nodes_from(labels)
        for i in range(n):
            for j in range(n):
                if i != j and finite(M[i, j]):
                    G.add_edge(i, j, weight=float(M[i, j]))

    return G, labels

def draw_graph_from_matrix(M, labels=None, layout='circular', title=None, save_path=None, symmetric=None, seed=42):
    """
    Visualize ETSP-like (undirected) or ATSP-like (directed) graph from matrix M.
    - Undirected (symmetric M): straight edges with weight labels.
    - Directed (asymmetric M): curved dominant arrow; reverse direction dashed (no arrow).
    """
    G, default_labels = build_graph_from_matrix(M, symmetric=symmetric)
    labels = default_labels if labels is None else labels

    if layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'spring':
        pos = nx.spring_layout(G, seed=seed)
    else:
        pos = nx.circular_layout(G)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=650, node_color="white", edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    if isinstance(G, nx.Graph):
        # ETSP-style (undirected)
        nx.draw_networkx_edges(G, pos, ax=ax)
        # weight labels
        edge_labels = {(u, v): int(w['weight']) for u, v, w in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    else:
        # ATSP-style (directed) — mirror your existing curved/dashed treatment
        visited = set()
        for u, v in G.edges():
            if (v, u) in visited or (u, v) in visited:
                continue
            visited.add((u, v))
            has_rev = (v, u) in G.edges()
            if has_rev:
                w_uv = G[u][v]['weight']
                w_vu = G[v][u]['weight']
                if w_uv <= w_vu:
                    _draw_curve(ax, pos[u], pos[v], rad=0.22, arrow=True,  lw=2.2, alpha=1.0, linestyle='-')
                    _draw_curve(ax, pos[v], pos[u], rad=-0.22, arrow=False, lw=1.2, alpha=0.6, linestyle='--')
                else:
                    _draw_curve(ax, pos[v], pos[u], rad=0.22, arrow=True,  lw=2.2, alpha=1.0, linestyle='-')
                    _draw_curve(ax, pos[u], pos[v], rad=-0.22, arrow=False, lw=1.2, alpha=0.6, linestyle='--')
            else:
                _draw_curve(ax, pos[u], pos[v], rad=0.0, arrow=True, lw=2.2, alpha=1.0, linestyle='-')

        # place integer weights roughly halfway along straight chord (like your ATSP script)
        def mid_point(p0, p1, t=0.5):
            return (p0[0]*(1-t)+p1[0]*t, p0[1]*(1-t)+p1[1]*t)
        for u, v in G.edges():
            w = int(G[u][v]['weight'])
            p0, p1 = pos[u], pos[v]
            m = mid_point(p0, p1, 0.55 if u < v else 0.45)
            ax.text(m[0], m[1], str(w), fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.5))

    if title:
        ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

    return G, pos

# Make the graphs for the 5% easiest and 5% hardest distance matrices
# Hardness = iteration
def _safe_int(x):
    try:
        xi = int(float(x))
        return xi
    except Exception:
        return None

def _slug(x):
    # compact, filesystem-safe tag
    return str(x).replace(" ", "").replace("/", "_").replace("\\", "_")

def select_easy_hard(df, pct=5):
    """
    Returns (easy_df, hard_df) selecting the bottom/top pct% by 'iteration'.
    Rows with missing iteration are dropped.
    """
    sdf = df.copy()
    # keep only rows with a numeric iteration
    sdf = sdf[pd.to_numeric(sdf['iteration'], errors='coerce').notna()].copy()
    sdf['iteration'] = sdf['iteration'].astype(float)
    if len(sdf) == 0:
        return sdf.iloc[0:0], sdf.iloc[0:0]

    n = len(sdf)
    k = max(1, math.floor(n * (pct / 100.0)))

    sdf = sdf.sort_values('iteration')
    easy = sdf.head(k)
    hard = sdf.tail(k)
    return easy, hard

def ensure_dirs():
    os.makedirs("./graphs/easy", exist_ok=True)
    os.makedirs("./graphs/hard", exist_ok=True)

def make_title(row):
    # Short, readable title for the plot
    parts = [
        f"{row.get('distribution','?')}",
        f"{row.get('generation_type','?')}",
        f"n={row.get('city_size','?')}",
        f"range={row.get('range','?')}",
        f"mut={row.get('mutation_type','?')}",
        f"gen={row.get('generation','?')}",
        f"iter={row.get('iteration','?')}",
    ]
    return " | ".join(parts)

def make_filename(row):
    # Deterministic filename with key config bits
    bits = [
        _slug(row.get('distribution','?')),
        _slug(row.get('generation_type','?')),
        f"n{_slug(row.get('city_size','?'))}",
        f"r{_slug(row.get('range','?'))}",
        f"m{_slug(row.get('mutation_type','?'))}",
        f"g{_slug(row.get('generation','?'))}",
        f"it{_slug(row.get('iteration','?'))}",
    ]
    return "_".join(bits) + ".png"

def save_set(rows_df, outdir, layout='circular', seed=42):
    for _, row in rows_df.iterrows():
        M = row['matrix']
        title = make_title(row)
        fname = make_filename(row)
        path = os.path.join(outdir, fname)

        # Let your symmetry detector decide ETSP vs ATSP
        symmetric = is_symmetric_matrix(M)
        draw_graph_from_matrix(
            M,
            layout=layout,
            title=title,
            save_path=path,
            symmetric=symmetric,
            seed=seed
        )

def save_easy_and_hard_graphs(percent=5, layout='circular', seed=42):
    """
    End-to-end: load, pick bottom/top percent by iteration, and save PNGs.
    """
    df = load_all_matrices()  # provides matrix + config + generation + iteration
    ensure_dirs()
    easy, hard = select_easy_hard(df, pct=percent)
    if len(easy) == 0 and len(hard) == 0:
        print("No matrices with valid 'iteration' to rank.")
        return

    # print(f"Saving {len(easy)} easy graphs to ./graphs/easy and {len(hard)} hard graphs to ./graphs/hard")
    # save_set(easy, "./graphs/easy", layout=layout, seed=seed)
    print(f"Saving {len(hard)} hard graphs to ./graphs/hard")
    save_set(hard, "./graphs/hard", layout=layout, seed=seed)

if __name__ == "__main__":
    # Change percent or layout if you like
    save_easy_and_hard_graphs(percent=0.5, layout='circular')