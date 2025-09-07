import types, sys, importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch

# --- Stub 'icecream.ic' for your generator import ---
stub = types.ModuleType("icecream")
def ic_stub(*args, **kwargs):
    return args if len(args) != 1 else args[0]
stub.ic = ic_stub
sys.modules["icecream"] = stub

# --- Load your generator script ---
MODULE_PATH = "core/generate_tsp.py"  # adjust if needed
spec = importlib.util.spec_from_file_location("generate_tsp", MODULE_PATH)
gen = importlib.util.module_from_spec(spec)
sys.modules["generate_tsp"] = gen
spec.loader.exec_module(gen)

# ---------------- Params ----------------
CITY_SIZE    = 8
DISTRIBUTION = "uniform"   # or "lognormal"
CONTROL      = 20
DIMENSIONS   = 2

# ---------------- Build ATSP ----------------
builder = (gen.TSPBuilder()
           .set_city_size(CITY_SIZE)
           .set_generation_type("asymmetric")
           .set_distribution(DISTRIBUTION)
           .set_control(CONTROL)
           .set_dimensions(DIMENSIONS))
instance = builder.build()

mat = instance.matrix.astype(float)
np.fill_diagonal(mat, np.inf)
n = mat.shape[0]
labels = list(range(n))

# ---------------- Print matrix ----------------
df = pd.DataFrame(mat, index=labels, columns=labels)
print("ATSP Distance Matrix (∞ on diagonal):")
print(df.to_string())
print(mat)

# ---------------- Graph prep ----------------
G = nx.DiGraph()
G.add_nodes_from(labels)
finite = np.isfinite

for i in range(n):
    for j in range(n):
        if i != j and finite(mat[i, j]):
            G.add_edge(i, j, weight=float(mat[i, j]))

pos = nx.circular_layout(G)

# Utility: draw a single curved edge (optionally arrowed)
def draw_curve(ax, p0, p1, rad=0.20, arrow=True, lw=1.8, alpha=1.0, linestyle='-'):
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
        shrinkA=12, shrinkB=12  # keep arrowheads off the node centers
    )
    if not arrow and linestyle != '-':
        patch.set_linestyle(linestyle)
    ax.add_patch(patch)

# ---------------- Draw ----------------
fig, ax = plt.subplots(figsize=(7.2, 7.2))
# ax.set_title("ATSP Directed Graph (solid arrow = smaller weight; dashed no-arrow = reverse)")

# Nodes & labels
nx.draw_networkx_nodes(G, pos, node_size=650, node_color="white", edgecolors="black", ax=ax)
nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

# For each unordered pair {u,v} with at least one direction, show:
# - the smaller-weight direction as a solid curved arrow
# - the reverse (if exists) as a thin dashed curve with NO arrowhead
visited = set()
for u, v in G.edges():
    if (v, u) in visited:
        continue
    if (u, v) in visited:
        continue
    visited.add((u, v))
    if (v, u) in G.edges():
        w_uv = G[u][v]['weight']
        w_vu = G[v][u]['weight']
        # choose smaller as dominant (arrowed)
        if w_uv <= w_vu:
            # u->v arrow (solid), v->u dashed no-arrow
            draw_curve(ax, pos[u], pos[v], rad=0.22, arrow=True,  lw=2.2, alpha=1.0, linestyle='-')
            draw_curve(ax, pos[v], pos[u], rad=-0.22, arrow=False, lw=1.2, alpha=0.6, linestyle='--')
        else:
            draw_curve(ax, pos[v], pos[u], rad=0.22, arrow=True,  lw=2.2, alpha=1.0, linestyle='-')
            draw_curve(ax, pos[u], pos[v], rad=-0.22, arrow=False, lw=1.2, alpha=0.6, linestyle='--')
    else:
        # only one direction exists: draw a single solid arrow
        draw_curve(ax, pos[u], pos[v], rad=0.0, arrow=True, lw=2.2, alpha=1.0, linestyle='-')

# Edge labels: place both weights if both directions exist, offsetting slightly
def mid_point(p0, p1, t=0.5):
    return (p0[0]*(1-t)+p1[0]*t, p0[1]*(1-t)+p1[1]*t)

for u, v in G.edges():
    w = int(G[u][v]['weight'])
    p0, p1 = pos[u], pos[v]
    m = mid_point(p0, p1, 0.55 if u < v else 0.45)
    ax.text(m[0], m[1], str(w), fontsize=8, bbox=dict(facecolor='white', edgecolor='none', pad=0.5))

ax.set_axis_off()
plt.tight_layout()
plt.show()
