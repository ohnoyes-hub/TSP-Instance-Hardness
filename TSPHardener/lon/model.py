import networkx as nx
import matplotlib.pyplot as plt

from analysis.load_json import load_lon_data

def build_lon(optima, transitions):
    G = nx.DiGraph()
    for opt in optima:
        G.add_node(opt["hash"], fitness=opt["iterations"], matrix=opt["matrix"])
    for src, dests in transitions.items():
        for dest, count in dests.items():
            G.add_edge(src, dest, weight=count)
    return G

local_optima, transitions = load_lon_data()
# lon = build_lon(local_optima, transitions)

# draw the graph (basic)
def visualize_lon(local_optima, transitions):
    G = build_lon(local_optima, transitions)
    
    # Node attributes
    node_colors = [data["iterations"] for _, data in G.nodes(data=True)]
    node_sizes = [data["iterations"] / 10 for _, data in G.nodes(data=True)]
    
    # Layout
    pos = nx.spring_layout(G, k=0.2, seed=42)
    
    # Draw
    plt.figure(figsize=(14, 10))
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.viridis,
        with_labels=False,
        edge_color="gray",
        alpha=0.6,
    )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                             norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    plt.colorbar(sm, label="Iterations (Hardness)")
    
    plt.title("Local Optima Network (Colored by Instance Hardness)")
    plt.savefig("./plot/local_optima_network.png", bbox_inches='tight')
    plt.show()

visualize_lon(local_optima, transitions)