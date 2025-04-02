from analysis_util.load_json import load_lon_data
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import matplotlib.pyplot as plt
import json

def build_lon(optima, transitions):
    G = nx.DiGraph()
    for opt in optima:
        G.add_node(opt["hash"], fitness=opt["iterations"])
    for src, dests in transitions.items():
        for dest, count in dests.items():
            G.add_edge(src, dest, weight=count)
    return G

# Step 2: Cluster similar optima based on cost differences
def cluster_similar_optima(lon_data, distance_threshold=0.05):
    optima_keys = list(lon_data["local_optima"].keys())
    costs = np.array([lon_data["local_optima"][key]["iterations"] for key in optima_keys])
    
    # Compute pairwise distances between optima
    cost_distances = squareform(pdist(costs.reshape(-1, 1)))
    
    clusters = {}  # Map each optima to a representative cluster center
    for i, key in enumerate(optima_keys):
        similar_optima = [optima_keys[j] for j in range(len(optima_keys)) if cost_distances[i, j] < distance_threshold]
        representative = min(similar_optima, key=lambda k: lon_data["local_optima"][k]["cost"])
        clusters[key] = representative
    
    # Update transitions to reflect clustering
    clustered_transitions = defaultdict(set)
    for source, targets in lon_data["filtered_transitions"].items():
        clustered_source = clusters[source]
        clustered_targets = {clusters[t] for t in targets}
        clustered_transitions[clustered_source].update(clustered_targets)
    
    lon_data["clustered_transitions"] = {k: list(v) for k, v in clustered_transitions.items()}
    return lon_data

# Step 3: Visualize the cleaned-up LON
def visualize_lon(lon_data):
    G = nx.DiGraph()
    
    for source, targets in lon_data["clustered_transitions"].items():
        for target in targets:
            G.add_edge(source, target)
    
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue', edge_color='gray')
    plt.title("Refined Local Optima Network")
    plt.show()

# Main function to run the refinement process
def refine_lon(file_path):
    local_optima, transitions = load_lon_data()
    lon_data = {
        "local_optima": local_optima,
        "filtered_transitions": transitions
    }
    lon_data = filter_transitions(lon_data)
    lon_data = cluster_similar_optima(lon_data)
    visualize_lon(lon_data)
    
    # Save the refined LON
    refined_file = file_path.replace(".json", "_refined.json")
    with open(refined_file, 'w') as f:
        json.dump(lon_data, f, indent=4)
    
    print(f"Refined LON saved to {refined_file}")

refine_lon("/Results/uniform_euclidean/city20_range5_wouter.json")