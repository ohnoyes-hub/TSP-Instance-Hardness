import networkx as nx

def build_lon(optima, transitions):
    G = nx.DiGraph()
    for opt in optima:
        G.add_node(opt["hash"], fitness=opt["iterations"], matrix=opt["matrix"])
    for src, dests in transitions.items():
        for dest, count in dests.items():
            G.add_edge(src, dest, weight=count)
    return G