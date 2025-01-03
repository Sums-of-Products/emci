import random
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import hashlib
import igraph as ig
import os


def calculate_edge_ratios(G: list[ig.Graph]):
    G = G[len(G) // 2 :]

    n = G[0].vcount()
    cumulative_matrix = np.zeros((n, n))

    for graph in G:
        cumulative_matrix += np.array(graph.get_adjacency().data)

    ratio_matrix = cumulative_matrix / len(G)

    return ratio_matrix


def read_graph_from_file(filename, random_weights=False):

    # file = open('./sample.gr', 'r')
    file = open(filename, "r")
    G = nx.Graph()
    lines = [tuple(map(int, line.strip().split(" "))) for line in file.readlines()]
    nodes_count = lines[0][0]
    edges = lines[1:]

    nodes = np.arange(1, nodes_count + 1)
    G.add_nodes_from(nodes)
    for edge in edges:
        weight = np.round(np.random.normal(), 2) if random_weights else 1

        G.add_edge(edge[0], edge[1], weight=weight)
    G.add_edges_from(edges)

    return G


def read_scores_from_file(filename):
    scores = {}

    with open(filename, "r") as file:
        n = int(next(file).strip())

        for _ in range(n):
            v, j_count = map(int, next(file).strip().split(" ")[:2])
            scores[v] = {}

            for _ in range(j_count):
                line = next(file).strip().split(" ")
                scores[v][frozenset(map(int, line[2:]))] = float(line[0])

    return scores


def calculate_and_save_edge_ratios(
    G_samples: list[ig.Graph], score_name: str, n: int, variation: list[str], i: int
):
    """
    Calculate edge ratios from graph samples and save the results to a file.

    Args:
    G_samples (list): List of graph samples from MCMC simulation.
    score_name (str): The name of the score function used.
    n (int): The sample size parameter.
    variation (list): List describing the variation in the simulation, e.g., ['rev'].
    """
    edge_ratios = calculate_edge_ratios(G_samples)

    directory_path = f"res/{score_name}"
    os.makedirs(directory_path, exist_ok=True)

    variation_desc = " ".join(map(str, variation)) if variation else "basic"
    file_path = f"{directory_path}/edge-ratios-n={n}.{i}.{variation_desc}.npy"

    np.save(file_path, edge_ratios)

    return edge_ratios, variation_desc


def get_graph_hash(G: ig.Graph) -> str:
    sorted_edges = sorted(tuple(G.es))
    edges_str = str(sorted_edges)
    hash_object = hashlib.sha256()
    hash_object.update(edges_str.encode())

    return hash_object.hexdigest()


# Get the default color cycle from Matplotlib
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

counter = 0
# Function to get the next color from the default color cycle


def get_next_color():
    global counter
    color = default_colors[counter % len(default_colors)]
    counter += 1
    return color


# Works only for undirected
# Edit: I think should work for both


def get_graph_hash_ig(G: ig.Graph) -> str:
    sorted_edges = sorted([tuple(sorted(edge)) for edge in G.get_edgelist()])
    edges_str = str(sorted_edges)
    hash_object = hashlib.sha256()
    hash_object.update(edges_str.encode())

    return hash_object.hexdigest()


memo = {}


def memo_by_graph(G: nx.DiGraph, key: str, value):
    if key not in memo:
        memo[key] = {}


def plot(G, title=""):
    _, ax = plt.subplots()
    visual_style = {}
    visual_style["vertex_label"] = list(map(lambda x: x, G.vs.indices))
    ig.plot(G, target=ax, **visual_style)
    plt.show()


def get_es_diff(G1: ig.Graph, G2: ig.Graph):
    G1_set = {e.tuple for e in G1.es}
    G2_set = {e.tuple for e in G2.es}

    both = G1_set.intersection(G2_set)
    return (G1_set - both).union(G2_set - both)


def seed(s):
    np.random.seed(s * 1 + 1)
    random.seed(s * 2 + 3)
