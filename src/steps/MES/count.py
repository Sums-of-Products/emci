import igraph as ig
import random
from functools import reduce
from operator import mul
import numpy as np
from src.utils import get_graph_hash

v_func_memo = {}


def v_func(G, r, v, clique_tree):
    # try:
    #     res = v_func_memo[frozenset(v)]
    #     return res
    # except KeyError:
    #     pass

    K = set(v)
    subproblems = C(G, K)

    results = [count(H) for H in subproblems]
    prod = reduce(mul, results) if len(results) > 0 else 1

    fps = FP(clique_tree, r, v)

    fp_lens = [len(fp) for fp in fps]
    fp_lens.insert(0, 0)

    res = phi(len(set(v)), 0, fp_lens, {}) * prod
    v_func_memo[frozenset(v)] = res

    return res


memo = {}


def count(G: ig.Graph, pool=None):
    # G is a UCCG

    # if not hasattr(G.vs, "labels"):
    #     G.vs["label"] = G.vs.indices.copy()

    # G_hash = get_graph_hash(G)

    # try:
    #     res = memo[G_hash]
    #     return res
    # except KeyError:
    #     pass

    # Get connected components of the graph
    G_subs = [
        G.induced_subgraph(component, implementation="copy_and_delete")
        for component in G.clusters()
    ]
    results = []

    # For each subgraph, count the AMOs and return the product
    for G_sub in G_subs:
        result = 0

        # cliques by label
        maximal_cliques = [
            tuple(G_sub.vs["label"][c] for c in clique)
            for clique in G_sub.maximal_cliques()
        ]

        clique_tree = build_clique_graph(maximal_cliques)

        r = maximal_cliques[0]

        for v in maximal_cliques:
            result += v_func(G_sub, r, v, clique_tree)

        results.append(result)

    # Product of each component
    result = reduce(lambda x, y: x * y, results, 1)
    # memo[G_hash] = result

    return result


def FP(T, r, v):
    res = []
    r = next((vertex for vertex in T.vs if vertex["label"] == r), None)
    v = next((vertex for vertex in T.vs if vertex["label"] == v), None)
    path = list(T.shortest_paths(r, v))[0]
    p = len(path)

    for i in range(0, p - 1):
        intersection = {value for value in path[i] if value in path[i + 1]}
        if intersection.issubset(set(v)):
            res.append(intersection)

    # Converted to set for uniqness and sorted
    res = set(tuple(sorted(s)) for s in res)

    return list(res)


fmemo = {}


def build_clique_graph(cliques: list[set]):
    clique_graph = ig.Graph()
    clique_graph.add_vertices(len(cliques))
    clique_graph.vs["label"] = cliques.copy()

    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            if set(cliques[i]).intersection(
                cliques[j]
            ):  # If cliques share at least one node
                clique_graph.add_edges([(i, j)])

    return clique_graph


def fac(n):
    if n in fmemo:
        return fmemo[n]

    if n <= 0:
        return 0
    if n == 1:
        return 1

    res = fac(n - 1) * n
    fmemo[n] = res
    return res


def phi(cliquesize, i, fp, pmemo):
    # pmemo is for the recursive nature of this function and should be empty for
    # each iteration
    if i in pmemo:
        return pmemo[i]

    sum = fac(cliquesize - fp[i])
    for j in range(i + 1, len(fp)):
        sum -= fac(fp[j] - fp[i]) * phi(cliquesize, j, fp, pmemo)
    pmemo[i] = sum
    return sum


def C(G: ig.Graph, K: set):

    # C_G(K) - algorithm 4
    S = [K.copy(), set(G.vs["label"]) - K]

    to = []
    L = set()
    output = []

    while S:
        X = next(s for s in S if len(s) != 0)
        if X is None:
            break

        v = random.choice(list(X))
        to.append(v)

        if not any(v in el for el in L) and (v not in K):
            L.add(frozenset(X))

            vertex_indices = [v.index for v in G.vs if v["label"] in X]
            subgraphs = []
            master_subgraph = G.subgraph(vertex_indices)
            for component in master_subgraph.clusters():
                subgraph = master_subgraph.subgraph(component)
                subgraphs.append(subgraph)

            # Output the undirected components of G[X].
            output.extend(subgraphs)

        X.remove(v)
        S_new = []
        vertex_index = next((node.index for node in G.vs if node["label"] == v), None)

        neighbors_v = set([G.vs["label"][v] for v in G.neighbors(vertex_index)])

        for Si in S:
            S_new.append(Si & neighbors_v)
            S_new.append(Si - neighbors_v)

        S = [Si for Si in S_new if Si]

    return output


def maximal_clique_tree(G: ig.Graph):
    clique_tree = ig.Graph()
    maximal_cliques = list(map(lambda clique: tuple(sorted(clique)), G.cliques))

    clique_tree.add_nodes_from(maximal_cliques)

    # Builds graph out of maximal cliques by content intersection
    for clique1 in maximal_cliques:
        for clique2 in maximal_cliques:
            if clique1 != clique2 and len(set(clique1).intersection(clique2)) > 0:
                # if(len([edge for edge in clique_tree.edges() if clique1 in edge]) == 0):
                clique_tree.add_edge(clique1, clique2)

    # clique_tree = nx.maximum_spanning_tree(clique_tree, algorithm="kruskal")
    return clique_tree


def get_maximal_cliques(cliques: ig.Graph):
    # Get maximal cliques out of a clique tree that includes minimal seperators

    maximal_cliques = []
    for clique1 in cliques:
        is_maximal = True
        for clique2 in cliques:
            if clique1 != clique2 and set(clique1).issubset(clique2):
                is_maximal = False
        if is_maximal:
            maximal_cliques.append(clique1)

    return maximal_cliques
