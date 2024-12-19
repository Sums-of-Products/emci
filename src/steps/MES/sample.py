import itertools
import random
import igraph as ig
import numpy as np

from .cpdag import CPDAG
from .count import C, FP, count, build_clique_graph, v_func
import matplotlib.pyplot as plt


def get_markov_equivalent_topological_orders(U: ig.Graph):
    # U is a the essential graph with only the undirected edges

    def get_topological_order(UCCG: ig.Graph):
        AMO = count(UCCG)

        maximal_cliques = [
            tuple(UCCG.vs["label"][c] for c in clique)
            for clique in UCCG.maximal_cliques()
        ]

        clique_tree = build_clique_graph(maximal_cliques)

        r = maximal_cliques[0]

        if len(maximal_cliques) == 1 and len(maximal_cliques[0]) == 1:
            return r

        # Maximal clique drawn with probability proportional to v_func
        p = list(map(lambda v: v_func(UCCG, r, v, clique_tree) / AMO, maximal_cliques))
        v = maximal_cliques[np.random.choice(np.arange(len(p)), p=p)]

        K = set(v)
        to = list(K)

        permutations = list(itertools.permutations(to))

        # uniformly drawn permutation of ι(v) without prefix in FP(v, T )
        FPs = FP(clique_tree, r, v)
        is_forbidden_to = True
        while is_forbidden_to:
            to = random.choice(permutations)

            is_start_with_fp = [np.array_equal(to[: len(fp)], fp) for fp in FPs]
            is_forbidden_to = any(is_start_with_fp)

        for H in C(UCCG, K):
            to += get_topological_order(H)

        return to

    # pre-process
    AMOs = count(U)

    # Gets the UCCGs from the essential graph
    UCCGs = [U.subgraph(component) for component in U.clusters()]
    UCCGs = list(filter(lambda UCCG: UCCG.vcount() > 1, UCCGs))

    tos = [get_topological_order(UCCG) for UCCG in UCCGs]

    return tos, AMOs


def MES(G: ig.Graph) -> ig.Graph:
    essential_g, _ = CPDAG(G)
    U = essential_g

    tos, AMOs = get_markov_equivalent_topological_orders(U)

    if len(tos) == 0:
        return G, AMOs

    equivalent_G: ig.Graph = G.copy()
    equivalent_G.delete_edges(equivalent_G.es)

    for to in tos:
        for e in U.es:
            source, target = e.source, e.target

            if source not in to or target not in to:
                continue
            if to.index(source) < to.index(target):
                equivalent_G.add_edge(source, target)
            else:
                equivalent_G.add_edge(target, source)

    for e in G.es:
        if not (
            equivalent_G.are_connected(e.source, e.target)
            or equivalent_G.are_connected(e.target, e.source)
        ):
            equivalent_G.add_edge(e.source, e.target)

    return equivalent_G, AMOs


def test_top_orders_distribution(G):
    # Gets many topological orders and plots a bar plot of it's occurrences

    essential_g, _ = CPDAG(G)
    U = essential_g

    all_tos = []
    for i in range(10000):
        tos, AMOs = get_markov_equivalent_topological_orders(U)
        all_tos.append(tos)
    all_tos = list(map(lambda tos: tuple(item for t in tos for item in t), all_tos))
    all_tos = list(map(lambda tos: "".join(str(item) for item in tos), all_tos))

    data = {}
    for to in all_tos:
        if to not in data:
            data[to] = 0
        data[to] += 1

    print(len(data))
    print(AMOs)
    plt.bar(list(data.keys()), list(data.values()))
    plt.show()
