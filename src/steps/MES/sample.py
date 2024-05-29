import itertools
import random
import igraph as ig
import numpy as np
from .count import C, FP, count, build_clique_graph, v_func
import matplotlib.pyplot as plt
import copy


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

    # if not hasattr(G.vs, "labels"):
    #     G.vs["label"] = G.vs.indices.copy()

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


def is_strongly_protected(G: ig.Graph, G_lines: ig.Graph, e: ig.Edge):
    a, b = e.source, e.target

    # a
    a_parents = list(G.predecessors(a))
    for c in a_parents:
        if not G.are_connected(c, b) and not G.are_connected(b, c):
            return True

    # b
    b_parents = list(G.predecessors(b))
    for c in b_parents:
        if c != a and not G.are_connected(c, a) and not G.are_connected(a, c):
            return True

    # c
    b_parents = list(G.predecessors(b))
    for c in b_parents:
        if c != a and G.are_connected(a, c):
            return True
    # d

    a_neighbors = list(G.neighbors(a))
    for c1, c2 in itertools.combinations(a_neighbors, 2):
        if G.are_connected(c1, b) and G.are_connected(c2, b):
            return True

    if len(G_lines.es) == 0:
        return False
    a_lines_neighbors = list(G_lines.neighbors(a))
    for c1, c2 in itertools.combinations(a_lines_neighbors, 2):
        if G.are_connected(c1, b) and G.are_connected(c2, b):
            return True

    return False


# Returns a tuple (undirected, directed) graphs
def CPDAG(D: ig.Graph) -> tuple[ig.Graph, ig.Graph]:
    G_i: ig.Graph = D.copy()
    G_lines = D.copy()
    G_lines.to_undirected()
    G_lines.delete_edges(G_lines.es)

    G_i_plus_1 = undirect_non_strongly_protected_arrows(G_i, G_lines)

    while len(G_i.es) != len(G_i_plus_1.es):
        G_i = G_i_plus_1.copy()
        G_i_plus_1 = undirect_non_strongly_protected_arrows(G_i, G_lines)

    return G_lines, G_i_plus_1


def undirect_non_strongly_protected_arrows(G: ig.Graph, G_lines: ig.Graph) -> ig.Graph:
    new_G: ig.Graph() = G.copy()

    for e in G.es:
        if not is_strongly_protected(G, G_lines, e):
            G_lines.add_edge(e.source, e.target)
            new_G.delete_edges([(e.source, e.target)])

    return new_G
