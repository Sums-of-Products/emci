import igraph as ig


def CPDAG(D: ig.Graph) -> tuple[ig.Graph, ig.Graph]:
    """
    Returns a tuple (undirected, directed) graphs
    forming together the CPDAG of G
    """
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
    new_G: ig.Graph = G.copy()

    for e in G.es:
        if not is_strongly_protected(G, G_lines, e):
            G_lines.add_edge(e.source, e.target)
            new_G.delete_edges([(e.source, e.target)])

    return new_G


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
