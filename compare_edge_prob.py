import numpy as np
import dill
import igraph as ig


def calculate_edge_ratios(G: list[ig.Graph]):
    G = G[len(G) // 2 :]

    n = G[0].vcount()
    cumulative_matrix = np.zeros((n, n))

    for graph in G:
        cumulative_matrix += np.array(graph.get_adjacency().data)

    ratio_matrix = cumulative_matrix / len(G)

    return ratio_matrix


file_name = "res/asia10k3/asia_sparse.dil"
g = globals()

with open(file_name, "rb") as file:
    list_of_variable_names = dill.load(file)
    for variable_name in list_of_variable_names:
        g[variable_name] = dill.load(file)


score_name = "sachs-1000"
n = 20000
i = 0


# exact = g["asia_exact_sparse"]
exact = np.load(f"res/{score_name}/exact-edge-prob.npy")

rev = np.load(f"res/{score_name}/edge-ratios-n={n}.0.rev.npy", allow_pickle=True)
part_rev_mes = np.load(
    f"res/{score_name}/edge-ratios-n={n}.0.rev mes partition.npy", allow_pickle=True
)
part_rev = np.load(
    f"res/{score_name}/edge-ratios-n={n}.0.rev partition.npy", allow_pickle=True
)


print(f"Structural w/ REV: {np.sum(np.abs(exact - rev))}")
print(f"Partition w/ REV: {np.sum(np.abs(exact - part_rev))}")
print(f"Partition w/ REV, MES: {np.sum(np.abs(exact - part_rev_mes))}")
