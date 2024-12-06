from math import floor
import sys
import numpy as np
import dill
import matplotlib.pyplot as plt
import igraph as ig

# Define the labels and colors for different chain types
chain_type_to_label = {
    "MES REV": "Basic w/ REV & MES",
    # "REV": "Basic w/ REV",
    # "REV Partition": "Partition w/ REV",
    # "REV MES Partition": "Partition w/ REV & MES",
}

chain_type_to_colour = {
    "MES REV": "violet",
    # "REV": "blue",
    # "REV Partition": "cyan",
    # "REV MES Partition": "magenta",
}

mes_probs = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 0.99]


def calculate_errors(G: list[ig.Graph], exact):
    n = G[0].vcount()  # Number of vertices in the graph
    cumulative_matrix = np.zeros((n, n))
    max_error = np.zeros(len(G))

    for i, graph in enumerate(G):
        cumulative_matrix += np.array(graph.get_adjacency().data)
        error = np.abs(exact - (cumulative_matrix / (i + 1)))
        max_error[i] = np.max(error)

    return max_error


# Get input arguments from the command line
score_name = sys.argv[1]
n = int(sys.argv[2])
num_runs = int(sys.argv[3])

# fig, axes = plt.subplots(3, 3)

# Iterate over the mes_probs list
for row_idx, p_m in enumerate(mes_probs):
    exact = np.load(f"res/{score_name}/exact-edge-prob.npy")
    errors = {key: {"sum": [], "max": []} for key in chain_type_to_label.keys()}

    # Iterate over the runs
    for i in range(num_runs):
        for chain_type in errors:
            chain = np.load(
                f"res/{score_name}/chain-n={n}.{i}.{chain_type.lower()}.p_m={p_m}.npy",
                allow_pickle=True,
            )
            chain = chain[len(chain) // 5 :]  # Remove the burn-in period

            max_errors = calculate_errors(chain, exact)
            errors[chain_type]["max"].append(max_errors)

    for chain_type in errors:
        errors_array = np.array(errors[chain_type]["max"])
        median_errors = np.median(errors_array, axis=0)
        row = floor(row_idx / 3)
        col = row_idx % 3
        plt.plot(
            np.arange(0, len(median_errors)),
            median_errors,
            # "--" if "MES" in chain_type else "-",
            label=(f"P(MES)={p_m}"),
        )
        # axes[row][col].set_title(f"P(MES)={p_m}")
        # axes[row][col].set_ylabel(f"Max Abs Error")
        # axes[row][col].set_xlim(0, len(median_errors))
        # axes[row][col].set_ylim(bottom=0)

plt.legend()
plt.title(f"{score_name} {num_runs} chains for different p_m values")
# plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
