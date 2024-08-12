import sys
import numpy as np
import dill
import matplotlib.pyplot as plt
import igraph as ig

chain_type_to_label = {
    # "": "Basic",
    # "REV MES": "Basic w/ REV & MES",
    # "REV": "Basic w/ REV",
    # "MES": "Basic w/ MES",
    "REV Partition": "Partition w/ REV",
    "REV MES Partition": "Partition w/ REV & MES",
}

chain_type_to_colour = {
    "": "green",
    "REV MES": "violet",
    "REV": "blue",
    "MES": "red",
    "REV Partition": "cyan",
    "REV MES Partition": "magenta",
}

data_counts = [100, 1000, 10000]


def calculate_errors(G: list[ig.Graph], exact):
    n = G[0].vcount()
    cumulative_matrix = np.zeros((n, n))
    sum_error = np.zeros(len(G))
    max_error = np.zeros(len(G))

    for i, graph in enumerate(G):
        cumulative_matrix += np.array(graph.get_adjacency().data)
        error = np.abs(exact - (cumulative_matrix / (i + 1)))
        sum_error[i] = np.sum(error)
        max_error[i] = np.max(error)

    return sum_error, max_error


score_name = sys.argv[1]
n = int(sys.argv[2])
num_runs = int(sys.argv[3])


# Plotting both metrics
fig, axes = plt.subplots(2, len(data_counts), figsize=(14, 10))

for j, m in enumerate(data_counts):
    exact = np.load(f"res/{score_name}-{m}/exact-edge-prob.npy")
    errors = {key: {"sum": [], "max": []} for key in chain_type_to_label.keys()}

    # Iterate over the runs
    for i in range(num_runs):
        for chain_type in errors:
            chain = np.load(
                f"res/{score_name}-{m}/chain-n={n}.{i}.{chain_type.lower()}.npy",
                allow_pickle=True,
            )
            chain = chain[len(chain) // 5 :]

            sum_errors, max_errors = calculate_errors(chain, exact)
            errors[chain_type]["sum"].append(sum_errors)
            errors[chain_type]["max"].append(max_errors)

    # Calculate median errors and plot for each metric
    for idx, metric in enumerate(["sum", "max"]):
        for chain_type in errors:
            errors_array = np.array(errors[chain_type][metric])
            median_errors = np.median(errors_array, axis=0)
            axes[idx][j].plot(
                np.arange(0, len(median_errors)),
                median_errors,
                "--" if "MES" in chain_type else "-",
                label=chain_type_to_label[chain_type] if idx == 0 else None,
                color=chain_type_to_colour[chain_type],
            )
        axes[idx][j].set_title(f"{score_name}-{m}" if idx == 0 else "")
        axes[idx][j].set_xlabel("Steps")
        axes[idx][j].set_ylabel(f"{metric.capitalize()} Abs Error")
        axes[idx][j].set_xlim(0, len(median_errors))
        axes[idx][j].set_ylim(bottom=0)

axes[0][0].legend()
plt.suptitle(f"{score_name} {num_runs} chains")
plt.show()
