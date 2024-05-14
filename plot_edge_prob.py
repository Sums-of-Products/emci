import numpy as np
import dill
import matplotlib.pyplot as plt
import igraph as ig

chain_type_to_label = {
    "REV MES": "Structural w/ REV & MES",
    "REV": "Structural w/ REV",
    "REV Partition": "Partition w/ REV",
    "REV MES Partition": "Partition w/ REV & MES",
}


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


file_name = "res/asia10k3/asia_sparse.dil"
g = globals()

with open(file_name, "rb") as file:
    list_of_variable_names = dill.load(file)  # Get the names of stored objects
    for variable_name in list_of_variable_names:
        g[variable_name] = dill.load(file)  # Get the objects themselves

score_name = "asia-10000"
# exact = np.load(f"res/{score_name}/exact-edge-prob.npy")
exact = g["asia_exact_sparse"]
n = 100000
num_runs = 9

errors = {key: {"sum": [], "max": []} for key in chain_type_to_label.keys()}

# Iterate over the runs
for i in range(num_runs):
    for chain_type in errors:
        chain = np.load(
            f"res/{score_name}/chain-n={n}.{i}.{chain_type.lower()}.npy",
            allow_pickle=True,
        )
        chain = chain[len(chain) // 10 :]

        sum_errors, max_errors = calculate_errors(chain, exact)
        errors[chain_type]["sum"].append(sum_errors)
        errors[chain_type]["max"].append(max_errors)

# Plotting both metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Calculate median errors and plot for each metric
for idx, metric in enumerate(["sum", "max"]):
    for chain_type in errors:
        errors_array = np.array(errors[chain_type][metric])
        median_errors = np.mean(errors_array, axis=0)
        axes[idx].plot(
            np.arange(0, len(median_errors)),
            median_errors,
            "--" if "MES" in chain_type else "-",
            label=chain_type_to_label[chain_type] if idx == 0 else None,
        )
    axes[idx].set_title(f"{metric.capitalize()} of Absolute Errors")
    axes[idx].set_xlabel("Chain")
    axes[idx].set_ylabel(f"{metric.capitalize()} Absolute Error")
    axes[idx].set_xlim(0, len(median_errors))
    axes[idx].set_ylim(bottom=0)

axes[0].legend()
plt.suptitle(f"{score_name} {num_runs} chains")
plt.show()
