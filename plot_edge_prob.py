import numpy as np
import dill
import matplotlib.pyplot as plt
import igraph as ig


def calculate_errors(G: list[ig.Graph], exact):
    n = G[0].vcount()
    cumulative_matrix = np.zeros((n, n))
    sum_error = np.zeros(len(G))

    for i, graph in enumerate(G):
        cumulative_matrix += np.array(graph.get_adjacency().data)
        sum_error[i] = np.sum(np.abs(exact - (cumulative_matrix / (i + 1))))

    return sum_error


file_name = "res/asia10k3/asia_sparse.dil"
g = globals()

with open(file_name, "rb") as file:
    list_of_variable_names = dill.load(file)  # Get the names of stored objects
    for variable_name in list_of_variable_names:
        g[variable_name] = dill.load(file)  # Get the objects themselves


score_name = "asia10k3"
# exact = np.load(f"res/{score_name}/exact-edge-prob.npy")
exact = g["asia_exact_sparse"]
n = 100000
num_runs = 7


errors = {"REV MES": [], "REV": [], "REV Partition": [], "REV MES Partition": []}

# Iterate over the runs
for i in range(num_runs):
    for chain_type in errors:
        chain = np.load(
            f"res/{score_name}/chain-n={n}.{i}.{chain_type.lower()}.npy",
            allow_pickle=True,
        )[1000:]

        # Calculate errors for the current run and chain
        current_errors = calculate_errors(chain, exact)

        # Append the calculated errors to the respective list in the dictionary
        if len(errors[chain_type]) == 0:
            errors[chain_type] = current_errors
        else:
            errors[chain_type] += current_errors

for chain_type in errors:
    errors[chain_type] /= num_runs

    plt.plot(
        np.arange(0, len(errors[chain_type])),
        errors[chain_type],
        "--" if "MES" in chain_type else "-",
        label=chain_type,
    )

plt.xlim(1000, n)
plt.legend()
plt.xlabel("Index of Graphs")
plt.ylabel("Sum Absolute Error")
plt.title(f"{score_name} {num_runs} runs")
plt.show()
