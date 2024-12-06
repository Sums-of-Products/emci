import numpy as np

post_rec = np.load("res/asia/n=100000.partition.npy")

post_jax = np.load("data/jax-res/posterior.npy")


def calculate_edge_ratios(post):
    n = len(post[0][0])
    cumulative_matrix = np.zeros((n, n))

    for graph in post:
        cumulative_matrix += np.array(graph)

    ratio_matrix = cumulative_matrix / len(post)

    return np.round(ratio_matrix, decimals=2)


print(post_rec)
print(calculate_edge_ratios(post_jax[7000:]))
