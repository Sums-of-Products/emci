import sumu

# A path to a space separated file of either discrete or continuous
# data. No header rows for variable names or arities (in the discrete
# case) are assumed. Discrete data is assumed to be integer encoded;
# continuous data uses "." as decimal separator.
data = sumu.Data("./data/data/alarm-500.dat")

# print(data)
dags, scores = sumu.Gadget(data=data).sample()

# # Causal effect computations only for continuous data.
# # dags are first converted to adjacency matrices.
# dags = [sumu.bnet.family_sequence_to_adj_mat(dag) for dag in dags]

# # All pairwise causal effects for each sampled DAG.
# # causal_effects[i] : effects for ith DAG,
# # where the the first n-1 values represent the effects from variable 1 to 2, ..., n,
# # the following n-1 values represent the effects from variable 2 to 1, 3, ..., n, etc.
# causal_effects = sumu.beeps(dags, data)