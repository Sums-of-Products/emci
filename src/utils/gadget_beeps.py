import argparse
import numpy as np

import sumu

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("datapath", help="path to data file")
parser.add_argument("K", help="how many candidate parents to include", type=int)
parser.add_argument("-c", "--candidate-parent-algorithm", help="candidate algorithm to use (default: greedy-lite)", choices=sumu.candidate_parent_algorithm.keys(), default="greedy-lite")
parser.add_argument("-s", "--score", help="score function to use", choices=["bdeu", "bge"], default="bdeu")
parser.add_argument("-e", "--ess", help="equivalent sample size for BDeu", type=int, default=10)
parser.add_argument("-m", "--max-id", help="maximum indegree for scores (default: no max-indegree)", type=int, default=-1)
parser.add_argument("-d", help="maximum indegree for psets which are not subsets of candidates (default: 2)", type=int, default=2)
parser.add_argument("-b", "--burn-in", help="number of burn-in samples (default: 1000)", type=int, default=1000)
parser.add_argument("-i", "--iterations", help="number of iterations after burn-in (default: 1000)", type=int, default=1000)
parser.add_argument("-n", "--nth", help="sample dag every nth iteration (default: 10)", type=int, default=10)
parser.add_argument("-nc", "--n-chains", help="number of Metropolis coupled MCMC chains (default: 16)", type=int, default=16)
parser.add_argument("-r", "--randomseed", help="random seed", type=int)
parser.add_argument("-o", "--output-prefix", help="path prefix for output files (default: input file path)", default=None)
args = parser.parse_args()
if args.output_prefix is None:
    args.output_prefix = args.datapath

if args.randomseed is not None:
    np.random.seed(args.randomseed)

data = sumu.Data(args.datapath, discrete=args.score == "bdeu")

params = {"data": data,
          "scoref": args.score,
          "ess": args.ess,
          "max_id": args.max_id,
          "K": args.K,
          "d": args.d,
          "cp_algo": args.candidate_parent_algorithm,
          "mc3_chains": args.n_chains,
          "burn_in": args.burn_in,
          "iterations": args.iterations,
          "thinning": args.nth,
          }

g = sumu.Gadget(**params)
dags, scores = g.sample()
candidates = g.C_array

candidates_header = "K = {} candidate parents for each node, computed with the algorithm {}.\n".format(args.K, args.candidate_parent_algorithm)
candidates_header += "Rows represent the nodes in ascending order."
np.savetxt(args.output_prefix + ".candidates", candidates, fmt="%i", header=candidates_header)
ps = sumu.aps()
print(ps)
dags_header = "# DAGs represented as \"|\" separated families. The first int is the node and the following, if any, the parents."
with open(args.output_prefix + ".dags", "w") as f:
    f.write(dags_header + "\n")
    for dag in dags:
        f.write(sumu.utils.io.dag_to_str(dag) + "\n")

if not data.discrete:
    dags_adj = [sumu.bnet.family_sequence_to_adj_mat(dag) for dag in dags]
    causal_effects = sumu.beeps(dags_adj, data)
    causal_header = "Empirical distribution over {} DAGs of pairwise causal effects.\n".format(len(dags))
    causal_header += " ".join("{}->{}".format(i, j) for i in range(data.n) for j in list(range(i)) + list(range(i+1, data.n)))
    np.savetxt(args.output_prefix + ".causal_effects", causal_effects, fmt="%.3f", header=causal_header)