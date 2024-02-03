import igraph as ig
import numpy as np
from scipy.special import binom

from .helpers import read_scores_from_file

class ScoreManager:
    def __init__(self, score_name: str, Flag = 0):
        self.scores = read_scores_from_file(f'data/scores/{score_name}.jkl')
        self.Flag = Flag

    # def P(self, M: ig.Graph):
    #     def f(n, G_i_count):
    #         return 1 / binom(n - 1, G_i_count)

    #     G_i_count = np.fromiter(
    #         map(lambda v: len(list(M.predecessors(v))), M.vs), int)

    #     return f(len(list(M.vs)), G_i_count).prod()

    def get_local_likelihood(self, v, pa_i):
        try:
            res = self.scores[v][pa_i]
        except KeyError:
            res = -np.inf

        return res

    def get_local_prior(self, v, pa_i, N):
        k = len(pa_i)

        # Use Koivisto prior
        prior = np.log(1 / binom(N - 1, k))
        return prior

    def get_local_score(self, v, pa_i, N):
        return self.get_local_likelihood(v, pa_i) + self.get_local_prior(v, pa_i, N)

    def get_score(self, G: ig.Graph):
        likelihood = 0
        prior = 0

        N = len(G.vs)
        # components = G.connected_components("weak")
        for v in G.vs:
            pi = frozenset(map(lambda x: x, G.predecessors(v)))
            local_score, local_prior = self.get_local_likelihood(
                v.index, pi), self.get_local_prior(v.index, pi, N)

            # If it is inf, just return
            if (local_score == -np.inf):
                return local_score, local_prior

            likelihood += local_score
            prior += local_prior

        return likelihood, prior


def R(likelihood_i, likelihood_i_p_1, prior_i, prior_i_p_1):
    if (likelihood_i_p_1 == -np.inf):
        return 0

    # Prevent overlow
    if (likelihood_i_p_1 - likelihood_i > 400):
        res = 1
    else:
        res = np.exp(likelihood_i_p_1 + prior_i_p_1 - likelihood_i - prior_i)

    return res
