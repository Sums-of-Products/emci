import igraph as ig
import numpy as np
from utils import read_scores_from_file
from scipy.special import binom


class ScoreManager:
    def __init__(self, score_name: str):
        self.scores = read_scores_from_file(f'data/scores/{score_name}.jkl')

    def P(self, M: ig.Graph):
        def f(n, G_i_count):
            return 1 / binom(n - 1, G_i_count)

        G_i_count = np.fromiter(
            map(lambda v: len(list(M.predecessors(v))), M.vs), int)

        return f(len(list(M.vs)), G_i_count).prod()

    def get_local_score(self, v, pa_i, n):
        k = len(pa_i)

        try:
            res = self.scores[v][pa_i]

            # Use Koivisto prior
            prior = np.log(1 / binom(n, k))
            res += prior
        except KeyError:
            res = -np.inf

        return res

    def get_score(self, G: ig.Graph):
        score = 0
        n = len(G.vs)

        for v in G.vs:
            pi = frozenset(map(lambda x: x, G.predecessors(v)))
            local_score = self.get_local_score(v.index, pi, n)

            # If it is inf, just return
            if (local_score == -np.inf):
                return local_score

            score += local_score

        return score


def R(current_score, proposed_score):
    if (proposed_score == -np.inf):
        return 0

    # Prevent overlow
    if (proposed_score - current_score > 700):
        exp = 1
    else:
        exp = np.exp(proposed_score - current_score)

    return exp
