import numpy as np
from src.steps import NEW_MES
from src.steps.REV import REV
import igraph as ig
from tqdm import tqdm

from src.utils import R, ScoreManager
import random


def mcmc(G: ig.Graph, N: int, additional_steps: list[str], score_manager: ScoreManager, beta=1, show_progress=False):
    """ Generator. 
        yields (Graph, score)
    """
    G_i: ig.Graph = G.copy()

    is_REV = 'REV' in additional_steps
    is_NEW_MES = 'NEW_MES' in additional_steps

    pbar = tqdm(
        range(N), bar_format='{desc}: {bar}') if show_progress else iter(lambda: True, None)

    for i in pbar: 
        if (is_NEW_MES and np.random.uniform() < 0.07):
            # print(len(G_i_plus_1.vs))
            G_i_plus_1, step_type = NEW_MES(score_manager).new_mes_move(G_i)
        else:
            G_i_plus_1, step_type = propose_next(G_i, is_REV, score_manager)

        likelihood_i, prior_i = score_manager.get_score(G_i)
        likelihood_i_p_1, prior_i_p_1 = score_manager.get_score(G_i_plus_1)

        # Temperature
        likelihood_i_p_1 *= beta

        if (step_type == 'REV'):
            G_i, likelihood_i, prior_i = G_i_plus_1, likelihood_i_p_1, prior_i_p_1
        else:
            A = np.min(
                [1, R(likelihood_i, likelihood_i_p_1, prior_i, prior_i_p_1)])
            if (np.random.uniform() <= A):
                G_i, likelihood_i, prior_i = G_i_plus_1, likelihood_i_p_1, prior_i_p_1

        score = likelihood_i + prior_i
        if (show_progress):
            pbar.set_description(f'Likelihood: {likelihood_i:.2f}, Prior: {prior_i:.2f}')

        yield G_i, likelihood_i


def propose_next(G_i: ig.Graph, is_REV, score_manager: ScoreManager):
    a, b = random.sample(list(G_i.vs), k=2)
    G_i_plus_1: ig.Graph = G_i.copy()

    new_edge_reversal_move = REV(score_manager).new_edge_reversal_move

    if (is_REV and np.random.uniform() < 0.07):
        return new_edge_reversal_move(G_i_plus_1)

    if (G_i.are_connected(a, b)):
        G_i_plus_1.delete_edges([(a, b)])
        return G_i_plus_1, 'remove'
    elif (G_i.are_connected(b, a)):
        G_i_plus_1.delete_edges([(b, a)])
        G_i_plus_1.add_edges([(a, b)])

        if G_i_plus_1.is_dag():
            return G_i_plus_1, 'reverse'
    else:
        G_i_plus_1.add_edges([(a, b)])
        if G_i_plus_1.is_dag():
            return G_i_plus_1, 'add'

    return G_i, False
