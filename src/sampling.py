import numpy as np
from src.new_edge_reversal import REV
from utils import get_es_diff
import igraph as ig
from tqdm import tqdm

from src.probabilities import R, ScoreManager
import random


def sample(G: ig.Graph, N: int, additional_steps, score_manager: ScoreManager):
    G_i: ig.Graph = G.copy()
    steps: list[tuple(ig.Graph, float)] = []

    is_REV = 'rev' in additional_steps
    pbar = tqdm(range(N), bar_format='{desc}: {bar}')
    for i in pbar:
        G_i_plus_1, step_type = propose_next(G_i, is_REV, score_manager)

        current_score = score_manager.get_score(G_i)
        proposed_score = score_manager.get_score(G_i_plus_1)

        if (step_type == 'REV'):
            G_i = G_i_plus_1
        elif (step_type):
            A = np.min([1, R(current_score, proposed_score)])
            if (np.random.uniform() <= A):

                G_i = G_i_plus_1
        pbar.set_description(f'Score: {current_score:.2f}')
        steps.append((G_i, current_score))

    return steps


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
