import numpy as np
from src.steps import REV, MES
import igraph as ig
from tqdm import tqdm
from typing import Iterator
from src.utils import R, ScoreManager
import random


def mcmc(
    G: ig.Graph,
    N: int,
    additional_steps: list[str],
    score_manager: ScoreManager,
    beta=1,
    show_progress=False,
) -> Iterator[tuple[ig.Graph, float]]:
    """Generator.
    yields (Graph, score)
    """
    G_i: ig.Graph = G.copy()

    pbar = (
        tqdm(range(N), bar_format="{desc}: {bar}")
        if show_progress
        else iter(lambda: True, None)
    )

    for i in pbar:
        G_i_plus_1, step_type = propose_next(G_i, additional_steps, score_manager)

        likelihood_i, prior_i = score_manager.get_score(G_i)
        likelihood_i_p_1, prior_i_p_1 = score_manager.get_score(G_i_plus_1)

        # Temperature
        likelihood_i_p_1 *= beta

        if step_type == "MES":
            G_i, likelihood_i, prior_i = G_i_plus_1, likelihood_i_p_1, prior_i_p_1
        if step_type == "REV":
            G_i, likelihood_i, prior_i = G_i_plus_1, likelihood_i_p_1, prior_i_p_1
        else:
            A = np.min([1, R(likelihood_i, likelihood_i_p_1, prior_i, prior_i_p_1)])
            if np.random.uniform() <= A:
                G_i, likelihood_i, prior_i = G_i_plus_1, likelihood_i_p_1, prior_i_p_1

        score = likelihood_i + prior_i
        if show_progress:
            pbar.set_description(f"Score: {score:.2f}")

        yield G_i, score


def propose_next(
    G_i: ig.Graph, additional_steps: list[str], score_manager: ScoreManager
):
    a, b = random.sample(list(G_i.vs), k=2)
    G_i_plus_1: ig.Graph = G_i.copy()

    is_REV = "rev" in additional_steps
    is_MES = "mes" in additional_steps

    new_edge_reversal_move = REV(score_manager).new_edge_reversal_move

    if is_MES and np.random.uniform() < 0.07:
        new_G, AMOs = MES(G_i_plus_1)
        return new_G, "MES"
    if is_REV and np.random.uniform() < 0.07:
        return new_edge_reversal_move(G_i_plus_1)

    if G_i.are_connected(a, b):
        G_i_plus_1.delete_edges([(a, b)])
        return G_i_plus_1, "remove"
    elif G_i.are_connected(b, a):
        G_i_plus_1.delete_edges([(b, a)])
        G_i_plus_1.add_edges([(a, b)])

        if G_i_plus_1.is_dag():
            return G_i_plus_1, "reverse"
    else:
        G_i_plus_1.add_edges([(a, b)])
        if G_i_plus_1.is_dag():
            return G_i_plus_1, "add"

    return G_i, False
