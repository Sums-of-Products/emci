import random
import igraph as ig
import itertools
from functools import reduce
import numpy as np
from scipy.special import binom
import copy
from tqdm import tqdm
from src.steps import MES, REV

from src.utils import ScoreManager

max_parents_size = 3


def partition_mcmc(
    G: ig.Graph, N, additional_steps, score_manager: ScoreManager, show_progress=False
):
    """Generator.
    yields (Partition, score), (Graph, score)
    """

    is_MES = "mes" in additional_steps
    is_REV = "rev" in additional_steps

    REV_move = REV(score_manager).new_edge_reversal_move

    A_i: list[set] = create_pratition(G)
    n = len(G.vs)
    scores = P_partition(A_i, n, score_manager)

    G_i = G

    pbar = (
        tqdm(range(N), bar_format="{desc}: {bar}")
        if show_progress
        else iter(lambda: True, None)
    )

    for i in pbar:
        skip = False

        if is_REV and np.random.uniform() < 0.067:
            m_i = len(A_i)
            G_i = sample_dag(A_i, len(G.vs), score_manager)
            G_i, type = REV_move(G_i)

            # Markov Equivalence step runs every time REV is sampled
            if is_MES:
                G_i, AMOs = MES(G_i)

            A_i = create_pratition(G_i)
            scores = P_partition(A_i, n, score_manager)
            # print(f'{i} REV {sum(scores.values())} {A_i} ')
            skip = True

        elif np.random.uniform() < 0.01:
            skip = True
        else:
            A_i_p_1, scores_p_1 = sample_partition(A_i, scores, n, score_manager)

        if not skip:
            m_i = len(A_i)
            m_i_p_1 = len(A_i_p_1)

            score = sum(scores.values())
            proposed_score = sum(scores_p_1.values())

            # print(f'{i} {proposed_score} {A_i_p_1}')

            score_delta = proposed_score - score
            if score_delta > 250:
                A = 1
            else:
                R = (nbd(A_i, m_i) / nbd(A_i_p_1, m_i_p_1)) * np.exp(score_delta)
                A = np.min([1, R])

            if np.random.uniform() < A:
                A_i = A_i_p_1
                scores = scores_p_1

        if show_progress:
            pbar.set_description(f"Score: {sum(scores.values()):.2f}")

        likelihood, prior = score_manager.get_score(G_i)
        score = likelihood + prior
        yield G_i, score


def sample_partition(
    prev_partitions: list[set],
    prev_scores: dict[int, float],
    n: int,
    score_manager: ScoreManager,
):
    partitions = copy.deepcopy(prev_partitions)
    scores: dict[int, float] = copy.deepcopy(prev_scores)

    m = len(partitions)

    j = random.randint(1, int(nbd(partitions, m)))
    vertices_to_rescore = set()

    if j < m:
        # Join partitions
        partition_1 = partitions.pop(j - 1)
        partition_2 = partitions.pop(j - 1)

        new_partition = partition_1 | partition_2
        partitions.insert(j - 1, new_partition)
        vertices_to_rescore |= partition_1
        if j - 2 >= 0:
            vertices_to_rescore |= partitions[j - 2]
    else:
        # Split partition
        i_min = find_i_min(partitions, j)
        c_min = find_c_min(partitions, j, i_min)
        new_partition = set(random.sample(list(partitions[i_min - 1]), c_min))
        partitions[i_min - 1] -= new_partition
        partitions.insert(i_min - 1, new_partition)

        vertices_to_rescore |= new_partition
        if i_min - 2 >= 0:
            vertices_to_rescore |= partitions[i_min - 2]

    for v in vertices_to_rescore:
        scores[v] = P_v(partitions, v, n, score_manager)

    return partitions, scores


def create_pratition(G: ig.Graph) -> list[set]:
    G_t: ig.Graph = G.copy()
    partitions = []

    for v in G_t.vs:
        v["original_index"] = v.index

    while len(G_t.vs) > 0:
        outpoints = [v for v in G_t.vs if G_t.degree(v, mode="in") == 0]
        partitions.insert(0, set([v["original_index"] for v in outpoints]))

        G_t.delete_vertices(outpoints)

    return partitions


def P_v(
    partitions: list[set], v: int, n, score_manager: ScoreManager, parent_sets=False
):
    # flat_vertices = itertools.chain.from_iterable(partitions)

    if not parent_sets:
        parent_sets = get_permissible_parent_sets(partitions, v)

    scores = [score_manager.get_local_score(v, pa, n) for pa in parent_sets]
    return reduce(np.logaddexp, scores, -np.inf)


def get_permissible_parent_sets(partitions: list[set], v: int):
    # Only parent sets with at least one member in the partition element
    # immediately to the right need to be included.
    partition_index = 0
    v_index_for_searching = 0

    # Find the partition containing v and its index.
    for i, partition in enumerate(partitions):
        if v in partition:
            partition_index = i
            break
        v_index_for_searching += len(partition)

    # If the vertex is in the last partition, return empty set
    if partition_index == len(partitions) - 1:
        return [frozenset()]

    # Get the vertices to the right of the current vertex's partition.
    vertices_to_right = list(
        itertools.chain.from_iterable(partitions[partition_index + 1 :])
    )

    # Generator to yield permissible parent sets
    def parent_sets_generator():
        partition_to_the_right = partitions[partition_index + 1]
        for size in range(1, max_parents_size + 1):
            for p in itertools.permutations(vertices_to_right, size):
                parent_set = frozenset(p)
                if len(parent_set.intersection(partition_to_the_right)) > 0:
                    yield parent_set

    parent_sets = list(set(parent_sets_generator()))

    return parent_sets


def P_partition(
    partitions: list[set], n: int, score_manager: ScoreManager
) -> list[float]:
    flat_vertices = list(itertools.chain.from_iterable(partitions))

    scores = {v: P_v(partitions, v, n, score_manager) for v in flat_vertices}
    return scores


def nbd_sum(partitions: tuple[set], m: int):
    return sum(
        [
            sum(
                [
                    binom(len(partitions[i - 1]), c)
                    for c in range(1, len(partitions[i - 1]))
                ]
            )
            for i in range(1, m + 1)
        ]
    )


def nbd(partitions: list[set], m: int):
    return m - 1 + nbd_sum(partitions, m)


def find_i_min(partitions: list[set], j):
    m = len(partitions)
    sum = 0
    i_min = 0

    while sum < j:
        i_min += 1
        sum = m - 1 + nbd_sum(partitions, i_min)

    return i_min


def find_c_min(partitions: list[set], j, i_min):
    m = len(partitions)
    base_sum = nbd_sum(partitions, i_min - 1)
    res = 0
    c_min = 0
    while res < j:
        c_min += 1
        res = (
            m
            - 1
            + base_sum
            + sum([binom(len(partitions[i_min - 1]), c) for c in range(1, c_min + 1)])
        )

    return c_min


def sample_dag(partitions: list[set], n: int, score_manager: ScoreManager):
    G = ig.Graph(directed=True)
    G.add_vertices(n)

    flat_vertices = itertools.chain.from_iterable(partitions)

    for v in flat_vertices:
        parent_sets = get_permissible_parent_sets(partitions, v)
        target_sum = P_v(partitions, v, n, score_manager)

        current_sum = np.log(0)
        j = target_sum + np.log(np.random.uniform())

        for pa_i in parent_sets:
            current_sum = np.logaddexp(
                current_sum, score_manager.get_local_score(v, pa_i, n)
            )

            if current_sum >= j:
                edges = [(p, v) for p in pa_i]
                G.add_edges(edges)
                break
        # pa_i_p = np.array([get_local_score(v, frozenset(pa_i), n)
        #                   for pa_i in parent_sets])

        # # Normalize probability
        # max_prob = np.max(pa_i_p)
        # pa_i_p_norm = np.exp(pa_i_p - max_prob)
        # pa_i_p_norm /= np.sum(pa_i_p_norm)

        # new_pa_i = np.random.choice(parent_sets, p=pa_i_p_norm)
        # edges = [(p, v) for p in new_pa_i]
        # G.add_edges(edges)

    return G
