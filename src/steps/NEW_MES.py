import igraph as ig
import networkx as nx
import random
import numpy as np
from .MES.count import count
class ScoreManager:
    # Placeholder for the actual ScoreManager implementation
    pass

class NEW_MES:
    def __init__(self, score_manager: ScoreManager):
        self.score_manager = score_manager

    def count(self, G: nx.Graph):
        # Placeholder for the actual count method implementation
        # This method should return the size of the Markov equivalence class for the given graph
        return 1

    def igraph_to_networkx(self, G: ig.Graph):
        """Converts an igraph graph to a networkx graph"""
        G_nx = nx.Graph()
        for edge in G.es:
            G_nx.add_edge(edge.source, edge.target)
        return G_nx

    def new_mes_move(self, G: ig.Graph):
        M = G.copy()
        n = len(M.vs)

        if len(M.es) < 2:
            return G, False

        # Select a random node
        node_idx = random.randint(0, n-1)

        # Generate all possible parent sets for the selected node
        possible_parents = []
        for i in range(n):
            if i != node_idx:
                possible_parents.append(i)

        # Count the Markov equivalence class size for each possible parent set
        equivalence_sizes = []
        parent_sets = []
        for parent in possible_parents:
            temp_graph = M.copy()
            temp_graph.add_edges([(parent, node_idx)])
            if (not temp_graph.is_dag()):
                continue
            temp_nx_graph = self.igraph_to_networkx(temp_graph)
           
            equivalence_size = count(temp_nx_graph)
            equivalence_sizes.append(equivalence_size)
            parent_sets.append(parent)

        if (len(parent_sets) == 0):
            return G, True
        # Normalize the equivalence sizes to get probabilities
        probabilities = np.array(equivalence_sizes) / sum(equivalence_sizes)

        # Choose a parent set randomly in relation to the class size
        chosen_parent = np.random.choice(parent_sets, p=probabilities)
        # chosen_parent = parent_sets[chosen_parent_set_idx]

        # Update the graph with the chosen parent set
        # Clear existing edges to the selected node
        # M.delete_edges(M.es.select(_target=node_idx))
        # Add new edges from the chosen parent set
        # for parent in chosen_parent_set:
        M.add_edge(chosen_parent, node_idx)

        # if (M.is_dag()):
        #     print("asd")
            
        return M, True
        # else:
        #     return G, False
