import igraph as ig
from matplotlib import pyplot as plt
import numpy as np
from src.probabilities import ScoreManager
from src.sampling import sample

score_name = 'insurance-1000'
score_manager = ScoreManager(score_name)
n = 20000

# Initial graph
G = ig.Graph(directed=True)
G.add_vertices(len(score_manager.scores))

steps = sample(G, n, [], score_manager)
scores = [step[1] for step in steps]
plt.plot(np.arange(len(scores)), scores, 'm-', label="Basic")

steps = sample(G, n, ['rev'], score_manager)
scores = [step[1] for step in steps]
plt.plot(np.arange(len(scores)), scores, 'g-', label="w/ REV")

plt.xlabel('')
plt.ylabel('Score')

plt.title(f'{score_name}')
plt.legend()
plt.show()
