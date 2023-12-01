import igraph as ig
from matplotlib import pyplot as plt
import numpy as np

from src.probabilities import ScoreManager, R
from src.sampling import sample

score_name = 'insurance-400'
score_manager = ScoreManager(score_name)
n = 10000
v_count = len(score_manager.scores)

G = ig.Graph(directed=True)
G.add_vertices(v_count)

steps = sample(G, n, [], score_manager)
scores = [step[1] for step in steps]
plt.plot(np.arange(len(scores)), scores, 'm-')

steps = sample(G, n, ['rev'], score_manager)
scores = [step[1] for step in steps]
plt.plot(np.arange(len(scores)), scores, 'g-')

plt.xlabel('')
plt.ylabel('Score')

plt.title(f'{score_name}')
plt.legend()
plt.show()
