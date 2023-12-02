import igraph as ig
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from src.probabilities import ScoreManager
from src.sampling import sample

sns.set(style="whitegrid")
plt.figure(figsize=(8, 4))

score_name = 'insurance-600'
score_manager = ScoreManager(score_name)
n = 20000
x = 5

# Initial graph
G = ig.Graph(directed=True)
G.add_vertices(len(score_manager.scores))

# Plot KDE of simple struct n*x more samples
for i in range(0, 2):
    steps = sample(G, n*x, [], score_manager)
    scores = [step[1] for step in steps][-20000:]
    sns.kdeplot(scores, color="blue", label="Struct", fill=True)

# Plot KDE struct with REV
for i in range(0, 2):
    steps = sample(G, n, ['rev'], score_manager)
    scores = [step[1] for step in steps][-20000:]
    sns.kdeplot(scores[len(scores)//2:], color="green",
                label="Struct w/ REV", fill=True)

plt.xlabel('')
plt.ylabel('Score')
plt.title(f'{score_name}')
plt.legend()
plt.show()
