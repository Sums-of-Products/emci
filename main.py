import igraph as ig
import seaborn as sns
from matplotlib import pyplot as plt
from src.probabilities import ScoreManager
from src.sampling import sample
import sys
import warnings
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")

if len(sys.argv) < 3:
    print("Usage: python main.py <score name> <n>")
    sys.exit(1)

score_name = sys.argv[1]

try:
    n = int(sys.argv[2])
except ValueError:
    print("Error: sys.argv[2] (n) is not an integer")
    sys.exit(1)

score_manager = ScoreManager(score_name)

# Initial graph
G = ig.Graph(directed=True)
G.add_vertices(len(score_manager.scores))

fig, (ax_main, ax_kde) = plt.subplots(nrows=1, ncols=2,
                                      gridspec_kw={'width_ratios': [4, 1]}, figsize=(8, 4))

ax_main.set_xlabel('Index')
ax_main.set_ylabel('Scores')
ax_kde.set_xlabel('Density')
ax_kde.set_ylabel('')
ax_kde.get_yaxis().set_visible(False)

# Plot
for i in range(0, 2):
    steps = sample(G, n, [], score_manager, True)
    scores = [step[1] for step in steps][len(steps)//5:]
    ax_main.plot(range(len(scores)), scores, color='blue')
    sns.kdeplot(scores, ax=ax_kde, vertical=True, color='blue', fill=True)

# Plot with REV
for i in range(0, 2):
    steps = sample(G, n, ['rev'], score_manager, True)
    scores = [step[1] for step in steps][len(steps)//5:]
    ax_main.plot(range(len(scores)), scores, color='green')
    sns.kdeplot(scores, ax=ax_kde, vertical=True, color='green', fill=True)


plt.show()
