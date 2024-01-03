import igraph as ig
import seaborn as sns
from matplotlib import pyplot as plt
from src.partition_mcmc import partition_mcmc
from src.utils import ScoreManager
from src.mcmc import mcmc
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

colors = ['blue', 'green', 'red', 'magenta']
variations = [[], ['rev']]
# Plot
for i in range(0, 2):
    sample_generator = mcmc(G, n, [], score_manager, 1, True)
    steps = list(sample_generator)

    scores = [step[1] for step in steps][-len(steps)//2:]
    ax_main.plot(range(len(scores)), scores, color='blue', label="Structural")
    sns.kdeplot(scores, ax=ax_kde, vertical=True, color='blue', fill=True)

# Plot with REV
for i in range(0, 2):
    sample_generator = mcmc(G, n, ['rev'], score_manager, 1, True)
    steps = list(sample_generator)

    scores = [step[1] for step in steps][-len(steps)//2:]
    ax_main.plot(range(len(scores)), scores,
                 color='green', label="Structural w/ REV")
    sns.kdeplot(scores, ax=ax_kde, vertical=True, color='green', fill=True)


for i in range(0, 2):
    # Partition
    partition_sample_generator = partition_mcmc(
        G, n, ['rev'], score_manager, True)
    steps = list(partition_sample_generator)

    scores = [step[1] for step in steps][-len(steps)//2:]
    ax_main.plot(range(len(scores)), scores, color='red',
                 label="Partition w/ REV")
    sns.kdeplot(scores, ax=ax_kde, vertical=True, color='red', fill=True)

for i in range(0, 2):
    # Partition with MES
    partition_sample_generator = partition_mcmc(
        G, n, ['rev', 'mes'], score_manager, True)
    steps = list(partition_sample_generator)

    scores = [step[1] for step in steps][-len(steps)//2:]
    ax_main.plot(range(len(scores)), scores, color='magenta',
                 label="Partition w/ REV & MES")
    sns.kdeplot(scores, ax=ax_kde, vertical=True, color='magenta', fill=True)

plt.legend()
plt.show()
