import igraph as ig
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from src.partition_mcmc import partition_mcmc
from src.utils import ScoreManager, calculate_and_save_edge_ratios
from src.mcmc import mcmc
import sys
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

if len(sys.argv) < 3:
    print("Usage: python main.py <score name> <n>")
    sys.exit(1)

score_name = sys.argv[1]
base_n = int(sys.argv[2])
score_manager = ScoreManager(score_name)

# Initial graph
emptyG = ig.Graph(directed=True)
emptyG.add_vertices(len(score_manager.scores))

fig, (ax_main, ax_kde) = plt.subplots(
    nrows=1, ncols=2, gridspec_kw={"width_ratios": [4, 1]}, figsize=(8, 4)
)

ax_main.set_xlabel("Index")
ax_main.set_ylabel("Scores")
ax_kde.set_xlabel("Density")
ax_kde.set_ylabel("")
ax_kde.get_yaxis().set_visible(False)

colors = ["blue", "green", "yellow", "magenta"]
ratios = [1, 1, 1, 1]
variations = [["rev", "mes"]]
repetitions = 1

for color, variation, ratio in zip(colors, variations, ratios):
    for i in range(repetitions):
        n = base_n * ratio

        sample_generator = mcmc(
            emptyG,
            n,
            variation,
            score_manager,
            1,
            True,
        )
        G, scores = zip(*sample_generator)
        G, scores = G[::ratio], scores[::ratio]

        edge_ratios, variation_desc = calculate_and_save_edge_ratios(
            G, score_name, n, variation, i
        )
        np.save(f"res/{score_name}/chain-n={n}.{i}.{variation_desc}", G)

        ax_main.plot(
            range(len(scores)),
            scores,
            color=color,
            label=variation_desc if i == 0 else "",
        )

        sns.kdeplot(scores, ax=ax_kde, vertical=True, color=color, fill=True)

# exit(1)
colors = ["cyan", "red"]
ratios = [1, 1]
variations = [["rev"], ["rev", "mes"]]
for color, variation, ratio in zip(colors, variations, ratios):
    for i in range(repetitions):

        n = base_n * ratio

        # Partition
        partition_sample_generator = partition_mcmc(
            emptyG, n, variation, score_manager, True
        )
        G, score = zip(*partition_sample_generator)

        edge_ratios, variation_desc = calculate_and_save_edge_ratios(
            G, score_name, n, variation + ["partition"], i
        )

        np.save(f"res/{score_name}/chain-n={n}.{i}.{variation_desc}", G)

        ax_main.plot(
            range(len(score)),
            score,
            color=color,
            label=variation_desc if i == 0 else None,
        )
        sns.kdeplot(score, ax=ax_kde, vertical=True, color=color, fill=True)

ax_main.legend(loc="upper left")
plt.legend()
plt.show()
