import sys
import numpy as np
import dill
import matplotlib.pyplot as plt
import igraph as ig
from src.utils import ScoreManager

chain_type_to_label = {
    # "": "Basic",
    # "REV MES": "Structural w/ REV & MES",
    # "REV": "Basic w/ REV",
    # "MES": "Basic w/ MES",
    "REV Partition": "Partition w/ REV",
    "REV MES Partition": "Partition w/ REV & MES",
}

chain_type_to_colour = {
    "": "green",
    "REV MES": "violet",
    "REV": "blue",
    "MES": "red",
    "REV Partition": "cyan",
    "REV MES Partition": "magenta",
}

chain_type_to_ratio = {
    "": 10,
    "REV MES": 1,
    "REV": 1,
    "MES": 10,
    "REV Partition": 1,
    "REV MES Partition": 1,
}
score_name = sys.argv[1]
n = int(sys.argv[2])
num_runs = int(sys.argv[3])

data_counts = [100, 1000, 10000]


def calculate_scores(G: ig.Graph):
    scores = []
    for g in G:
        like, prior = score_manager.get_score(g)
        scores.append(like + prior)

    return scores


def type_to_line(type: str):
    if "MES" in chain_type:
        return "--"
    if "REV" in chain_type:
        return ":"

    return "-"


# Plotting both metrics
fig, axes = plt.subplots(1, len(data_counts), figsize=(15, 5))

for j, m in enumerate(data_counts):
    score_manager = ScoreManager(f"{score_name}-{m}")
    scores = {key: [] for key in chain_type_to_label.keys()}

    # Iterate over the runs
    for i in range(num_runs):
        for chain_type in scores:
            chain = np.load(
                f"res/{score_name}-{m}/chain-n={n * chain_type_to_ratio[chain_type]}.{i}.{chain_type.lower()}.npy",
                allow_pickle=True,
            )

            scores[chain_type].append(calculate_scores(chain))

    # Calculate median errors and plot for each metric
    for chain_type in scores:
        scores_array = np.array(scores[chain_type])
        for idx, score in enumerate(scores_array):
            print(j)
            axes[j].plot(
                np.arange(0, len(score)),
                score,
                type_to_line(chain_type),
                color=chain_type_to_colour[chain_type],
                label=chain_type_to_label[chain_type] if idx == 0 else "",
                alpha=0.6,
            )
            axes[j].set_title(f"{score_name}-{m}")
            axes[j].set_xlabel("Steps")
            axes[j].set_ylabel(f"Score")

axes[0].legend()
# plt.suptitle(f"{score_name} {num_runs} chains")
# Place the legend outside the right of the last subplot
# handles, labels = axes.get_legend_handles_labels()
# fig.legend(
#     handles, labels, loc="upper right", bbox_to_anchor=(0.14, 0.97), borderaxespad=0.0
# )
# fig.legend(
#     handles, labels, loc="upper right", bbox_to_anchor=(0.2, 0.97), borderaxespad=0.0
# )
# plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.show()
