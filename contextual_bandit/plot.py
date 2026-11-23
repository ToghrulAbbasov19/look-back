"""Plot the policy- and model-learning panels of the contextual-bandit figure.

python -m contextual_bandit.plot --input results/contextual_bandit --output results/contextual_bandit/figure
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("reinforce", "all", "REINFORCE", "#9bb8c9"),
    ("reinforce_baseline", "all", "REINFORCE + baseline", "#6e98ba"),
    ("cca", "all", "CCA", "#b394d1"),
    ("rloo_wm", "real", "RLOO + WM", "#5a9891"),
    ("rloo", "all", "RLOO", "#69b7ae"),
    ("rlb_cwm", "real", "RLB + CWM (Ours)", "#ef5b55"),
    ("rlb", "all", "RLB (Ours)", "#e48659"),
)
SIGMAS = (0.1, 30.0, 60.0)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def mean_and_sem(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std(ddof=1) / np.sqrt(len(values))


def policy_curves(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            float(row["noise_std"]),
            row["method"],
            row["budget_mode"],
            int(row["charged_interactions"]),
        )
        groups[key].append(float(row["expected_return"]))
    return groups


def model_curves(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (float(row["noise_std"]), row["kind"], int(float(row["step"])))
        groups[key].append(float(row["conditional_mean_rmse"]))
    return groups


def plot(input_dir, output):
    input_dir = Path(input_dir)
    policies = policy_curves(read_csv(input_dir / "individual_learning_curves.csv"))
    models = model_curves(read_csv(input_dir / "offline" / "learning_curves.csv"))

    figure, axes = plt.subplots(1, 6, figsize=(10.8, 3.0))
    for index, sigma in enumerate(SIGMAS):
        axis = axes[index]
        for method, mode, label, color in METHODS:
            keys = sorted(key for key in policies if key[:3] == (sigma, method, mode))
            x = np.asarray([key[3] for key in keys], dtype=float) / 1e6
            stats = np.asarray([mean_and_sem(policies[key]) for key in keys])
            axis.plot(x, stats[:, 0], color=color, lw=1.4, label=label)
            axis.fill_between(
                x,
                stats[:, 0] - stats[:, 1],
                stats[:, 0] + stats[:, 1],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        for method, _, label, color in METHODS:
            if method not in ("rloo_wm", "rlb_cwm"):
                continue
            keys = sorted(key for key in policies if key[:3] == (sigma, method, "all"))
            x = np.asarray([key[3] for key in keys], dtype=float) / 1e6
            stats = np.asarray([mean_and_sem(policies[key]) for key in keys])
            axis.plot(x, stats[:, 0], color=color, lw=1.25, ls=(0, (2, 2)))
            axis.fill_between(
                x,
                stats[:, 0] - stats[:, 1],
                stats[:, 0] + stats[:, 1],
                color=color,
                alpha=0.1,
                linewidth=0,
            )
        axis.set_title(rf"$\sigma={sigma:g}$", fontsize=9)
        axis.set_xlim(0, 10)
        axis.set_ylim(-80, 2)
        axis.set_xticks((0, 5, 10))
        axis.set_xticklabels(("0", "5M", "10M"))
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.set_xlabel("Environment interactions", fontsize=8)
        axis.tick_params(labelsize=7)
        if index == 0:
            axis.set_ylabel("Reward per agent", fontsize=8)
        else:
            axis.set_yticklabels([])

    model_colors = {"world": "#5a9891", "counterfactual": "#ef5b55"}
    model_labels = {"world": "WM", "counterfactual": "CWM"}
    for index, sigma in enumerate(SIGMAS, 3):
        axis = axes[index]
        for kind in ("world", "counterfactual"):
            keys = sorted(key for key in models if key[:2] == (sigma, kind))
            x = np.asarray([key[2] for key in keys])
            stats = np.asarray([mean_and_sem(models[key]) for key in keys])
            color = model_colors[kind]
            axis.plot(x, stats[:, 0], color=color, lw=1.4, label=model_labels[kind])
            axis.fill_between(
                x,
                np.maximum(stats[:, 0] - stats[:, 1], 1e-5),
                stats[:, 0] + stats[:, 1],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        axis.set_title(rf"$\sigma={sigma:g}$", fontsize=9)
        axis.set_yscale("log")
        axis.set_xlim(0, 5000)
        axis.set_xticks((0, 2500, 5000))
        axis.set_xticklabels(("0", r"$2.5k$", r"$5k$"))
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.set_xlabel("Optimizer updates", fontsize=8)
        axis.tick_params(labelsize=7)
        if index == 3:
            axis.set_ylabel("Conditional mean RMSE", fontsize=8)
            axis.legend(frameon=False, fontsize=7, loc="lower left")
        else:
            axis.set_yticklabels([])

    figure.text(0.255, 0.89, "Policy learning", ha="center", fontsize=10, weight="bold")
    figure.text(0.75, 0.89, "Model learning", ha="center", fontsize=10, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    solid = plt.Line2D(
        [], [], color="black", lw=1.2, label="Only real interactions counted"
    )
    dotted = plt.Line2D(
        [], [], color="black", lw=1.2, ls=(0, (2, 2)), label="Imaginations also counted"
    )
    figure.legend(
        [*handles, solid, dotted],
        [*labels, solid.get_label(), dotted.get_label()],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=5,
        frameon=False,
        fontsize=7,
        handlelength=2.0,
        columnspacing=1.2,
    )
    figure.subplots_adjust(left=0.06, right=0.995, bottom=0.18, top=0.78, wspace=0.12)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=Path("results/contextual_bandit"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/contextual_bandit/figure")
    )
    args = parser.parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()
