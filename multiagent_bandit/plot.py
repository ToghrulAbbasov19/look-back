"""Plot the policy- and model-learning panels of the multi-agent figure.

python -m multiagent_bandit.plot --root runs/multiagent_bandit --output results/multiagent_bandit/figure
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


POPULATIONS = (8, 20, 60)
SEEDS = range(6)
POLICY_METHODS = (
    "reinforce",
    "reinforce_baseline",
    "cca",
    "rloo_wm",
    "rloo",
    "rlb_cwm",
    "rlb",
)
LABELS = {
    "reinforce": "REINFORCE",
    "reinforce_baseline": "REINFORCE + baseline",
    "cca": "CCA",
    "rloo_wm": "RLOO + WM",
    "rloo": "RLOO",
    "rlb_cwm": "RLB + CWM (Ours)",
    "rlb": "RLB (Ours)",
}
COLORS = {
    "reinforce": "#a9c7db",
    "reinforce_baseline": "#5f8fae",
    "cca": "#b79ad6",
    "rloo_wm": "#2a9d8f",
    "rloo": "#59a5a5",
    "rlb_cwm": "#f15b5a",
    "rlb": "#f28e52",
}
MODEL_COLORS = {"wm": COLORS["rloo_wm"], "cwm": COLORS["rlb_cwm"]}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def interpolate_runs(xs: list[np.ndarray], ys: list[np.ndarray], grid: np.ndarray):
    values = np.vstack([np.interp(grid, x, y) for x, y in zip(xs, ys)])
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def policy_path(
    root: Path, population: int, method: str, seed: int, include_generated: bool = False
) -> Path:
    if method in {"rloo_wm", "rlb_cwm"}:
        family = "learned_all" if include_generated else "learned"
    else:
        family = "direct"
    return root / family / f"k{population}" / method / f"seed_{seed}" / "progress.jsonl"


def policy_curve(
    root: Path, population: int, method: str, include_generated: bool = False
):
    xs, ys = [], []
    for seed in SEEDS:
        rows = read_jsonl(
            policy_path(root, population, method, seed, include_generated)
        )
        if include_generated:
            # Separate runs in which generated branches are charged to the budget.
            x = np.asarray([r["charged_interactions"] for r in rows], dtype=float)
        else:
            x = np.asarray([r["environment_interactions"] for r in rows], dtype=float)
        y = np.asarray([r["expected_return"] for r in rows], dtype=float)
        keep = np.r_[True, np.diff(x) > 0]
        xs.append(x[keep])
        ys.append(y[keep])
    maximum = min(x[-1] for x in xs)
    grid = np.linspace(0.0, min(maximum, 10_000_000.0), 301)
    return grid, *interpolate_runs(xs, ys, grid)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def model_curve(root: Path, population: int, kind: str):
    xs, ys = [], []
    for seed in SEEDS:
        path = (
            root
            / "offline"
            / f"k{population}"
            / f"seed_{seed}"
            / kind
            / "learning_curve.csv"
        )
        rows = read_csv(path)
        xs.append(np.asarray([float(r["step"]) for r in rows]))
        ys.append(np.asarray([float(r["conditional_mean_rmse"]) for r in rows]))
    grid = np.unique(np.concatenate(xs))
    return grid, *interpolate_runs(xs, ys, grid)


def style_axis(ax, labelsize=8):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(color="#777777", labelsize=labelsize, length=3)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.75)


def make_figure(root: Path, output: Path):
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(10.0, 4.45))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.05, 0.95),
        left=0.07,
        right=0.99,
        bottom=0.15,
        top=0.69,
        wspace=0.27,
    )
    policy_grid = outer[0].subgridspec(1, 3, wspace=0.18)
    model_grid = outer[1].subgridspec(1, 3, wspace=0.18)
    model_axes = [fig.add_subplot(model_grid[i]) for i in range(3)]
    policy_axes = [fig.add_subplot(policy_grid[i]) for i in range(3)]

    for column, population in enumerate(POPULATIONS):
        ax = policy_axes[column]
        for method in POLICY_METHODS:
            x, mean, std = policy_curve(root, population, method)
            ax.plot(
                x, mean, color=COLORS[method], linewidth=1.55, linestyle="-", zorder=3
            )
            ax.fill_between(
                x, mean - std, mean + std, color=COLORS[method], alpha=0.12, linewidth=0
            )
        # Dotted curves last: RLB + CWM (all counted) nearly coincides with RLB.
        for method in ("rloo_wm", "rlb_cwm"):
            x_all, mean_all, _ = policy_curve(
                root, population, method, include_generated=True
            )
            ax.plot(
                x_all,
                mean_all,
                color=COLORS[method],
                linewidth=1.45,
                linestyle=(0, (1.5, 2.2)),
                alpha=1.0,
                zorder=6,
            )
        style_axis(ax, labelsize=8)
        ax.set_xlim(0, 10_000_000)
        ax.set_ylim(-80, 1)
        ax.set_xticks((0, 5_000_000, 10_000_000), ("0", "5M", "10M"))
        ax.set_yticks((-75, -50, -25, 0))
        ax.set_title(rf"$K={population}$", pad=3)
        if column == 0:
            ax.set_ylabel("Reward per agent")
        else:
            ax.tick_params(labelleft=False)
        if column == 1:
            ax.set_xlabel("Environment interactions")

        maxis = model_axes[column]
        for kind in ("wm", "cwm"):
            x, mean, std = model_curve(root, population, kind)
            lower = np.maximum(mean - std, np.finfo(float).tiny)
            maxis.plot(x, mean, color=MODEL_COLORS[kind], linewidth=1.55, zorder=3)
            maxis.fill_between(
                x, lower, mean + std, color=MODEL_COLORS[kind], alpha=0.14, linewidth=0
            )
        style_axis(maxis)
        maxis.set_yscale("log")
        maxis.set_xlim(0, 5000)
        maxis.set_ylim(1.0, 1.5e2)
        maxis.set_xticks((0, 2500, 5000), ("0", "2.5k", "5k"))
        maxis.set_title(rf"$K={population}$", pad=3)
        if column == 0:
            maxis.set_ylabel("Conditional mean RMSE")
        else:
            maxis.tick_params(labelleft=False)
        if column == 1:
            maxis.set_xlabel("Optimizer updates")

    fig.text(
        0.268,
        0.735,
        "Policy learning",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.754,
        0.735,
        "Model learning",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

    method_handles = [
        Line2D(
            [0], [0], color=COLORS[m], linewidth=1.55, linestyle="-", label=LABELS[m]
        )
        for m in POLICY_METHODS
    ]
    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        columnspacing=1.65,
        handlelength=2.0,
        handletextpad=0.45,
        fontsize=8,
    )
    convention_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=1.3,
            label="Only real interactions counted",
        ),
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=1.3,
            linestyle=(0, (1.5, 2.2)),
            label="Imaginations also counted",
        ),
    ]
    fig.legend(
        handles=convention_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=2,
        frameon=False,
        columnspacing=2.5,
        handlelength=2.7,
        fontsize=8,
    )
    model_axes[0].legend(
        handles=[
            Line2D([0], [0], color=MODEL_COLORS["wm"], linewidth=1.5, label="WM"),
            Line2D([0], [0], color=MODEL_COLORS["cwm"], linewidth=1.5, label="CWM"),
        ],
        loc="lower left",
        frameon=False,
        handlelength=1.6,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("runs/multiagent_bandit"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/multiagent_bandit/figure")
    )
    args = parser.parse_args()
    make_figure(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
