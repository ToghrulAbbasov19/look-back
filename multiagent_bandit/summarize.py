"""Print the final tables of a finished `run.py` root.

python -m multiagent_bandit.summarize --root runs/multiagent_bandit
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .run import POPULATIONS, SEEDS

ROWS = (
    ("reinforce", "direct", "real"),
    ("reinforce_baseline", "direct", "real"),
    ("cca", "direct", "real"),
    ("rloo", "direct", "real"),
    ("rloo_wm", "learned_all", "all transitions"),
    ("rloo_wm", "learned", "real only"),
    ("rlb", "direct", "real"),
    ("rlb_cwm", "learned_all", "all transitions"),
    ("rlb_cwm", "learned", "real only"),
)
LABELS = {
    "reinforce": "REINFORCE",
    "reinforce_baseline": "REINFORCE + baseline",
    "cca": "CCA",
    "rloo": "RLOO",
    "rloo_wm": "RLOO + WM",
    "rlb": "RLB (Ours)",
    "rlb_cwm": "RLB + CWM (Ours)",
}


def summarize(root):
    root = Path(root)
    populations, seeds = POPULATIONS, SEEDS
    print(
        "Expected reward per agent at 10M counted interactions (mean ± SD over seeds; optimum 0)"
    )
    print("| Method | Counting | " + " | ".join(f"K={k}" for k in populations) + " |")
    for method, folder, counting in ROWS:
        cells = []
        for k in populations:
            values = [
                json.loads(
                    (
                        root / folder / f"k{k}" / method / f"seed_{s}" / "result.json"
                    ).read_text()
                )["expected_return"]
                for s in seeds
            ]
            cells.append(f"{np.mean(values):.2f} ± {np.std(values, ddof=1):.2f}")
        print(f"| {LABELS[method]} | {counting} | " + " | ".join(cells) + " |")
    print("\nTest conditional-mean RMSE after 5,000 updates")
    for k in populations:
        rmse = {
            kind: np.mean(
                [
                    json.loads(
                        (
                            root
                            / "offline"
                            / f"k{k}"
                            / f"seed_{s}"
                            / kind
                            / "metrics.json"
                        ).read_text()
                    )["metrics"]["test"]["conditional_mean_rmse"]
                    for s in seeds
                ]
            )
            for kind in ("wm", "cwm")
        }
        print(f"K={k}: WM {rmse['wm']:.3f}, CWM {rmse['cwm']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("runs/multiagent_bandit"))
    summarize(parser.parse_args().root)


if __name__ == "__main__":
    main()
