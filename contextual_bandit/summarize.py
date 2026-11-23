"""Collect a finished `run.py` root into CSV files and print the final tables.

python -m contextual_bandit.summarize --root runs/contextual_bandit --output results/contextual_bandit
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

LABELS = {
    "reinforce": "REINFORCE",
    "reinforce_baseline": "REINFORCE + baseline",
    "cca": "CCA",
    "rloo": "RLOO",
    "rloo_wm": "RLOO + WM",
    "rlb": "RLB (Ours)",
    "rlb_cwm": "RLB + CWM (Ours)",
}
ORDER = (
    ("reinforce", "all"),
    ("reinforce_baseline", "all"),
    ("cca", "all"),
    ("rloo", "all"),
    ("rloo_wm", "all"),
    ("rloo_wm", "real"),
    ("rlb", "all"),
    ("rlb_cwm", "all"),
    ("rlb_cwm", "real"),
)


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    return float(np.mean(values)), float(np.std(values, ddof=1))


def summarize(root, output):
    root, output = Path(root), Path(output)
    protocol = json.loads((root / "protocol.json").read_text())
    sigmas, seeds = protocol["sigmas"], protocol["seeds"]

    runs, curves, finals = [], [], defaultdict(list)
    for job in protocol["all_rl_jobs"]:
        folder = Path(job["output"])
        result = json.loads((folder / "result.json").read_text())
        method, mode = job["method"], job["budget_mode"]
        sigma, seed = job["environment"]["noise_std"], job["seed"]
        runs.append(
            {
                key: result[key]
                for key in (
                    "method",
                    "budget_mode",
                    "noise_std",
                    "seed",
                    "M",
                    "charged_interactions",
                    "environment_interactions",
                    "model_samples",
                    "updates",
                    "regret",
                    "expected_return",
                )
            }
        )
        finals[(sigma, method, mode)].append(result["expected_return"])
        for line in (folder / "progress.jsonl").read_text().splitlines():
            curves.append(
                {
                    "noise_std": sigma,
                    "method": method,
                    "budget_mode": mode,
                    "seed": seed,
                    **json.loads(line),
                }
            )

    model_runs, model_curves, model_finals = [], [], defaultdict(list)
    for sigma in sigmas:
        for seed in seeds:
            for kind, name in (("world", "wm"), ("counterfactual", "cwm")):
                folder = root / "offline" / f"sigma_{sigma:g}" / f"seed_{seed}" / name
                metrics = json.loads((folder / "metrics.json").read_text())
                test = metrics["metrics"]["test"]
                model_runs.append(
                    {
                        "noise_std": sigma,
                        "seed": seed,
                        "kind": kind,
                        "parameter_count": metrics["parameter_count"],
                        **test,
                    }
                )
                model_finals[(sigma, kind)].append(test["conditional_mean_rmse"])
                with (folder / "learning_curve.csv").open(newline="") as handle:
                    for point in csv.DictReader(handle):
                        model_curves.append(
                            {
                                "noise_std": sigma,
                                "seed": seed,
                                "kind": kind,
                                **{
                                    k: float(v) if v else None for k, v in point.items()
                                },
                            }
                        )

    summary = []
    for (sigma, method, mode), values in sorted(finals.items()):
        assert len(values) == len(seeds), (sigma, method, mode)
        mean, sd = mean_sd(values)
        summary.append(
            {
                "noise_std": sigma,
                "method": method,
                "budget_mode": mode,
                "seeds": len(values),
                "mean_expected_return": mean,
                "sd_expected_return": sd,
            }
        )
    model_summary = []
    for (sigma, kind), values in sorted(model_finals.items()):
        mean, sd = mean_sd(values)
        model_summary.append(
            {
                "noise_std": sigma,
                "kind": kind,
                "seeds": len(values),
                "mean_conditional_mean_rmse": mean,
                "sd_conditional_mean_rmse": sd,
            }
        )

    (output / "offline").mkdir(parents=True, exist_ok=True)
    write_csv(output / "runs.csv", runs)
    write_csv(output / "individual_learning_curves.csv", curves)
    write_csv(output / "final_summary.csv", summary)
    write_csv(output / "offline" / "runs.csv", model_runs)
    write_csv(output / "offline" / "learning_curves.csv", model_curves)
    write_csv(output / "offline" / "final_summary.csv", model_summary)

    lookup = {(r["noise_std"], r["method"], r["budget_mode"]): r for r in summary}
    print(
        "Expected reward at 10M counted interactions (mean ± SD over seeds; optimum 0)"
    )
    print("| Method | Counting | " + " | ".join(f"σ={w:g}" for w in sigmas) + " |")
    for method, mode in ORDER:
        counting = (
            {"all": "all transitions", "real": "real only"}[mode]
            if method.endswith("wm")
            else "real"
        )
        cells = [
            f"{lookup[(w, method, mode)]['mean_expected_return']:.3f} ± "
            f"{lookup[(w, method, mode)]['sd_expected_return']:.3f}"
            for w in sigmas
        ]
        print(f"| {LABELS[method]} | {counting} | " + " | ".join(cells) + " |")
    print("\nTest conditional-mean RMSE after 5,000 updates")
    for sigma in sigmas:
        wm, cwm = (
            next(r for r in model_summary if r["noise_std"] == sigma and r["kind"] == k)
            for k in ("world", "counterfactual")
        )
        print(
            f"σ={sigma:g}: WM {wm['mean_conditional_mean_rmse']:.5f}, CWM {cwm['mean_conditional_mean_rmse']:.5f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("runs/contextual_bandit"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/contextual_bandit")
    )
    args = parser.parse_args()
    summarize(args.root, args.output)


if __name__ == "__main__":
    main()
