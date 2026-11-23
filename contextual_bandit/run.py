"""Run the full contextual-bandit experiment (noise sigma = 0.1, 30, 60; seeds 0-5).

Stages: REINFORCE + baseline reference policies -> paired datasets -> WM/CWM fits
-> all other policies. Completed runs are skipped, so the command can be rerun.

    python -m contextual_bandit.run --workers 16
"""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import time

from .config import TrainConfig
from .environment import BanditConfig


SIGMAS = (0.1, 30.0, 60.0)
SEEDS = tuple(range(6))
# (method, budget mode). "real" counts only simulator calls; "all" also counts generated ones.
COMBINATIONS = (
    ("reinforce_baseline", "all"),
    ("reinforce", "all"),
    ("cca", "all"),
    ("rloo", "all"),
    ("rlb", "all"),
    ("rloo_wm", "all"),
    ("rloo_wm", "real"),
    ("rlb_cwm", "all"),
    ("rlb_cwm", "real"),
)


def perform(task):
    stage, kwargs = task
    if stage == "rl":
        from .train import run

        result = run(kwargs)
        return {
            key: result[key]
            for key in (
                "method",
                "budget_mode",
                "noise_std",
                "seed",
                "regret",
                "seconds",
            )
        }
    from .offline import train_density

    result = train_density(**kwargs)
    return {
        "kind": result["kind"],
        "seed": result["seed"],
        "noise_std": result["environment"]["noise_std"],
        "test_mean_rmse": result["metrics"]["test"]["conditional_mean_rmse"],
    }


def run_stage(pool, root, name, tasks):
    pending = []
    for stage, kwargs in tasks:
        output = Path(kwargs["output"])
        if (output / ("result.json" if stage == "rl" else "metrics.json")).exists():
            continue
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"incomplete run, inspect or delete it: {output}")
        pending.append((stage, kwargs))
    print(
        f"{name}: {len(tasks) - len(pending)} complete, launching {len(pending)}",
        flush=True,
    )
    futures = {pool.submit(perform, task): task for task in pending}
    for index, future in enumerate(as_completed(futures), 1):
        print(f"{name} {index}/{len(pending)}: {future.result()}", flush=True)


def build_jobs(root):
    jobs = []
    for sigma in SIGMAS:
        for seed in SEEDS:
            for method, mode in COMBINATIONS:
                job = {
                    "method": method,
                    "budget_mode": mode,
                    "M": 4,
                    "seed": seed,
                    "environment": asdict(BanditConfig(noise_std=sigma)),
                    "training": asdict(TrainConfig()),
                    "output": str(
                        root
                        / "rl"
                        / f"sigma_{sigma:g}"
                        / f"{method}_{mode}"
                        / f"seed_{seed}"
                    ),
                }
                if method in ("rloo_wm", "rlb_cwm"):
                    job["model_checkpoint"] = str(
                        root
                        / "offline"
                        / f"sigma_{sigma:g}"
                        / f"seed_{seed}"
                        / ("wm" if method == "rloo_wm" else "cwm")
                        / "model.pt"
                    )
                jobs.append(job)
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("runs/contextual_bandit"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(root)
    protocol = {
        "sigmas": list(SIGMAS),
        "seeds": list(SEEDS),
        "all_rl_jobs": jobs,
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    started = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        references = [job for job in jobs if job["method"] == "reinforce_baseline"]
        run_stage(pool, root, "reference policies", [("rl", job) for job in references])

        from .offline import collect_dataset
        from .reference import export_reference

        models = []
        for job in references:
            sigma, seed = job["environment"]["noise_std"], job["seed"]
            folder = root / "offline" / f"sigma_{sigma:g}" / f"seed_{seed}"
            reference = folder / "reference.npz"
            if not reference.exists():
                export_reference(Path(job["output"]) / "checkpoint.pt", reference)
            data = folder / "data"
            if not data.exists():
                collect_dataset(data, reference, seed=seed, noise_std=sigma)
            for kind, name in (("world", "wm"), ("counterfactual", "cwm")):
                models.append(
                    (
                        "offline",
                        {
                            "dataset": str(data),
                            "output": str(folder / name),
                            "kind": kind,
                            "seed": seed,
                            "steps": 5000,
                        },
                    )
                )
        run_stage(pool, root, "offline models", models)

        remaining = [
            ("rl", job) for job in jobs if job["method"] != "reinforce_baseline"
        ]
        run_stage(pool, root, "policy comparison", remaining)
    print(f"done in {(time.time() - started) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
