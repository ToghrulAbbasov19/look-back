"""Run the full multi-agent experiment (K = 8, 20, 60 agents; seeds 0-5).

Stages: simulator methods (REINFORCE + baseline also serves as the data-collection
reference) -> paired datasets -> WM/CWM fits -> RLOO + WM and RLB + CWM in both
budget modes. Completed runs are skipped, so the command can be rerun.

    python -m multiagent_bandit.run --workers 48
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

from .environment import MultiAgentConfig
from .train import TrainConfig

POPULATIONS = (8, 20, 60)
SEEDS = tuple(range(6))
SIMULATOR_METHODS = ("reinforce_baseline", "reinforce", "cca", "rloo", "rlb")
MODEL_METHODS = ("rloo_wm", "rlb_cwm")
# "real" counts only simulator calls; "all" also counts generated branches.
BUDGET_FOLDERS = {"real": "learned", "all": "learned_all"}


def perform(task):
    stage, kwargs = task
    if stage == "rl":
        from .train import run

        result = run(kwargs)
        return {
            key: result[key]
            for key in ("method", "budget_mode", "seed", "expected_return")
        } | {
            "num_agents": kwargs["environment"]["num_agents"],
            "seconds": round(result["seconds"]),
        }
    from .offline import train_density

    result = train_density(**kwargs)
    return {
        "kind": result["kind"],
        "seed": result["seed"],
        "num_agents": result["environment"]["num_agents"],
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
    # Longest jobs first (largest K, CCA), so the pool finishes together.
    pending.sort(
        key=lambda t: (
            -t[1].get("environment", {}).get("num_agents", 0),
            t[1].get("method") != "cca",
        )
    )
    futures = {pool.submit(perform, task): task for task in pending}
    for index, future in enumerate(as_completed(futures), 1):
        print(f"{name} {index}/{len(pending)}: {future.result()}", flush=True)


def job(root, folder, method, agents, seed, budget_mode="all", model=None):
    entry = {
        "method": method,
        "seed": seed,
        "M": 4,
        "budget_mode": budget_mode,
        "environment": asdict(MultiAgentConfig(num_agents=agents)),
        "training": asdict(TrainConfig()),
        "output": str(root / folder / f"k{agents}" / method / f"seed_{seed}"),
    }
    if model is not None:
        entry["model_checkpoint"] = str(model)
    return entry


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("runs/multiagent_bandit"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    direct = [
        job(root, "direct", method, agents, seed)
        for agents in POPULATIONS
        for method in SIMULATOR_METHODS
        for seed in SEEDS
    ]
    learned = [
        job(
            root,
            BUDGET_FOLDERS[mode],
            method,
            agents,
            seed,
            mode,
            root
            / "offline"
            / f"k{agents}"
            / f"seed_{seed}"
            / ("wm" if method == "rloo_wm" else "cwm")
            / "model.pt",
        )
        for mode in ("real", "all")
        for agents in POPULATIONS
        for method in MODEL_METHODS
        for seed in SEEDS
    ]
    protocol = {
        "populations": list(POPULATIONS),
        "seeds": list(SEEDS),
        "all_rl_jobs": direct + learned,
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    started = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        run_stage(pool, root, "simulator methods", [("rl", j) for j in direct])

        from .offline import collect_dataset
        from .reference import export_reference

        models = []
        for agents in POPULATIONS:
            for seed in SEEDS:
                folder = root / "offline" / f"k{agents}" / f"seed_{seed}"
                reference = folder / "reference.npz"
                if not reference.exists():
                    export_reference(
                        root
                        / "direct"
                        / f"k{agents}"
                        / "reinforce_baseline"
                        / f"seed_{seed}"
                        / "checkpoint.pt",
                        reference,
                    )
                data = folder / "data"
                if not (data / "dataset.json").exists():
                    collect_dataset(data, reference, seed=seed)
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
                                "batch_size": 256,
                            },
                        )
                    )
        run_stage(pool, root, "offline models", models)
        run_stage(pool, root, "learned-model methods", [("rl", j) for j in learned])
    print(f"done in {(time.time() - started) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
