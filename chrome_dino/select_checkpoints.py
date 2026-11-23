"""Evaluate policy checkpoints and rank them by mean frame-0 pass rate."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


RESULT_RE = re.compile(r"all\s+([\d.]+)%.*decisions\s+(\d+)")


@dataclass(frozen=True)
class Job:
    arm: str
    checkpoint: int
    seed: int
    log: Path


def parse_result(path: Path) -> tuple[float, int] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    match = RESULT_RE.search(path.read_text(errors="replace"))
    if match is None:
        return None
    return float(match.group(1)), int(match.group(2))


def run_job(job: Job, args: argparse.Namespace, repo: Path) -> tuple[Job, str]:
    if parse_result(job.log) is not None:
        return job, "reused"

    tmp = job.log.with_name(f".{job.log.name}.tmp.{os.getpid()}")
    command = [
        args.python,
        "-u",
        str(repo / "evaluate_policy.py"),
        job.arm,
        str(job.checkpoint),
        str(args.action_repeat),
        str(args.episodes),
        str(job.seed),
        str(args.episode_seed),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = str(args.omp_threads)
    env["DINO_RESULTS_ROOT"] = str(args.results_root)
    env["PIXEL_EVAL_BIRD_SPAWN_PROB"] = str(args.bird_spawn_prob)
    env["PIXEL_EVAL_BIRD_MIN_SPEED"] = str(args.bird_min_speed)
    low, mid, high = 437, 363, 330
    env["PIXEL_EVAL_BIRD_Y_POSITIONS"] = ",".join(
        str(x) for x in (low, *([mid] * args.mid_bird_weight), high)
    )
    if args.cactus_spawn_prob is not None:
        env["PIXEL_EVAL_BASE_CACTUS_SPAWN_PROB"] = str(args.cactus_spawn_prob)
        env["PIXEL_EVAL_MAX_CACTUS_SPAWN_PROB"] = str(args.cactus_spawn_prob)
    with tmp.open("w") as output:
        proc = subprocess.run(
            command,
            cwd=repo,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0 or parse_result(tmp) is None:
        failed = job.log.with_suffix(job.log.suffix + ".failed")
        tmp.replace(failed)
        return job, f"FAILED (see {failed})"
    tmp.replace(job.log)
    return job, "completed"


def summarize(
    arm: str,
    checkpoints: list[int],
    seeds: list[int],
    logdir: Path,
    episode_seed: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for checkpoint in checkpoints:
        frame0: list[float] = []
        frame0_decisions: list[int] = []
        complete = True
        for seed in seeds:
            log = logdir / f"{arm}_s{seed}_c{checkpoint}_e{episode_seed}.log"
            value = parse_result(log)
            if value is None:
                complete = False
                break
            frame0.append(value[0])
            frame0_decisions.append(value[1])
        if complete:
            row: dict[str, float | int] = {
                "checkpoint": checkpoint,
                "frame0_mean": sum(frame0) / len(frame0),
                "frame0_decisions_mean": sum(frame0_decisions) / len(frame0_decisions),
                **{f"seed{seed}_frame0": value for seed, value in zip(seeds, frame0)},
                **{
                    f"seed{seed}_frame0_decisions": value
                    for seed, value in zip(seeds, frame0_decisions)
                },
            }
            rows.append(row)
    return rows


def write_summary(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        "--arm",
        dest="arm",
        required=True,
        help="method prefix without _s<seed>",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--end", type=int, default=2000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--episode-seed", type=int, default=2_200_000_000)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--omp-threads", type=int, default=2)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--bird-spawn-prob", type=float, default=0.6)
    parser.add_argument("--bird-min-speed", type=float, default=20.0)
    parser.add_argument("--mid-bird-weight", type=int, default=2)
    parser.add_argument(
        "--cactus-spawn-prob",
        type=float,
        default=None,
        help="Fix both base and max cactus probability; omit for the "
        "native 0.3 -> 0.7 schedule used in training.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--results-root",
        default="runs/checkpoints",
        help="checkpoint root, relative to the repository unless absolute",
    )
    parser.add_argument(
        "--outdir",
        default="runs/evaluation",
        help="directory for per-episode logs and the summary CSV, "
        "relative to the repo root unless absolute",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = repo / results_root
    args.results_root = results_root
    logdir = Path(args.outdir)
    if not logdir.is_absolute():
        logdir = repo / logdir
    logdir.mkdir(parents=True, exist_ok=True)
    checkpoints = list(range(args.start, args.end + 1, args.step))

    jobs: list[Job] = []
    missing_checkpoints: list[str] = []
    for checkpoint in checkpoints:
        for seed in args.seeds:
            policy = results_root / f"{args.arm}_s{seed}" / f"policy-{checkpoint}.pth"
            if not policy.is_file():
                missing_checkpoints.append(os.path.relpath(policy, repo))
                continue
            log = logdir / (
                f"{args.arm}_s{seed}_c{checkpoint}_e{args.episode_seed}.log"
            )
            jobs.append(Job(args.arm, checkpoint, seed, log))

    if missing_checkpoints:
        print(
            "Missing checkpoints; validation requires every requested seed/checkpoint:"
        )
        for path in missing_checkpoints:
            print(f"  {path}")
        return 2

    reused = sum(parse_result(job.log) is not None for job in jobs)
    print(
        f"{args.arm}: {len(checkpoints)} checkpoints x {len(args.seeds)} seeds; "
        f"{reused}/{len(jobs)} results already complete, concurrency={args.jobs}",
        flush=True,
    )
    failures = 0
    finished = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_job, job, args, repo) for job in jobs]
        for future in as_completed(futures):
            job, status = future.result()
            finished += 1
            if status.startswith("FAILED"):
                failures += 1
                print(
                    f"[{finished}/{len(jobs)}] c{job.checkpoint} s{job.seed} "
                    f"{status}",
                    flush=True,
                )
            elif finished % 10 == 0 or finished == len(jobs):
                print(f"[{finished}/{len(jobs)}] evaluations accounted for", flush=True)

    rows = summarize(args.arm, checkpoints, args.seeds, logdir, args.episode_seed)
    summary_path = logdir / f"validation_{args.arm}_e{args.episode_seed}.csv"
    write_summary(summary_path, rows)
    ranked = sorted(
        rows, key=lambda row: (-float(row["frame0_mean"]), int(row["checkpoint"]))
    )

    print(f"\n{args.arm}: ranked by mean frame-0 pass rate")
    print(" rank  checkpoint  frame 0  decisions (frame 0)")
    for rank, row in enumerate(ranked, 1):
        line = (
            f" {rank:>4}  {int(row['checkpoint']):>10}  "
            f"{float(row['frame0_mean']):>7.1f}%  "
            f"{float(row['frame0_decisions_mean']):>19.0f}"
        )
        print(line)
    print(f"CSV: {summary_path}")
    return 1 if failures or len(rows) != len(checkpoints) else 0


if __name__ == "__main__":
    raise SystemExit(main())
