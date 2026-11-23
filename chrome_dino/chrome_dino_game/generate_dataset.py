"""Generate fixed-length train-mode Dino episodes as parquet files."""

import argparse
import io
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from chrome_dino_game import envs
from oracle import DINO_X, oracle_action

GAME_DEFAULTS = {
    "BIRD_SPAWN_PROB": envs.BIRD_SPAWN_PROB,
    "BIRD_MIN_SPEED": envs.BIRD_MIN_SPEED,
    "BIRD_Y_POSITIONS": tuple(envs.BIRD_Y_POSITIONS[:3]),
}
SCHEMA = pa.schema(
    [
        ("episode_id", pa.int64()),
        ("step_id", pa.int64()),
        ("frames", pa.binary()),
        ("actions", pa.int64()),
        ("policy_type", pa.int64()),
        ("rewards", pa.float32()),
        ("crashes", pa.int64()),
        ("passes", pa.int64()),
        ("deviated", pa.int64()),
    ]
)

def next_obstacle(raw):
    pending = [o for o in raw._obstacles if o.needs_collision_check]
    return min(pending, key=lambda o: o.rect.x) if pending else None


def frames_to_next_obstacle(raw) -> float:
    obstacle = next_obstacle(raw)
    return (
        float("inf")
        if obstacle is None
        else (obstacle.rect.x - DINO_X) / max(raw._speed, 1)
    )


def configure_obstacles(cfg: dict):
    envs.BIRD_SPAWN_PROB = (
        GAME_DEFAULTS["BIRD_SPAWN_PROB"]
        if cfg["bird_spawn_prob"] is None
        else cfg["bird_spawn_prob"]
    )
    envs.BIRD_MIN_SPEED = (
        GAME_DEFAULTS["BIRD_MIN_SPEED"]
        if cfg["bird_min_speed"] is None
        else cfg["bird_min_speed"]
    )
    low, mid, high = GAME_DEFAULTS["BIRD_Y_POSITIONS"]
    envs.BIRD_Y_POSITIONS = (low, *[mid] * cfg["mid_bird_weight"], high)


def step_decision(raw, action: int, k: int):
    """Hold `action` for k game frames, rendering only the last one."""
    render = raw._render_frame
    raw._render_frame = lambda: None
    reward, crashes, passes, obs = 0.0, 0, 0, None
    try:
        for j in range(k):
            if j == k - 1:
                raw._render_frame = render
            obs, r, *_ = raw.step(envs.Action(action))
            reward += float(r)
            crashes += int(r < 0)
            passes += int(r > 0)
    finally:
        raw._render_frame = render
    return obs, reward, crashes, passes


def jpeg(frame: np.ndarray, quality: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(frame).convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def run_episode(job: dict) -> dict:
    cfg, path = job["cfg"], Path(job["path"])
    if path.exists():
        return {"name": job["name"], "skipped": True}
    # A fresh game per episode: sprite animation state must not depend on earlier episodes.
    raw = envs.Env(render_mode="rgb_array", game_mode="train", train_frame_limit=10**9)
    if job.get("bird_phase"):
        # Start with the bird sprite pair swapped, as in the released data (configs/sprite_phase).
        raw._init_game_data()
        birds = raw._assets.birds
        birds[0], birds[1] = birds[1], birds[0]
    k = job["k"]
    configure_obstacles(cfg)
    rng = np.random.default_rng(job["seed"])
    obs, _ = raw.reset(seed=job["seed"])
    jump_lo, jump_hi = (5.0, 9.0) if k > 1 else (0.0, 8.0)

    cols = {name: [] for name in SCHEMA.names}
    decided = []
    for t in range(job["decisions"]):
        deviating = False
        nxt = next_obstacle(raw) if cfg["deviate_prob"] > 0 else None
        if (
            nxt is not None
            and (nxt.rect.x - DINO_X) / max(raw._speed, 1) <= cfg["deviate_ttc"]
        ):
            flag = next((f for o, f in decided if o is nxt), None)
            if flag is None:
                flag = bool(rng.random() < cfg["deviate_prob"])
                decided.append((nxt, flag))
            deviating = flag

        if cfg["policy"] == "random" or deviating or rng.random() < cfg["epsilon"]:
            action = int(rng.integers(0, 3))
        else:
            action = int(oracle_action(raw, jump_ttc=jump_hi, jump_lo=jump_lo))
            airborne = (
                raw._agent.state == envs.DinoState.JUMP and raw._agent._jump_timer >= k
            )
            if (
                action == envs.Action.STAND
                and rng.random() < cfg["filler_prob"]
                and (airborne or frames_to_next_obstacle(raw) > cfg["safe_ttc"])
            ):
                action = int(rng.choice([envs.Action.JUMP, envs.Action.DUCK]))

        cols["episode_id"].append(job["episode_id"])
        cols["step_id"].append(t)
        cols["frames"].append(jpeg(obs, job["jpeg_quality"]))
        cols["actions"].append(action)
        cols["policy_type"].append(cfg["policy_type"])
        obs, reward, crashes, passes = step_decision(raw, action, k)
        cols["rewards"].append(reward)
        cols["crashes"].append(crashes)
        cols["passes"].append(passes)
        cols["deviated"].append(int(deviating))

    raw.close()
    tmp = path.with_suffix(".tmp")
    pq.write_table(pa.table(cols, schema=SCHEMA), tmp, compression="zstd")
    tmp.rename(path)
    return {
        "name": job["name"],
        "skipped": False,
        "crashes": sum(cols["crashes"]),
        "passes": sum(cols["passes"]),
        "deviated": sum(cols["deviated"]),
        "crash_decisions": sum(c > 0 for c in cols["crashes"]),
        "actions": np.bincount(cols["actions"], minlength=3).tolist(),
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", required=True)
    p.add_argument("--expert-episodes", type=int, default=600)
    p.add_argument("--random-episodes", type=int, default=0)
    p.add_argument(
        "--decisions",
        type=int,
        default=150,
        help="decisions per episode (k game frames each)",
    )
    p.add_argument("--k", type=int, default=4, help="action repeat")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue into a folder with the same generation_config.json",
    )
    p.add_argument("--filler-prob", type=float, default=0.7)
    p.add_argument("--safe-ttc", type=float, default=24.0)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--deviate-prob", type=float, default=0.3)
    p.add_argument("--deviate-ttc", type=float, default=16.0)
    p.add_argument("--expert-bird-spawn-prob", type=float, default=0.6)
    p.add_argument("--expert-bird-min-speed", type=int, default=20)
    p.add_argument("--expert-mid-bird-weight", type=int, default=2)
    p.add_argument(
        "--bird-phase",
        default=None,
        help="file with one 0/1 per expert episode: starting bird-sprite phase "
        "(reproduces the released datasets exactly; default all 0)",
    )
    a = p.parse_args()
    bird_phase = [0] * a.expert_episodes
    if a.bird_phase:
        bits = Path(a.bird_phase).read_text().strip()
        if len(bits) < a.expert_episodes or set(bits) - {"0", "1"}:
            raise SystemExit(f"{a.bird_phase}: need one 0/1 per expert episode")
        bird_phase = [int(b) for b in bits[: a.expert_episodes]]

    expert = dict(
        policy="oracle",
        policy_type=0,
        filler_prob=a.filler_prob,
        safe_ttc=a.safe_ttc,
        epsilon=a.epsilon,
        deviate_prob=a.deviate_prob,
        deviate_ttc=a.deviate_ttc,
        bird_spawn_prob=a.expert_bird_spawn_prob,
        bird_min_speed=a.expert_bird_min_speed,
        mid_bird_weight=a.expert_mid_bird_weight,
    )
    random_cfg = dict(
        policy="random",
        policy_type=1,
        filler_prob=0.0,
        safe_ttc=a.safe_ttc,
        epsilon=0.0,
        deviate_prob=0.0,
        deviate_ttc=a.deviate_ttc,
        bird_spawn_prob=None,
        bird_min_speed=None,
        mid_bird_weight=1,
    )
    config = {
        "game_mode": "train",
        "k": a.k,
        "decisions": a.decisions,
        "seed": a.seed,
        "jpeg_quality": a.jpeg_quality,
        "expert": expert,
        "random": random_cfg,
        "game_defaults": GAME_DEFAULTS,
        "JUMP_VEL": envs.JUMP_VEL,
        "MAX_SPEED": envs.MAX_SPEED,
    }

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "generation_config.json"
    if any(out.glob("*.parquet")) or config_path.exists():
        if not a.resume:
            raise SystemExit(f"{out} is not empty; pass --resume to continue it")
        old = json.loads(config_path.read_text())
        if old != json.loads(json.dumps(config)):
            raise SystemExit(f"--resume with different settings than {config_path}")
    config_path.write_text(json.dumps(config, indent=2))

    jobs = []
    for cfg, n, id_offset in (
        (expert, a.expert_episodes, 0),
        (random_cfg, a.random_episodes, 1_000_000),
    ):
        for ep in range(n):
            episode_id = id_offset + ep
            name = f"{cfg['policy_type']}_episode_{ep:06d}"
            jobs.append(
                {
                    "name": name,
                    "path": str(out / f"{name}.parquet"),
                    "cfg": cfg,
                    "episode_id": episode_id,
                    "seed": (a.seed * 10_000_019 + episode_id) % 2**32,
                    "k": a.k,
                    "decisions": a.decisions,
                    "jpeg_quality": a.jpeg_quality,
                    "bird_phase": bird_phase[ep] if cfg is expert else 0,
                }
            )
    print(
        f"{len(jobs)} episodes x {a.decisions} decisions (k={a.k}, train mode) -> {out}, {a.workers} workers"
    )
    print(f"expert: {expert}\nrandom: {random_cfg}", flush=True)

    t0, stats = time.time(), {0: [], 1: []}
    with get_context("spawn").Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_episode, jobs, chunksize=1), 1):
            if not res["skipped"]:
                stats[int(res["name"][0])].append(res)
            if i % 50 == 0 or i == len(jobs):
                parts = []
                for pt, label in ((0, "expert"), (1, "random")):
                    if stats[pt]:
                        s = stats[pt]
                        acts = np.sum([r["actions"] for r in s], axis=0)
                        parts.append(
                            f"{label} {len(s)} eps: crashes/ep {np.mean([r['crashes'] for r in s]):.1f} "
                            f"passes/ep {np.mean([r['passes'] for r in s]):.1f} "
                            f"S/J/D {'/'.join(f'{x:.0%}' for x in acts / acts.sum())}"
                        )
                print(
                    f"  {i}/{len(jobs)} ({time.time() - t0:.0f} s)  "
                    + " | ".join(parts),
                    flush=True,
                )

    summary = {}
    for pt, label in ((0, "expert"), (1, "random")):
        s = stats[pt]
        if s:
            rows = len(s) * a.decisions
            summary[label] = {
                "episodes": len(s),
                "rows": rows,
                "crashes": int(sum(r["crashes"] for r in s)),
                "passes": int(sum(r["passes"] for r in s)),
                "decisions_with_crash_fraction": round(
                    sum(r["crash_decisions"] for r in s) / rows, 4
                ),
                "deviated": int(sum(r["deviated"] for r in s)),
            }
    (out / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
