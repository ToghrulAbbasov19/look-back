"""Train REINFORCE, REINFORCE + baseline, RLOO, or RLB in the simulator.

All four use the same roots and policy; RLOO and RLB roll out groups that share a
root, RLB additionally sharing the future simulator noise.
"""

import argparse
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from action_repeat import frames_to_decisions, make_repeat_pixel_envs
from common import DEVICE, DINO_RL_DIR, envs
from estimators import (
    leave_one_out_advantages,
    leave_one_out_loss,
    reinforce_loss,
)
from policy_core import (
    IMAGE_SIZE,
    K,
    make_pixel_envs,
    make_policy,
    make_value,
)
from policy_rollouts import ParallelEnvs, collect_group, collect_group_parallel
from policy_roots import (
    parse_root_speed_bins,
    parse_root_ttc_bins,
)


def auto_validate_checkpoint(
    results_dir: str,
    results_root: str,
    log_dir: str,
    checkpoint: int,
    action_repeat: int,
    episodes: int,
    episode_seed: int,
    env_overrides: dict[str, str],
) -> None:
    """Run greedy validation after saving a checkpoint."""
    os.makedirs(log_dir, exist_ok=True)
    arm, seed_text = results_dir.rsplit("_s", 1)
    try:
        run_seed = int(seed_text)
    except ValueError as exc:
        raise ValueError(
            "--auto-validate requires --results-dir ending in _s<seed>"
        ) from exc
    child_env = os.environ.copy()
    child_env.update(env_overrides)
    child_env["DINO_RESULTS_ROOT"] = results_root
    child_env["OMP_NUM_THREADS"] = "2"
    log_path = os.path.join(
        log_dir, f"{arm}_s{run_seed}_c{checkpoint}_e{episode_seed}.log"
    )
    command = [
        sys.executable,
        "-u",
        os.path.join(DINO_RL_DIR, "evaluate_policy.py"),
        arm,
        str(checkpoint),
        str(action_repeat),
        str(episodes),
        str(run_seed),
        str(episode_seed),
    ]
    print(f"[auto-validation] checkpoint {checkpoint}, {episodes} episodes -> {log_path}")
    with open(log_path, "w") as output:
        proc = subprocess.run(
            command,
            cwd=DINO_RL_DIR,
            env=child_env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode:
        print(f"[auto-validation] FAILED (exit {proc.returncode}); see {log_path}")
    else:
        print(f"[auto-validation] complete: {log_path}")


@dataclass(frozen=True)
class TrainingConfig:

    method: str
    group_size: int
    action_repeat: int = 1
    n_updates: int = 1200
    batch: int = 32
    lr: float = 1e-2
    value_lr: float = 3e-2
    gamma: float = 0.99
    entropy_coef: float = 0.03
    max_grad_norm: float = 20.0
    num_workers: int = 0
    max_steps: int = 250
    save_every: int = 25
    results_dir: str | None = None
    collision_penalty: float = -1.0
    seed: int | None = None
    root_ttc_bins: tuple = ()
    root_speed_bins: tuple = ()
    auto_validate: bool = False
    auto_validate_episodes: int = 30
    auto_validate_episode_seed: int = 2_200_000_000
    output_root: str | None = None
    validation_log_dir: str | None = None


def train(config: TrainingConfig):
    method = config.method
    group_size = config.group_size
    action_repeat = config.action_repeat
    n_updates = config.n_updates
    batch = config.batch
    lr = config.lr
    value_lr = config.value_lr
    gamma = config.gamma
    entropy_coef = config.entropy_coef
    max_grad_norm = config.max_grad_norm
    num_workers = config.num_workers
    max_steps = config.max_steps
    save_every = config.save_every
    results_dir = config.results_dir
    collision_penalty = config.collision_penalty
    seed = config.seed
    root_ttc_bins = config.root_ttc_bins
    root_speed_bins = config.root_speed_bins
    auto_validate = config.auto_validate
    auto_validate_episodes = config.auto_validate_episodes
    auto_validate_episode_seed = config.auto_validate_episode_seed
    output_root = config.output_root
    validation_log_dir = config.validation_log_dir

    assert method in {"reinforce", "reinforce_baseline", "rloo", "rlb"}
    grouped = method in {"rloo", "rlb"}
    if grouped:
        assert group_size >= 2 and batch % group_size == 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(
        f"deterministic training: seed={seed}, CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}"
    )
    rng = np.random.default_rng(seed)
    results_dir = results_dir or f"pixel_{method}"
    if action_repeat > 1:
        _frames = max_steps
        max_steps = frames_to_decisions(_frames, action_repeat)
        print(
            f"ACTION REPEAT k={action_repeat}: {_frames} game frames -> "
            f"{max_steps} decisions/episode"
        )
    independent_future_noise = method == "rloo"
    if method == "rloo":
        print(
            "RLOO: group members share the root and use independent future simulator noise"
        )
    elif method == "rlb":
        print("RLB: group members share the root and future simulator noise")
    if grouped:
        print(f"groups: {batch // group_size} x {group_size}")
    label = ", ".join(f"{lo}-{hi}" for lo, hi in root_ttc_bins)
    print(f"standing TTC roots: {label}; balanced speeds: {root_speed_bins}")
    print("initial frame stack: four decision frames ending at the root")

    print(
        f"=== {method}  (PIXELS: {K}x{IMAGE_SIZE[1]}x{IMAGE_SIZE[0]}, "
        f"action repeat {action_repeat})  "
        f"batch={batch} lr={lr} ==="
    )
    penvs = None
    if num_workers:
        penvs = ParallelEnvs(
            batch,
            n_workers=num_workers,
            game_mode="train",
            action_repeat=action_repeat,
        )
        gym_envs = []
        n_actions = 3
        print(
            f"parallel env stepping: {penvs.n_workers} worker processes for {batch} envs"
        )
    else:
        if action_repeat > 1:
            gym_envs, _ = make_repeat_pixel_envs(
                batch, game_mode="train", k=action_repeat
            )
        else:
            gym_envs = make_pixel_envs(batch, game_mode="train")
        n_actions = gym_envs[0].action_space.n

    policy = make_policy(K, n_actions, orthogonal=True).to(DEVICE)
    print("init: orthogonal (head gain 0.01)")

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    use_value = method == "reinforce_baseline"
    value_network = make_value(K, orthogonal=True).to(DEVICE) if use_value else None
    value_optimizer = (
        torch.optim.Adam(value_network.parameters(), lr=value_lr)
        if use_value
        else None
    )
    output_root = output_root or os.path.join(DINO_RL_DIR, "runs", "checkpoints")
    validation_log_dir = validation_log_dir or os.path.join(
        DINO_RL_DIR, "runs", "validation"
    )
    out_dir = os.path.join(output_root, results_dir)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise RuntimeError(
            f"refusing to overwrite nonempty results directory {out_dir}"
        )
    os.makedirs(out_dir, exist_ok=True)
    validation_env = {
        "PIXEL_EVAL_BASE_CACTUS_SPAWN_PROB": str(envs.BASE_CACTUS_SPAWN_PROB),
        "PIXEL_EVAL_MAX_CACTUS_SPAWN_PROB": str(envs.MAX_CACTUS_SPAWN_PROB),
        "PIXEL_EVAL_BIRD_SPAWN_PROB": str(envs.BIRD_SPAWN_PROB),
        "PIXEL_EVAL_BIRD_MIN_SPEED": str(envs.BIRD_MIN_SPEED),
        "PIXEL_EVAL_BIRD_Y_POSITIONS": ",".join(map(str, envs.BIRD_Y_POSITIONS)),
    }
    if auto_validate:
        print(
            "automatic validation: every saved checkpoint; "
            f"{auto_validate_episodes} frame-0 episodes, "
            f"episode seed={auto_validate_episode_seed}; environment matches training overrides"
        )

    reward_hist, entropy_hist, grad_hist, spread_hist = [], [], [], []
    t_start = time.time()

    for update_i in range(n_updates):
        if penvs is not None:
            episodes = collect_group_parallel(
                penvs,
                policy,
                grouped=grouped,
                group_size=group_size,
                collision_penalty=collision_penalty,
                max_steps=max_steps,
                rng=rng,
                n=batch,
                root_ttc_bins=root_ttc_bins,
                independent_future_noise=independent_future_noise,
                root_speed_bins=root_speed_bins,
            )
        else:
            episodes = collect_group(
                gym_envs,
                policy,
                grouped=grouped,
                group_size=group_size,
                collision_penalty=collision_penalty,
                max_steps=max_steps,
                rng=rng,
                root_ttc_bins=root_ttc_bins,
                independent_future_noise=independent_future_noise,
                root_speed_bins=root_speed_bins,
            )

        policy_optimizer.zero_grad()
        if use_value:
            value_optimizer.zero_grad()

        if grouped:  # RLOO / RLB
            advantages = leave_one_out_advantages(
                episodes, gamma, group_size
            )
            loss, entropy_sum, _ = leave_one_out_loss(
                episodes, advantages, entropy_coef, DEVICE
            )
        else:
            loss, entropy_sum = reinforce_loss(
                episodes,
                gamma,
                entropy_coef,
                DEVICE,
                value_network=value_network,
            )

        loss.backward()
        grad_norm = torch.sqrt(
            sum(
                (parameter.grad.detach() ** 2).sum()
                for parameter in policy.parameters()
                if parameter.grad is not None
            )
        ).item()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=max_grad_norm)
        policy_optimizer.step()
        if use_value:
            value_optimizer.step()

        rewards = np.array([episode.total_reward for episode in episodes])
        mean_entropy = entropy_sum / len(episodes)
        reward_hist.append(float(rewards.mean()))
        entropy_hist.append(mean_entropy)
        grad_hist.append(grad_norm)
        spread_hist.append(float(rewards.std()))

        if update_i % 5 == 0 or update_i == n_updates - 1:
            done = (update_i + 1) * batch
            el = time.time() - t_start
            eta = (el / (update_i + 1)) * (n_updates - update_i - 1) / 3600
            print(
                f"[{method}] upd {update_i:5d}  ep {done:7d}  avg_reward={rewards.mean():7.2f}  "
                f"entropy={mean_entropy:.4f}  |g|={grad_norm:8.2f}"
                f"{'*' if grad_norm > max_grad_norm else ' '}  "
                f"{el / max(update_i + 1, 1):5.1f}s/upd  ETA {eta:4.1f}h"
            )

        if (update_i + 1) % save_every == 0 or update_i + 1 == n_updates:
            torch.save(
                policy.state_dict(), os.path.join(out_dir, f"policy-{update_i + 1}.pth")
            )
            if use_value:
                torch.save(
                    value_network.state_dict(),
                    os.path.join(out_dir, f"value-{update_i + 1}.pth"),
                )
            np.save(os.path.join(out_dir, "reward_history.npy"), np.array(reward_hist))
            np.save(
                os.path.join(out_dir, "entropy_history.npy"), np.array(entropy_hist)
            )
            np.save(os.path.join(out_dir, "gradnorm_history.npy"), np.array(grad_hist))
            np.save(
                os.path.join(out_dir, "group_spread_history.npy"), np.array(spread_hist)
            )
            if auto_validate:
                auto_validate_checkpoint(
                    results_dir,
                    output_root,
                    validation_log_dir,
                    update_i + 1,
                    action_repeat,
                    auto_validate_episodes,
                    auto_validate_episode_seed,
                    validation_env,
                )

    el = time.time() - t_start
    print(
        f"\ndone: {n_updates} updates in {el / 3600:.2f} h ({el / n_updates:.1f} s/update)"
    )
    if penvs is not None:
        penvs.close()
    for e in gym_envs:
        e.close()
    return reward_hist


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("reinforce", "reinforce_baseline", "rloo", "rlb"),
        required=True,
    )
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--value-lr", type=float, default=3e-2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coef", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=20.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=80,
        help="raw game-frame budget before division by action repeat",
    )
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--root-ttc-bins", default="8-12")
    parser.add_argument("--root-speed-bins", default="21-26,27-31,32-37,38-43")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--output-root",
        default=None,
        help="checkpoint root (default: runs/checkpoints)",
    )
    parser.add_argument(
        "--validation-log-dir",
        default=None,
        help="automatic-validation log directory (default: runs/validation)",
    )
    parser.add_argument("--collision-penalty", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bird-spawn-prob", type=float, default=0.6)
    parser.add_argument("--bird-min-speed", type=float, default=20.0)
    parser.add_argument("--mid-bird-weight", type=int, default=2)
    parser.add_argument("--auto-validate", action="store_true")
    parser.add_argument("--auto-validate-episodes", type=int, default=30)
    parser.add_argument("--auto-validate-episode-seed", type=int, default=2_200_000_000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_groups < 1 or args.group_size < 1:
        parser.error("--num-groups and --group-size must be positive")
    try:
        ttc_bins = parse_root_ttc_bins(args.root_ttc_bins)
        speed_bins = parse_root_speed_bins(args.root_speed_bins)
    except ValueError as exc:
        parser.error(str(exc))
    if args.mid_bird_weight < 1:
        parser.error("--mid-bird-weight must be at least 1")

    envs.BIRD_SPAWN_PROB = args.bird_spawn_prob
    envs.BIRD_MIN_SPEED = args.bird_min_speed
    low, mid, high = envs.BIRD_Y_POSITIONS[:3]
    envs.BIRD_Y_POSITIONS = (low, *([mid] * args.mid_bird_weight), high)
    batch = args.num_groups * args.group_size
    print(f"batch: {args.num_groups} groups x {args.group_size} = {batch}")

    config = TrainingConfig(
        method=args.method,
        group_size=args.group_size,
        action_repeat=args.action_repeat,
        n_updates=args.updates,
        batch=batch,
        lr=args.lr,
        value_lr=args.value_lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        max_steps=args.max_steps,
        save_every=args.save_every,
        results_dir=args.results_dir,
        collision_penalty=args.collision_penalty,
        seed=args.seed,
        root_ttc_bins=ttc_bins,
        root_speed_bins=speed_bins,
        auto_validate=args.auto_validate,
        auto_validate_episodes=args.auto_validate_episodes,
        auto_validate_episode_seed=args.auto_validate_episode_seed,
        output_root=args.output_root,
        validation_log_dir=args.validation_log_dir,
    )
    train(config)


if __name__ == "__main__":
    main()
