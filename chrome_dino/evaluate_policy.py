"""Evaluate greedy policies on deterministic simulator episodes."""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from collections import deque

import numpy as np
import torch

from action_repeat import make_repeat_pixel_envs
from common import DEVICE, envs
from policy_core import K, make_policy, stack_to_tensor, to_gray


def _float_override(name):
    value = os.environ.get(name)
    return None if value is None else float(value)


def _apply_environment_overrides():
    for name in (
        "BASE_CACTUS_SPAWN_PROB",
        "MAX_CACTUS_SPAWN_PROB",
        "BIRD_SPAWN_PROB",
        "BIRD_MIN_SPEED",
    ):
        value = _float_override(f"PIXEL_EVAL_{name}")
        if value is not None:
            setattr(envs, name, value)
    positions = os.environ.get("PIXEL_EVAL_BIRD_Y_POSITIONS")
    if positions:
        envs.BIRD_Y_POSITIONS = tuple(int(x) for x in positions.split(","))
    if any(key.startswith("PIXEL_EVAL_") for key in os.environ):
        print(
            "evaluation environment: "
            f"cactus={envs.BASE_CACTUS_SPAWN_PROB}->{envs.MAX_CACTUS_SPAWN_PROB}, "
            f"bird={envs.BIRD_SPAWN_PROB}, min_speed={envs.BIRD_MIN_SPEED}, "
            f"bird_y={envs.BIRD_Y_POSITIONS}"
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", help="checkpoint directory prefix")
    parser.add_argument("checkpoint")
    parser.add_argument("action_repeat", type=int)
    parser.add_argument("episodes", type=int, nargs="?", default=10)
    parser.add_argument("run_seeds", nargs="?", default="1,2,3")
    parser.add_argument("first_episode_seed", type=int, nargs="?", default=1000)
    return parser


def evaluate_seed(arm, checkpoint, action_repeat, episodes, run_seed, first_seed):
    results_root = os.environ.get(
        "DINO_RESULTS_ROOT", os.path.join(HERE, "runs", "checkpoints")
    )
    path = os.path.join(
        results_root, f"{arm}_s{run_seed}", f"policy-{checkpoint}.pth"
    )
    if not os.path.exists(path):
        print(f"  {arm}_s{run_seed}: no policy-{checkpoint}")
        return

    policy = make_policy(K, 3)
    policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    policy.to(DEVICE).eval()
    gym_envs, raw_envs = make_repeat_pixel_envs(
        1, game_mode="normal", k=action_repeat
    )
    env, raw = gym_envs[0], raw_envs[0]
    low, mid, *_, high = envs.BIRD_Y_POSITIONS
    stats = {"cactus": [0, 0], "LOW": [0, 0], "MID": [0, 0], "HIGH": [0, 0]}
    lengths, action_counts = [], np.zeros(3, dtype=int)
    episode_seed = first_seed
    cap = 1500 // action_repeat

    try:
        while len(lengths) < episodes:
            observation, _ = env.reset(seed=episode_seed)
            episode_seed += 1
            stack = deque([to_gray(observation)] * K, maxlen=K)
            steps, done = 0, False
            while steps < cap and not done:
                with torch.no_grad():
                    action = int(
                        torch.argmax(policy(stack_to_tensor(stack).unsqueeze(0))[0])
                    )
                action_counts[action] += 1
                pending = [
                    obstacle
                    for obstacle in raw._obstacles
                    if obstacle.needs_collision_check
                ]
                observation, reward, done, *_ = env.step(envs.Action(action))
                stack.append(to_gray(observation))
                steps += 1
                for obstacle in pending:
                    if obstacle.needs_collision_check:
                        continue
                    kind = (
                        "cactus"
                        if not isinstance(obstacle, envs.Bird)
                        else {low: "LOW", mid: "MID", high: "HIGH"}.get(
                            obstacle.rect.y, "?"
                        )
                    )
                    stats[kind][1] += 1
                    stats[kind][0] += reward >= 0
            lengths.append(steps)
    finally:
        env.close()

    passed = sum(value[0] for value in stats.values())
    total = sum(value[1] for value in stats.values())

    def count(kind):
        return f"{stats[kind][0]}/{stats[kind][1]}"

    action_percent = 100 * action_counts / action_counts.sum()
    print(
        f"  {arm + '_s' + str(run_seed):14s} ckpt={checkpoint}  "
        f"all {100 * passed / max(total, 1):5.1f}%  cactus {count('cactus'):>6s}  "
        f"LOW {count('LOW'):>5s}  MID {count('MID'):>5s}  "
        f"HIGH {count('HIGH'):>5s}  decisions {np.mean(lengths):5.0f}  "
        f"S/J/D {action_percent[0]:.0f}/{action_percent[1]:.0f}/{action_percent[2]:.0f}  "
        f"({episodes} episodes from seed {first_seed})",
        flush=True,
    )


def main():
    args = build_parser().parse_args()
    _apply_environment_overrides()
    arm = args.method
    for run_seed in (int(value) for value in args.run_seeds.split(",")):
        evaluate_seed(
            arm,
            args.checkpoint,
            args.action_repeat,
            args.episodes,
            run_seed,
            args.first_episode_seed,
        )


if __name__ == "__main__":
    main()
