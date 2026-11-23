"""Train one policy, counting real and generated transitions separately."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .config import Multiplier, TrainConfig
from .environment import ContextualBandit, BanditConfig
from .estimators import (
    BRANCH_METHODS,
    MODEL_METHODS,
    check_method,
    group_cost,
    sample_branches,
    score_loss,
)
from .models import Agent


@torch.no_grad()
def evaluate(agent, eye, env):
    p = agent.policy(eye).softmax(-1).double().numpy()
    p /= p.sum(-1, keepdims=True)
    ids = np.arange(len(eye))
    regret = float((p * (ids[:, None] - ids[None, :]) ** 2).sum(-1).mean())
    return {
        "expected_return": -regret,
        "regret": regret,
        "optimal_action_probability": float(p.diagonal().mean()),
        "greedy_accuracy": float((p.argmax(-1) == ids).mean()),
    }


def run(job):
    cfg, ecfg = TrainConfig(**job["training"]), BanditConfig(**job["environment"])
    method = check_method(job["method"])
    M = job.get("M", 4) if method in BRANCH_METHODS else 1
    budget_mode = job.get("budget_mode", "all")
    cost = group_cost(method, M, budget_mode)
    if cfg.budget % cost or cfg.batch_size % cost:
        raise ValueError(
            f"budget and batch_size must be multiples of the group cost {cost}"
        )
    output = Path(job["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed = job["seed"]
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    env = ContextualBandit(ecfg, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    agent = Agent(
        ecfg.num_actions,
        ecfg.feedback_dim,
        "cca" if method == "cca" else "baseline",
        cfg.hidden,
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate)
    multiplier = Multiplier(cfg)
    eye = torch.eye(ecfg.num_actions)
    model = None
    if method in MODEL_METHODS:
        from .world_models import load_model

        model = load_model(job["model_checkpoint"])
        expected_kind = "wm" if method == "rloo_wm" else "cwm"
        if model.kind != expected_kind:
            raise ValueError(f"{method} requires a {expected_kind} checkpoint")
    job = {**job, "method": method, "M": M}
    (output / "config.json").write_text(json.dumps(job, indent=2) + "\n")
    env.save(output / "environment.npz")
    start = time.perf_counter()
    real = generated = roots = updates = charged = 0
    totals, count = {}, 0

    def record():
        return {
            "charged_interactions": charged,
            "environment_interactions": real,
            "model_samples": generated,
            "root_groups": roots,
            "updates": updates,
            "seconds": time.perf_counter() - start,
            **evaluate(agent, eye, env),
        }

    final = record()
    with (output / "progress.jsonl").open("w") as progress:
        progress.write(json.dumps(final) + "\n")
        while charged < cfg.budget:
            group_count = min(cfg.batch_size, cfg.budget - charged) // cost
            contexts = rng.integers(ecfg.num_actions, size=group_count)
            context = eye[contexts]
            if method in BRANCH_METHODS:
                with torch.no_grad():
                    probabilities = agent.policy(context).softmax(-1).numpy()
                batch = sample_branches(
                    env, probabilities, contexts, rng, method, M, model
                )
                actions = torch.from_numpy(batch.actions)
                reward = torch.from_numpy(batch.rewards)
                losses = score_loss(
                    agent,
                    context,
                    actions,
                    reward,
                    branching=True,
                    entropy_coef=cfg.entropy_coef,
                )
                real += batch.environment_interactions
                generated += batch.model_samples
            else:
                noise = rng.normal(0, ecfg.noise_std, size=group_count).astype(np.float32)
                uniforms = rng.random(group_count)
                with torch.no_grad():
                    probabilities = agent.policy(context).softmax(-1).numpy()
                actions_np = (
                    (uniforms[:, None] > probabilities.cumsum(-1))
                    .sum(-1)
                    .clip(max=ecfg.num_actions - 1)
                )
                rewards_np, feedback = env.outcome(contexts, actions_np, noise)
                actions, reward = (
                    torch.from_numpy(actions_np),
                    torch.from_numpy(rewards_np),
                )
                if method == "reinforce":
                    losses = score_loss(
                        agent,
                        context,
                        actions[:, None],
                        reward[:, None],
                        branching=False,
                        entropy_coef=cfg.entropy_coef,
                    )
                else:
                    losses = agent.losses(
                        context,
                        actions,
                        reward,
                        torch.from_numpy(feedback),
                        cfg.entropy_coef,
                    )
                real += group_count
            weight = multiplier.value if method == "cca" else 0.0
            total = (
                losses["policy"]
                + losses["value"]
                + losses["supervised"]
                + weight * losses["independence"]
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            detached = {name: float(value.detach()) for name, value in losses.items()}
            residual = (
                multiplier.update(detached["independence"], detached["entropy"])
                if method == "cca"
                else 0.0
            )
            detached.update(
                multiplier=weight,
                constraint_residual=residual,
                observed_reward=float(reward.mean()),
            )
            for name, value in detached.items():
                totals[name] = totals.get(name, 0.0) + group_count * value
            count += group_count
            roots += group_count
            charged += group_count * cost
            updates += 1
            if updates % cfg.log_every == 0 or charged == cfg.budget:
                final = {
                    **record(),
                    **{name: value / count for name, value in totals.items()},
                }
                progress.write(json.dumps(final, allow_nan=False) + "\n")
                progress.flush()
                totals, count = {}, 0
    torch.save(
        {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "job": job,
            "rng": rng.bit_generator.state,
            "multiplier": {"log_value": multiplier.log_value, "ema": multiplier.ema},
        },
        output / "checkpoint.pt",
    )
    result = {
        "method": method,
        "seed": seed,
        "M": M,
        "budget_mode": budget_mode,
        "noise_std": ecfg.noise_std,
        **final,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=10_000_000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="charged transitions per optimizer update",
    )
    parser.add_argument(
        "--M", type=int, default=4, help="branches including the real root"
    )
    parser.add_argument("--budget-mode", choices=("all", "real"), default="all")
    parser.add_argument("--noise-std", type=float, default=30.0)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.004)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--model-checkpoint", type=Path)
    args = parser.parse_args()
    cfg = TrainConfig(
        budget=args.budget,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        threads=args.threads,
        log_every=args.log_every,
    )
    job = {
        "seed": args.seed,
        "method": args.method,
        "M": args.M,
        "budget_mode": args.budget_mode,
        "environment": asdict(BanditConfig(n=args.n, noise_std=args.noise_std)),
        "training": asdict(cfg),
        "output": str(args.output),
    }
    if args.model_checkpoint:
        job["model_checkpoint"] = str(args.model_checkpoint)
    if check_method(args.method) in MODEL_METHODS and not args.model_checkpoint:
        parser.error("learned methods require --model-checkpoint")
    print(json.dumps(run(job), indent=2))


if __name__ == "__main__":
    main()
