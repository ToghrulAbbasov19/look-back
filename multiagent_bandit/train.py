"""Train the K agent policies, counting real and generated joint episodes."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from .environment import MultiAgentBandit, MultiAgentConfig
from .models import Agents
from .sampling import (
    BRANCH_METHODS,
    METHODS,
    MODEL_METHODS,
    budget_schedule,
    check_method,
    group_cost,
    loo_advantages,
    sample_branches,
)


@dataclass(frozen=True)
class TrainConfig:
    budget: int = 10_000_000
    batch_size: int = 256  # Cumulative charged interactions per optimizer update.
    learning_rate: float = 4e-4
    entropy_coef: float = 0.004
    hidden: int = 32
    beta: float = 0.1
    dual_learning_rate: float = 0.01
    dual_ema: float = 0.99
    initial_multiplier: float = 1.0
    log_every: int = 100
    threads: int = 1


class Multipliers:
    """One CCA Lagrange multiplier per agent."""

    def __init__(self, config, num_agents):
        self.config = config
        self.log_value = np.full(num_agents, np.log(config.initial_multiplier))
        self.ema = None

    @property
    def value(self):
        return np.exp(self.log_value)

    def update(self, kl, entropy):
        cfg = self.config
        residual = kl - cfg.beta * entropy
        self.ema = (
            residual
            if self.ema is None
            else cfg.dual_ema * self.ema + (1 - cfg.dual_ema) * residual
        )
        self.log_value = np.clip(
            self.log_value + cfg.dual_learning_rate * self.ema,
            np.log(1e-8),
            np.log(1e6),
        )
        return residual


@torch.no_grad()
def probabilities_all_contexts(agents, eye):
    p = (
        agents.policy(eye[None].expand(agents.num_agents, -1, -1))
        .softmax(-1)
        .double()
        .numpy()
    )
    return p / p.sum(-1, keepdims=True)


@torch.no_grad()
def evaluate(agents, eye):
    p = probabilities_all_contexts(agents, eye)
    indices = np.arange(len(eye))
    payoff = -((indices[:, None] - indices[None, :]).astype(float) ** 2)
    local = (p * payoff).sum(-1).mean(-1)
    return {
        "expected_return": float(local.mean()),
        "joint_expected_return": float(local.sum()),
        "per_agent_return": local.tolist(),
        "optimal_action_probability": float(p.diagonal(axis1=1, axis2=2).mean()),
        "greedy_accuracy": float((p.argmax(-1) == indices).mean()),
    }


def losses_for_batch(agents, batch, method, eye, feedback, entropy_coef):
    """Per-agent losses: all M branch scores, averaged over groups and branches."""
    method = check_method(method)
    K = agents.num_agents
    context_indices = torch.from_numpy(batch.contexts.reshape(K, -1))
    actions = torch.from_numpy(batch.actions.reshape(K, -1))
    rewards = torch.from_numpy(batch.rewards.reshape(K, -1))
    if method in ("reinforce_baseline", "cca"):
        return agents.losses(
            eye[context_indices], actions, rewards, feedback, entropy_coef
        )
    # Evaluate the policy once per context, then gather the sampled entries.
    logp = agents.policy(eye[None].expand(K, -1, -1)).log_softmax(-1)
    identity = torch.arange(K)[:, None]
    selected = logp[identity, context_indices, actions]
    entropy_table = -(logp.exp() * logp).sum(-1)
    entropy = entropy_table[identity, context_indices].mean(-1)
    if method in BRANCH_METHODS:
        advantages = torch.from_numpy(loo_advantages(batch.rewards).reshape(K, -1))
    else:
        advantages = rewards
    policy = -(selected * advantages.detach()).mean(-1)
    zero = policy.new_zeros(K)
    return {
        "policy": policy - entropy_coef * entropy,
        "value": zero,
        "supervised": zero,
        "independence": zero,
        "entropy": entropy,
        "advantage_mean": advantages.mean(-1),
        "advantage_variance": advantages.var(-1, unbiased=False),
    }


def run(job):
    """Run once into a new/empty directory; never overwrite a previous result."""
    cfg = TrainConfig(**job.get("training", {}))
    env_cfg = MultiAgentConfig(**job.get("environment", {}))
    method = check_method(job["method"])
    seed, K = job.get("seed", 0), env_cfg.num_agents
    M = job.get("M", 4) if method in BRANCH_METHODS else 1
    budget_mode = job.get("budget_mode", "all")
    cost = group_cost(method, K, M, budget_mode, env_cfg.num_actions)
    schedule = tuple(budget_schedule(cfg.budget, cfg.batch_size, cost))
    model_path = job.get("model_checkpoint")
    model = None
    if method in MODEL_METHODS:
        if model_path is None:
            raise ValueError("learned methods require model_checkpoint")
        from .world_models import load_model

        model = load_model(model_path)
    output = Path(job["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    env = MultiAgentBandit(env_cfg, seed)
    root_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    branch_rng = np.random.default_rng(np.random.SeedSequence([seed, 4]))
    model_rng = np.random.default_rng(np.random.SeedSequence([seed, 5]))
    nuisance_rng = np.random.default_rng(np.random.SeedSequence([seed, 6]))
    agents = Agents(
        K, env_cfg.num_actions, env_cfg.feedback_dim, method, cfg.hidden, seed
    )
    optimizer = torch.optim.Adam(agents.parameters(), lr=cfg.learning_rate)
    multipliers = Multipliers(cfg, K)
    eye = torch.eye(env_cfg.num_actions)

    @torch.no_grad()
    def root_action_sampler(contexts, uniforms):
        # Root actions use the float32 policy (the branches use float64 tables).
        context = eye[torch.from_numpy(contexts.T.copy())]
        p = agents.policy(context).softmax(-1).numpy()
        return (
            (uniforms.T[..., None] > p.cumsum(-1))
            .sum(-1)
            .clip(max=env_cfg.num_actions - 1)
            .T.copy()
        )

    run_config = {
        "seed": seed,
        "method": method,
        "M": M,
        "environment": asdict(env_cfg),
        "training": asdict(cfg),
        "budget_mode": budget_mode,
        "group_cost": cost,
    }
    (output / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    env.save(output / "environment.npz")
    started = time.perf_counter()
    record = {
        "charged_interactions": 0,
        "environment_interactions": 0,
        "factual_environment_interactions": 0,
        "counterfactual_environment_interactions": 0,
        "model_samples": 0,
        "root_groups": 0,
        "updates": 0,
        "unused_tail_interactions": 0,
        **evaluate(agents, eye),
    }
    totals, count, interactions, model_samples, completed_groups = {}, 0, 0, 0, 0
    factual_interactions = counterfactual_interactions = 0
    target_groups = cfg.budget // cost
    with (output / "progress.jsonl").open("w") as progress:
        progress.write(json.dumps(record) + "\n")
        for update, groups in enumerate(schedule, 1):
            probabilities = (
                probabilities_all_contexts(agents, eye)
                if method in BRANCH_METHODS
                else None
            )
            batch = sample_branches(
                env,
                probabilities,
                groups,
                method,
                M,
                root_rng,
                branch_rng,
                model=model,
                model_rng=model_rng,
                nuisance_rng=nuisance_rng,
                root_action_sampler=root_action_sampler,
            )
            if method == "cca":
                noise = noise_rng.normal(
                    0,
                    env_cfg.feedback_noise_std,
                    size=(groups, K, env_cfg.feedback_dim),
                ).astype(np.float32)
                feedback = torch.from_numpy(
                    env.feedback(batch.root_contexts, batch.root_actions, noise)
                ).permute(1, 0, 2)
            else:
                feedback = None
            losses = losses_for_batch(
                agents, batch, method, eye, feedback, cfg.entropy_coef
            )
            weights = multipliers.value if method == "cca" else np.zeros(K)
            total = (
                losses["policy"]
                + losses["value"]
                + losses["supervised"]
                + torch.as_tensor(weights, dtype=torch.float32) * losses["independence"]
            )
            optimizer.zero_grad(set_to_none=True)
            total.sum().backward()
            optimizer.step()
            values = {
                key: value.detach().numpy().astype(float)
                for key, value in losses.items()
            }
            residual = (
                multipliers.update(values["independence"], values["entropy"])
                if method == "cca"
                else np.zeros(K)
            )
            values.update(multiplier=weights, constraint_residual=residual)
            for key, value in values.items():
                totals[key] = totals.get(key, np.zeros(K)) + groups * value
            count += groups
            interactions += batch.environment_interactions
            model_samples += batch.model_samples
            factual_interactions += batch.factual_environment_interactions
            counterfactual_interactions += batch.counterfactual_environment_interactions
            completed_groups += groups
            if update % cfg.log_every == 0 or completed_groups == target_groups:
                charged = interactions + (model_samples if budget_mode == "all" else 0)
                record = {
                    "charged_interactions": charged,
                    "environment_interactions": interactions,
                    "model_samples": model_samples,
                    "factual_environment_interactions": factual_interactions,
                    "counterfactual_environment_interactions": counterfactual_interactions,
                    "root_groups": completed_groups,
                    "unused_tail_interactions": 0,
                    "updates": update,
                    **evaluate(agents, eye),
                    **{
                        key: float((value / count).mean())
                        for key, value in totals.items()
                    },
                }
                if method == "cca":
                    record["per_agent_multiplier"] = (
                        totals["multiplier"] / count
                    ).tolist()
                    record["per_agent_independence"] = (
                        totals["independence"] / count
                    ).tolist()
                progress.write(json.dumps(record, allow_nan=False) + "\n")
                progress.flush()
                totals, count = {}, 0
        tail = cfg.budget - target_groups * cost
        if tail:
            # Groups are whole, so spend the remainder on episodes that are not trained on.
            discarded = sample_branches(
                env,
                None,
                tail,
                "reinforce",
                1,
                root_rng,
                branch_rng,
                root_action_sampler=root_action_sampler,
            )
            interactions += discarded.environment_interactions
            factual_interactions += discarded.factual_environment_interactions
            record.update(
                charged_interactions=cfg.budget,
                environment_interactions=interactions,
                factual_environment_interactions=factual_interactions,
                counterfactual_environment_interactions=counterfactual_interactions,
                unused_tail_interactions=tail,
            )
            progress.write(json.dumps(record, allow_nan=False) + "\n")
            progress.flush()
    charged = interactions + (model_samples if budget_mode == "all" else 0)
    torch.save(
        {
            "agents": agents.state_dict(),
            "optimizer": optimizer.state_dict(),
            "job": run_config,
            "root_rng": root_rng.bit_generator.state,
            "branch_rng": branch_rng.bit_generator.state,
            "noise_rng": noise_rng.bit_generator.state,
            "model_rng": model_rng.bit_generator.state,
            "nuisance_rng": nuisance_rng.bit_generator.state,
            "multipliers": {"log_value": multipliers.log_value, "ema": multipliers.ema},
        },
        output / "checkpoint.pt",
    )
    result = {**run_config, "seconds": time.perf_counter() - started, **record}
    (output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument(
        "--M", type=int, default=4, help="actions per group, including the root"
    )
    parser.add_argument("--budget", type=int, default=10_000_000)
    parser.add_argument(
        "--budget-mode",
        choices=("all", "real"),
        default="all",
        help="count generated episodes too, or only simulator calls",
    )
    parser.add_argument(
        "--model-checkpoint", type=Path, help="WM or CWM, for rloo_wm / rlb_cwm"
    )
    args = parser.parse_args()
    job = {
        "method": args.method,
        "output": str(args.output),
        "seed": args.seed,
        "M": args.M,
        "budget_mode": args.budget_mode,
        "model_checkpoint": str(args.model_checkpoint)
        if args.model_checkpoint
        else None,
        "training": asdict(TrainConfig(budget=args.budget)),
        "environment": asdict(MultiAgentConfig(num_agents=args.num_agents)),
    }
    print(json.dumps(run(job), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
