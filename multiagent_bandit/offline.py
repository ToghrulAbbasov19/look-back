"""Collect paired joint episodes and fit the per-agent WM and CWM of the joint cost."""

import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .environment import MultiAgentBandit, MultiAgentConfig
from .world_models import DensityConfig, TransitionDensity, density_checkpoint


def joint_actions(probabilities, contexts, uniforms):
    """Inverse-CDF draws under frozen independent local policies."""
    conditional = probabilities[np.arange(contexts.shape[-1]), contexts]
    return (
        (uniforms[..., None] > conditional.cumsum(-1))
        .sum(-1)
        .clip(max=conditional.shape[-1] - 1)
    )


def read_reference(path):
    with np.load(path, allow_pickle=False) as saved:
        probabilities = np.asarray(saved["probabilities"], dtype=np.float64)
    probabilities /= probabilities.sum(-1, keepdims=True)
    return probabilities


def collect_dataset(
    output,
    reference,
    *,
    seed=0,
    n=10,
    random_fraction=0.5,
    train_pairs=4096,
    validation_pairs=1024,
    test_pairs=2048,
):
    """One root joint episode plus, for each agent i, the episode with only i's action changed.

    That gives K pairs per root for K+1 simulator calls. Actions come from a
    50:50 mix of the reference policy and a uniform policy. Splits are by root.
    """
    probabilities = read_reference(reference)
    K, S = probabilities.shape[:2]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mixture = (1 - random_fraction) * probabilities + random_fraction / S
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4101]))
    count = train_pairs + validation_pairs + test_pairs
    contexts = rng.integers(S, size=(count, K), dtype=np.int64)
    root_actions = joint_actions(mixture, contexts, rng.random((count, K)))
    alternative_actions = joint_actions(mixture, contexts, rng.random((count, K)))
    environment = MultiAgentBandit(MultiAgentConfig(num_agents=K, n=n), seed)
    root_costs = -environment.reward(contexts, root_actions)
    alternative_costs = np.empty((count, K), dtype=np.float32)
    for i in range(K):
        intervened = root_actions.copy()
        intervened[:, i] = alternative_actions[:, i]
        alternative_costs[:, i] = -environment.reward(contexts, intervened)
    split = np.empty(count, dtype=np.int8)
    order = rng.permutation(count)
    split[order[:train_pairs]] = 0
    split[order[train_pairs : train_pairs + validation_pairs]] = 1
    split[order[train_pairs + validation_pairs :]] = 2
    arrays = {
        "contexts": contexts,
        "root_actions": root_actions,
        "alternative_actions": alternative_actions,
        "root_costs": root_costs,
        "alternative_costs": alternative_costs,
        "split": split,
        "reference_probabilities": probabilities,
        "mixture_probabilities": mixture,
    }
    path = output / "dataset.npz"
    np.savez_compressed(path, **arrays)
    train = split == 0
    costs = np.concatenate(
        (
            np.broadcast_to(root_costs[train, None], (train_pairs, K)).ravel(),
            alternative_costs[train].ravel(),
        )
    ).astype(np.float64)
    delta = alternative_costs[train].astype(np.float64) - root_costs[train, None]
    metadata = {
        "seed": seed,
        "n": n,
        "num_agents": K,
        "random_fraction": random_fraction,
        "train_pairs": train_pairs,
        "validation_pairs": validation_pairs,
        "test_pairs": test_pairs,
        "environment_interactions": count * (K + 1),
        "cost_center": float(costs.mean()),
        "cost_scale": max(float(costs.std()), 0.05),
        "delta_scale": max(float(np.sqrt(np.mean(delta**2))), 0.05),
    }
    (output / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_dataset(path):
    path = Path(path)
    path = path / "dataset.npz" if path.is_dir() else path
    metadata = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    return arrays, metadata


def expand_pairs(arrays, split, kind):
    mask = arrays["split"] == split
    c, ra, aa = [
        arrays[key][mask].T
        for key in ("contexts", "root_actions", "alternative_actions")
    ]
    rz = np.broadcast_to(arrays["root_costs"][mask], c.shape)
    az = arrays["alternative_costs"][mask].T
    result = {
        "contexts": np.concatenate((c, c), -1),
        "actions": np.concatenate((ra, aa), -1),
        "costs": np.concatenate((rz, az), -1),
        "root_group_ids": np.tile(np.flatnonzero(mask), 2),
    }
    if kind == "counterfactual":
        result.update(
            root_contexts=np.concatenate((c, c), -1),
            root_actions=np.concatenate((aa, ra), -1),
            root_costs=np.concatenate((az, rz), -1),
        )
    return {
        key: torch.from_numpy(np.array(value, copy=True))
        for key, value in result.items()
    }


def model_kwargs(examples, index=slice(None)):
    return {
        key: value[:, index]
        for key, value in examples.items()
        if key in ("root_contexts", "root_actions", "root_costs")
    }


def true_marginal_moments(probabilities):
    """Exact per-agent cost moments under the collection policies (evaluation only)."""
    S = probabilities.shape[-1]
    q = (np.arange(S)[:, None] - np.arange(S)[None, :]) ** 2
    means = (probabilities * q).sum((-1, -2)) / S
    seconds = (probabilities * q**2).sum((-1, -2)) / S
    return means, np.maximum(0.0, seconds - means**2)


@torch.no_grad()
def evaluate_density(model, examples, *, probabilities, max_examples=2048):
    """Held-out NLL and conditional-mean RMSE against the true conditional mean.

    The true mean comes from the simulator equations and is used only here, never in training.
    """
    model.eval()
    count = examples["actions"].shape[1]
    index = torch.linspace(0, count - 1, min(count, max_examples)).round().long()
    c, a, target = [examples[key][:, index] for key in ("contexts", "actions", "costs")]
    kwargs = model_kwargs(examples, index)
    mean, _ = model.moments(c, a, **kwargs)
    nll = model.negative_log_likelihood(c, a, target, **kwargs)
    local_cost = (c.double() - a.double()).square()
    if model.config.kind == "world":
        own_means, _ = true_marginal_moments(probabilities)
        true_mean = local_cost + torch.from_numpy(own_means.sum() - own_means)[:, None]
    else:
        true_mean = (
            kwargs["root_costs"].double()
            + local_cost
            - (
                kwargs["root_contexts"].double() - kwargs["root_actions"].double()
            ).square()
        )
    return {
        "nll": float(nll.mean()),
        "conditional_mean_rmse": float((mean - true_mean).square().mean().sqrt()),
    }


def write_rows(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_density(
    dataset,
    output,
    *,
    kind,
    seed=0,
    steps=5000,
    batch_size=256,
    learning_rate=3e-3,
    final_learning_rate=3e-5,
    hidden=16,
    min_scale=0.05,
    log_every=100,
    threads=1,
):
    torch.set_num_threads(threads)
    arrays, metadata = load_dataset(dataset)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    examples = {
        name: expand_pairs(arrays, index, kind)
        for index, name in enumerate(("train", "validation", "test"))
    }
    config = DensityConfig(
        kind=kind,
        n=metadata["n"],
        num_agents=metadata["num_agents"],
        hidden=hidden,
        cost_center=metadata["cost_center"],
        cost_scale=metadata["cost_scale"],
        delta_scale=metadata["delta_scale"],
        min_scale=min_scale,
        seed=seed,
    )
    model = TransitionDensity(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed + 90131)
    log_steps = (
        {0, 1, 2, 5, 10, 20, 50, steps} | set(range(log_every, steps + 1, log_every))
    ) & set(range(steps + 1))
    rows, train = [], examples["train"]
    start, train_seconds, last_loss = time.perf_counter(), 0.0, None
    for step in range(steps + 1):
        if step in log_steps:
            metrics = evaluate_density(
                model,
                examples["validation"],
                probabilities=arrays["mixture_probabilities"],
            )
            rows.append(
                {
                    "step": step,
                    "wall_seconds": time.perf_counter() - start,
                    "optimizer_seconds": train_seconds,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "last_minibatch_nll": last_loss,
                    **metrics,
                }
            )
        if step == steps:
            break
        tick = time.perf_counter()
        progress = step / max(steps - 1, 1)
        lr = final_learning_rate + 0.5 * (learning_rate - final_learning_rate) * (
            1 + math.cos(math.pi * progress)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        # All agents use the same root indices; their parameters and losses stay separate.
        index = torch.randint(
            train["contexts"].shape[1], (batch_size,), generator=generator
        )
        losses = model.negative_log_likelihood(
            train["contexts"][:, index],
            train["actions"][:, index],
            train["costs"][:, index],
            **model_kwargs(train, index),
        ).mean(-1)
        loss = losses.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_seconds += time.perf_counter() - tick
        last_loss = float(loss.detach() / config.num_agents)
    model.eval().requires_grad_(False)
    final_metrics = {
        split: evaluate_density(
            model,
            data,
            probabilities=arrays["mixture_probabilities"],
            max_examples=data["actions"].shape[1],
        )
        for split, data in examples.items()
    }
    run = {
        "kind": kind,
        "seed": seed,
        "optimizer_steps": steps,
        "batch_size_per_agent": batch_size,
        "environment": {"n": config.n, "num_agents": config.num_agents},
        "split_root_groups": {
            name: metadata[f"{name}_pairs"] for name in ("train", "validation", "test")
        },
        "learning_rate": learning_rate,
        "final_learning_rate": final_learning_rate,
        "min_scale": min_scale,
        "parameter_count_per_agent": model.parameter_count,
        "offline_environment_interactions": metadata["environment_interactions"],
        "wall_seconds": time.perf_counter() - start,
        "metrics": final_metrics,
    }
    torch.save(density_checkpoint(model, training=run), output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(run, indent=2) + "\n")
    write_rows(output / "learning_curve.csv", rows)
    return run
