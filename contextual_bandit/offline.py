"""Collect paired transitions and fit the small Gaussian WM and CWM."""

import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .environment import ContextualBandit, BanditConfig
from .world_models import DensityConfig, TransitionDensity, density_checkpoint


def read_reference(path):
    with np.load(path, allow_pickle=False) as saved:
        probabilities = np.asarray(saved["probabilities"], dtype=np.float64)
    return probabilities / probabilities.sum(-1, keepdims=True)


def collect_dataset(
    output,
    reference,
    *,
    seed=0,
    n=10,
    noise_std=10.0,
    random_fraction=0.5,
    train_pairs=4096,
    validation_pairs=1024,
    test_pairs=2048,
):
    """Collect pairs of transitions that share a context and the noise u.

    Both actions are independent draws from a 50:50 mix of the reference policy
    and a uniform policy. Splits are assigned to whole pairs.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    probabilities = read_reference(reference)
    mixture = (1 - random_fraction) * probabilities + random_fraction / (2 * n + 1)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4101]))
    pair_count = train_pairs + validation_pairs + test_pairs
    contexts = rng.integers(0, 2 * n + 1, size=pair_count, dtype=np.int64)
    cumulative = mixture[contexts].cumsum(-1)
    uniforms = rng.random((pair_count, 2))
    actions = (uniforms[..., None] > cumulative[:, None, :]).sum(-1).astype(np.int64)
    actions = np.minimum(actions, 2 * n)
    noise = rng.normal(0.0, noise_std, pair_count).astype(np.float32)
    env = ContextualBandit(BanditConfig(n=n, noise_std=noise_std), seed)
    next_states = np.stack(
        [env.transition(contexts, actions[:, j], noise).next_state for j in range(2)], -1
    )
    split = np.empty(pair_count, dtype=np.int8)
    order = rng.permutation(pair_count)
    split[order[:train_pairs]] = 0
    split[order[train_pairs : train_pairs + validation_pairs]] = 1
    split[order[train_pairs + validation_pairs :]] = 2
    # The noise is not saved: the models never see it.
    arrays = {
        "contexts": contexts,
        "actions": actions,
        "next_states": next_states,
        "split": split,
        "reference_probabilities": probabilities,
        "mixture_probabilities": mixture,
    }
    path = output / "dataset.npz"
    np.savez_compressed(path, **arrays)
    training_next_states = next_states[split == 0].astype(np.float64)
    metadata = {
        "seed": seed,
        "n": n,
        "noise_std": noise_std,
        "random_fraction": random_fraction,
        "train_pairs": train_pairs,
        "validation_pairs": validation_pairs,
        "test_pairs": test_pairs,
        "environment_interactions": 2 * pair_count,
        "next_state_center": float(training_next_states.mean()),
        "next_state_scale": max(float(training_next_states.std()), 0.05),
    }
    (output / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_dataset(path):
    path = Path(path)
    path = path / "dataset.npz" if path.is_dir() else path
    metadata = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as saved:
        arrays = {
            k: saved[k].copy() for k in ("contexts", "actions", "next_states", "split")
        }
    return arrays, metadata


def expand_pairs(arrays, split, kind):
    """WM: both transitions as independent examples. CWM: each pair in both directions."""
    mask = arrays["split"] == split
    pair_ids = np.flatnonzero(mask)
    c, a, y = (arrays[key][mask] for key in ("contexts", "actions", "next_states"))
    examples = {
        "contexts": np.tile(c, 2),
        "actions": np.concatenate([a[:, 0], a[:, 1]]),
        "next_states": np.concatenate([y[:, 0], y[:, 1]]),
        "pair_ids": np.tile(pair_ids, 2),
    }
    if kind == "counterfactual":
        examples.update(
            root_contexts=np.tile(c, 2),
            root_actions=np.concatenate([a[:, 1], a[:, 0]]),
            root_next_states=np.concatenate([y[:, 1], y[:, 0]]),
        )
    return {key: torch.from_numpy(value) for key, value in examples.items()}


def model_kwargs(examples, index=slice(None)):
    return {
        key: value[index]
        for key, value in examples.items()
        if key in ("root_contexts", "root_actions", "root_next_states")
    }


@torch.no_grad()
def evaluate_density(model, examples):
    """Held-out NLL and conditional-mean RMSE against the true conditional mean.

    The true mean comes from the simulator equations and is used only here, never in training.
    """
    model.eval()
    mean, _ = model(examples["contexts"], examples["actions"], **model_kwargs(examples))
    target = examples["next_states"].to(mean)
    nll = model.negative_log_likelihood(
        examples["contexts"], examples["actions"], target, **model_kwargs(examples)
    )
    if model.config.kind == "world":
        true_mean = examples["actions"].to(mean) - model.config.n
    else:
        true_mean = (
            examples["root_next_states"].to(mean)
            + examples["actions"].to(mean)
            - examples["root_actions"].to(mean)
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
    hidden=8,
    min_scale=0.05,
    log_every=100,
    threads=1,
):
    torch.set_num_threads(threads)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arrays, metadata = load_dataset(dataset)
    examples = {
        name: expand_pairs(arrays, index, kind)
        for index, name in enumerate(("train", "validation", "test"))
    }
    config = DensityConfig(
        kind=kind,
        n=metadata["n"],
        hidden=hidden,
        next_state_center=metadata["next_state_center"],
        next_state_scale=metadata["next_state_scale"],
        min_scale=min_scale,
    )
    torch.manual_seed(seed + (901 if kind == "world" else 902))
    model = TransitionDensity(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed + 90131)
    log_steps = {0, 1, 2, 5, 10, 20, 50, steps} | set(
        range(log_every, steps + 1, log_every)
    )
    log_steps = {step for step in log_steps if step <= steps}
    rows = []
    train = examples["train"]
    start = time.perf_counter()
    train_seconds = 0.0
    last_loss = None
    for step in range(steps + 1):
        if step in log_steps:
            metrics = evaluate_density(model, examples["validation"])
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
        model.train()
        tick = time.perf_counter()
        progress = step / max(steps - 1, 1)
        lr = final_learning_rate + 0.5 * (learning_rate - final_learning_rate) * (
            1 + math.cos(math.pi * progress)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        index = torch.randint(
            len(train["contexts"]), (batch_size,), generator=generator
        )
        loss = model.negative_log_likelihood(
            train["contexts"][index],
            train["actions"][index],
            train["next_states"][index],
            **model_kwargs(train, index),
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_seconds += time.perf_counter() - tick
        last_loss = float(loss.detach())
    model.eval().requires_grad_(False)
    final_metrics = {
        split: evaluate_density(model, data) for split, data in examples.items()
    }
    run = {
        "kind": kind,
        "seed": seed,
        "optimizer_steps": steps,
        "batch_size": batch_size,
        "environment": {"n": metadata["n"], "noise_std": metadata["noise_std"]},
        "split_pairs": {
            name: metadata[f"{name}_pairs"] for name in ("train", "validation", "test")
        },
        "learning_rate": learning_rate,
        "final_learning_rate": final_learning_rate,
        "min_scale": min_scale,
        "parameter_count": model.parameter_count,
        "offline_environment_interactions": metadata["environment_interactions"],
        "wall_seconds": time.perf_counter() - start,
        "metrics": final_metrics,
    }
    torch.save(density_checkpoint(model, training=run), output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(run, indent=2) + "\n")
    write_rows(output / "learning_curve.csv", rows)
    return run
