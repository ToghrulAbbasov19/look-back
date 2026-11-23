"""Score-function estimators for one decision; M counts the real root."""

from dataclasses import dataclass

import numpy as np


METHODS = (
    "reinforce",
    "reinforce_baseline",
    "cca",
    "rloo",
    "rlb",
    "rloo_wm",
    "rlb_cwm",
)
BRANCH_METHODS = ("rloo", "rlb", "rloo_wm", "rlb_cwm")
MODEL_METHODS = ("rloo_wm", "rlb_cwm")


def check_method(method):
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    return method


def group_cost(method, M, budget_mode):
    method = check_method(method)
    if method in BRANCH_METHODS:
        return 1 if method in MODEL_METHODS and budget_mode == "real" else M
    return 1


def leave_one_out(rewards):
    """Leave-one-out advantages on the last, branch, axis."""
    return rewards - (rewards.sum(-1, keepdims=True) - rewards) / (
        rewards.shape[-1] - 1
    )


def score_loss(agent, context, actions, rewards, *, branching, entropy_coef):
    """All branch scores contribute; return/model outputs never receive gradients."""
    logp = agent.policy(context).log_softmax(-1)
    entropy = -(logp.exp() * logp).sum(-1).mean()
    advantage = leave_one_out(rewards) if branching else rewards
    chosen = logp.gather(-1, actions)
    policy = -(chosen * advantage.detach()).mean() - entropy_coef * entropy
    zero = policy.new_zeros(())
    return {
        "policy": policy,
        "value": zero,
        "supervised": zero,
        "independence": zero,
        "entropy": entropy,
        "advantage_mean": advantage.detach().mean(),
        "advantage_variance": advantage.detach().var(unbiased=False),
    }


@dataclass
class BranchBatch:
    contexts: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    rewards: np.ndarray
    environment_interactions: int
    model_samples: int


def sample_branches(env, probabilities, contexts, rng, method, M, model=None):
    """Sample a group of M branches from one context.

    RLOO draws independent noise per branch; RLB shares one noise. With a learned
    model only branch 0 is simulated, and the model never sees the noise.
    """
    method = check_method(method)
    contexts = np.asarray(contexts)
    groups = len(contexts)
    p = np.asarray(probabilities)
    # Same draws for every method, so RLOO and RLB coincide at zero noise.
    noise = rng.normal(0, env.config.noise_std, size=(groups, M)).astype(np.float32)
    uniforms = rng.random((groups, M))
    actions = (uniforms[..., None] > p[:, None, :].cumsum(-1)).sum(-1)
    actions = actions.clip(max=env.config.num_actions - 1)
    all_contexts = np.broadcast_to(contexts[:, None], actions.shape)
    if method in MODEL_METHODS:
        root = env.transition(contexts, actions[:, 0], noise[:, 0]).next_state
        kwargs = {}
        if method == "rlb_cwm":
            kwargs = {
                "root_contexts": all_contexts[:, 1:].reshape(-1),
                "root_actions": np.repeat(actions[:, 0], M - 1),
                "root_next_states": np.repeat(root, M - 1),
            }
        generated = model.sample(
            all_contexts[:, 1:].reshape(-1), actions[:, 1:].reshape(-1), rng, **kwargs
        )
        next_states = np.column_stack((root, np.asarray(generated).reshape(groups, M - 1)))
        environment_interactions, model_samples = groups, groups * (M - 1)
    else:
        if method == "rlb":
            noise = np.broadcast_to(noise[:, :1], noise.shape)
        next_states = env.transition(all_contexts, actions, noise).next_state
        environment_interactions, model_samples = groups * M, 0
    rewards = env.reward_from_next_state(contexts[:, None] - env.config.n, next_states)
    return BranchBatch(
        contexts, actions, next_states, rewards, environment_interactions, model_samples
    )
