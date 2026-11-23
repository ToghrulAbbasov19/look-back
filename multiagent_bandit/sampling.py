"""Sampling of root episodes and of each agent's alternative actions."""

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


def group_cost(method, num_agents, M, budget_mode="all", num_actions=21):
    """Charged joint episodes per root group ("real" does not charge generated branches)."""
    method = check_method(method)
    if method not in BRANCH_METHODS:
        return 1
    return (
        1
        if method in MODEL_METHODS and budget_mode == "real"
        else 1 + num_agents * (M - 1)
    )


def budget_schedule(budget, batch_size, cost):
    """Groups per update, rounded cumulatively so every prefix stays within one group of target.

    The caller spends the leftover budget % cost on episodes that are not used for training.
    """
    previous = 0
    for target in range(batch_size, budget + batch_size, batch_size):
        completed = min(target, budget) // cost
        if completed > previous:
            yield completed - previous
        previous = completed


def categorical(probabilities, uniforms):
    """Inverse-CDF sampling from independent uniforms in [0, 1)."""
    return (
        (uniforms[..., None] > probabilities.cumsum(-1))
        .sum(-1)
        .clip(max=probabilities.shape[-1] - 1)
    )


def alternative_actions(probabilities, contexts, uniforms):
    """Same draws as `categorical`, by binary search, without large temporary arrays."""
    cdf = probabilities.cumsum(-1)
    K, S = probabilities.shape[:2]
    identity = np.arange(K).reshape((K,) + (1,) * (contexts.ndim - 1))
    lower = np.zeros(contexts.shape, dtype=np.int64)
    upper = np.full(contexts.shape, S, dtype=np.int64)
    for _ in range(S.bit_length()):
        middle = (lower + upper) // 2
        active = lower < upper
        right = uniforms > cdf[identity, contexts, middle.clip(max=S - 1)]
        lower = np.where(active & right, middle + 1, lower)
        upper = np.where(active & ~right, middle, upper)
    return lower.clip(max=S - 1)


def joint_actions(probabilities, contexts, uniforms):
    """probabilities[agent, context, action], contexts[..., agent]."""
    return categorical(probabilities[np.arange(contexts.shape[-1]), contexts], uniforms)


def loo_advantages(rewards):
    """Last axis is all M branches, including the factual root."""
    M = rewards.shape[-1]
    return rewards - (rewards.sum(-1, keepdims=True) - rewards) / (M - 1)


@dataclass
class BranchBatch:
    # contexts, actions, rewards: [agent, root group, branch].
    contexts: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    root_contexts: np.ndarray
    root_actions: np.ndarray
    environment_interactions: int
    model_samples: int = 0
    factual_environment_interactions: int = 0
    counterfactual_environment_interactions: int = 0


def sample_branches(
    env,
    probabilities,
    groups,
    method,
    M,
    root_rng,
    branch_rng,
    *,
    model=None,
    model_rng=None,
    nuisance_rng=None,
    root_action_sampler=None,
):
    """Sample one root joint episode per group and M-1 alternatives per agent.

    The alternatives for agent i change only agent i's action and are used only
    for agent i's gradient. RLOO redraws the other agents' contexts and actions;
    RLB keeps them from the root episode.
    """
    method = check_method(method)
    nuisance_rng = branch_rng if nuisance_rng is None else nuisance_rng
    num_agents, S = env.config.num_agents, env.config.num_actions
    cost = group_cost(method, num_agents, M, num_actions=S)
    learned = method in MODEL_METHODS
    if method not in BRANCH_METHODS:
        M = 1
    root_c = root_rng.integers(S, size=(groups, num_agents))
    root_uniforms = root_rng.random((groups, num_agents))
    root_a = (
        joint_actions(probabilities, root_c, root_uniforms)
        if root_action_sampler is None
        else root_action_sampler(root_c, root_uniforms)
    )
    root_r = env.reward(root_c, root_a)
    contexts = np.repeat(root_c.T[:, :, None], M, axis=-1)
    actions = np.empty((num_agents, groups, M), dtype=np.int64)
    rewards = np.empty((num_agents, groups, M), dtype=np.float32)
    actions[:, :, 0] = root_a.T
    rewards[:, :, 0] = root_r[None, :]
    if M == 1:
        return BranchBatch(
            contexts,
            actions,
            rewards,
            root_c,
            root_a,
            groups * cost,
            factual_environment_interactions=groups,
        )
    # Alternative actions use their own RNG, so all four branch methods draw the same ones.
    target_contexts = contexts[:, :, 1:]
    alternatives = alternative_actions(
        probabilities, target_contexts, branch_rng.random(target_contexts.shape)
    )
    actions[:, :, 1:] = alternatives
    if learned:
        observed = {}
        if method == "rlb_cwm":
            observed = {
                "root_contexts": target_contexts,
                "root_actions": np.broadcast_to(
                    root_a.T[:, :, None], target_contexts.shape
                ).copy(),
                "root_costs": np.broadcast_to(
                    -root_r[None, :, None], target_contexts.shape
                ).copy(),
            }
        # The models see only this agent's context and action (plus the root, for the CWM).
        costs = np.asarray(
            model.sample(target_contexts, alternatives, model_rng, **observed),
            dtype=np.float32,
        )
        rewards[:, :, 1:] = -costs
        return BranchBatch(
            contexts,
            actions,
            rewards,
            root_c,
            root_a,
            groups,
            num_agents * groups * (M - 1),
            factual_environment_interactions=groups,
        )
    shape = (num_agents, groups, M - 1, num_agents)
    if method == "rlb":
        branch_c = np.broadcast_to(root_c[None, :, None, :], shape).copy()
        branch_a = np.broadcast_to(root_a[None, :, None, :], shape).copy()
    else:
        branch_c = nuisance_rng.integers(S, size=shape)
        branch_c[np.arange(num_agents), :, :, np.arange(num_agents)] = target_contexts
        branch_a = joint_actions(probabilities, branch_c, nuisance_rng.random(shape))
    branch_a[np.arange(num_agents), :, :, np.arange(num_agents)] = alternatives
    rewards[:, :, 1:] = env.reward(
        branch_c.reshape(-1, num_agents), branch_a.reshape(-1, num_agents)
    ).reshape(num_agents, groups, M - 1)
    return BranchBatch(
        contexts,
        actions,
        rewards,
        root_c,
        root_a,
        groups * cost,
        factual_environment_interactions=groups,
        counterfactual_environment_interactions=groups * (cost - 1),
    )
