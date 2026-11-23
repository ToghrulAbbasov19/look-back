"""Policy-gradient losses.

RLOO and RLB use the same leave-one-out loss. They differ only in how the rollouts
of a group are coupled (see policy_roots.build_root_seeds).
"""

import torch
import torch.nn.functional as F

from common import discounted_returns
from policy_core import discount_weights


def leave_one_out_advantages(episodes, gamma: float, group_size: int):
    """A_i = G_i - mean of the other returns in its group (G = discounted return from t=0)."""
    returns = torch.stack(
        [discounted_returns(episode.rewards, gamma)[0] for episode in episodes]
    )
    grouped_returns = returns.view(-1, group_size)
    if group_size == 1:
        baselines = torch.zeros_like(grouped_returns)
    else:
        baselines = (
            grouped_returns.sum(1, keepdim=True) - grouped_returns
        ) / (group_size - 1)
    return (grouped_returns - baselines).reshape(-1)


def leave_one_out_loss(episodes, advantages, entropy_coef: float, device):
    loss = torch.zeros((), device=device)
    entropy_sum = 0.0
    entropy_means = []
    for episode, advantage in zip(episodes, advantages):
        entropy = torch.stack(episode.entropies)
        loss = (
            loss
            - advantage * torch.stack(episode.log_probs).sum()
            - entropy_coef * entropy.sum()
        )
        entropy_mean = entropy.mean().item()
        entropy_sum += entropy_mean
        entropy_means.append(entropy_mean)
    return loss / len(episodes), entropy_sum, entropy_means


def reinforce_loss(
    episodes,
    gamma: float,
    entropy_coef: float,
    device,
    value_network=None,
):
    loss = torch.zeros((), device=device)
    entropy_sum = 0.0
    for episode in episodes:
        returns = discounted_returns(episode.rewards, gamma)
        if value_network is None:
            advantages = returns
        else:
            values = value_network(torch.stack(episode.states))
            loss = loss + F.mse_loss(values, returns)
            advantages = returns - values.detach()

        log_probs = torch.stack(episode.log_probs)
        entropy = torch.stack(episode.entropies)
        # gamma^t on the score at decision t (discounted policy gradient)
        discounts = discount_weights(
            len(returns), gamma, device=returns.device, dtype=returns.dtype
        )
        loss = (
            loss
            - (discounts * log_probs * advantages).sum()
            - entropy_coef * entropy.sum()
        )
        entropy_sum += entropy.mean().item()
    return loss / len(episodes), entropy_sum
