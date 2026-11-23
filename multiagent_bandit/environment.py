"""K agents with independent contexts, a deterministic shared reward, and per-agent feedback."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class MultiAgentConfig:
    num_agents: int = 8
    n: int = 10
    feedback_dim: int = 32
    feedback_noise_std: float = 1.0

    @property
    def num_actions(self):
        return 2 * self.n + 1


class MultiAgentBandit:
    """Indices 0..2N denote -N..N. Every agent receives the same joint reward.

    W_context/W_action have axes [receiving agent, feedback channel, one-hot of
    all agents]. Feedback includes the receiver's own action, so CCA must learn
    to ignore it.
    """

    def __init__(self, config, seed):
        self.config = config
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        m, k, s = config.num_agents, config.feedback_dim, config.num_actions
        self.W_context = rng.standard_normal((m, k, m * s)).astype(np.float32)
        self.W_action = rng.standard_normal((m, k, m * s)).astype(np.float32)
        # Embedding bags compute the sparse one-hot contractions without
        # materializing a [batch, receiver, source, channel] array.
        table = np.concatenate((self.W_context, self.W_action), axis=-1)
        self._table = torch.from_numpy(table.transpose(0, 2, 1).copy().reshape(-1, k))
        self._source_offsets = np.arange(m) * s
        self._receiver_offsets = torch.arange(m)[:, None, None] * (2 * m * s)

    def _validate(self, contexts, actions):
        contexts, actions = np.asarray(contexts), np.asarray(actions)
        return contexts, actions

    def reward(self, contexts, actions):
        """Joint reward -sum_i (C_i - A_i)^2, the same for every agent."""
        contexts, actions = self._validate(contexts, actions)
        return -np.square((contexts - actions).astype(np.float32)).sum(-1)

    def feedback(self, contexts, actions, feedback_noise):
        contexts, actions = self._validate(contexts, actions)
        m, k, s = (
            self.config.num_agents,
            self.config.feedback_dim,
            self.config.num_actions,
        )
        noise = np.asarray(feedback_noise, dtype=np.float32)
        indices = np.concatenate(
            (contexts + self._source_offsets, actions + self._source_offsets + m * s),
            axis=-1,
        )
        indices = torch.from_numpy(indices)[None, :, :] + self._receiver_offsets
        with torch.no_grad():
            output = F.embedding_bag(
                indices.reshape(-1, 2 * m), self._table, mode="sum"
            )
        return output.reshape(m, len(contexts), k).permute(1, 0, 2).numpy() + noise

    def outcome(self, contexts, actions, feedback_noise):
        return self.reward(contexts, actions), self.feedback(
            contexts, actions, feedback_noise
        )

    def save(self, path):
        np.savez_compressed(path, W_context=self.W_context, W_action=self.W_action)
