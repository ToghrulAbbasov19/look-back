"""Contextual bandit with a noisy transition and a deterministic reward.

Contexts and actions are integers in [-N, N], stored as indices 0..2N.
The next state is x = (c, a + u) with noise u ~ N(0, sigma^2). Its second part,
a + u, is continuous and never clipped.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BanditConfig:
    n: int = 10
    feedback_dim: int = 32
    noise_std: float = 10.0

    @property
    def num_actions(self):
        return 2 * self.n + 1


@dataclass(frozen=True)
class TerminalState:
    context: np.ndarray
    next_state: np.ndarray
    action: np.ndarray
    noise: np.ndarray


class ContextualBandit:
    def __init__(self, config: BanditConfig, seed: int):
        self.config = config
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        size = (config.feedback_dim, config.num_actions)
        self.U = rng.standard_normal(size).astype(np.float32)
        self.V = rng.standard_normal(size).astype(np.float32)
        self.W = rng.standard_normal(config.feedback_dim).astype(np.float32)

    def transition(self, contexts, actions, noise):
        contexts, actions = np.asarray(contexts), np.asarray(actions)
        noise = np.asarray(noise, dtype=np.float32)
        return TerminalState(
            context=(contexts - self.config.n).astype(np.float32),
            next_state=(actions - self.config.n).astype(np.float32) + noise,
            action=actions.copy(),
            noise=noise.copy(),
        )

    @property
    def reward_shift(self):
        return self.config.noise_std**2

    def reward_from_next_state(self, context, next_state):
        """Deterministic reward, including the shift used during policy training."""
        return -np.square(np.asarray(context) - np.asarray(next_state)).astype(
            np.float32
        ) + np.float32(self.reward_shift)

    def reward(self, state: TerminalState):
        return self.reward_from_next_state(state.context, state.next_state)

    def feedback(self, state: TerminalState, *, reveal_noise=True):
        contexts = (state.context + self.config.n).astype(int)
        sensor = self.U[:, contexts].T + self.V[:, state.action].T
        return sensor + state.noise[..., None] * self.W if reveal_noise else sensor

    def outcome(self, contexts, actions, noise, *, reveal_noise=True):
        state = self.transition(contexts, actions, noise)
        return self.reward(state), self.feedback(state, reveal_noise=reveal_noise)

    def expected_payoff(self):
        """E[R - shift | C, A] = -(C - A)^2 for every context (rows) and action (columns)."""
        ids = np.arange(self.config.num_actions, dtype=float)
        return -((ids[:, None] - ids[None, :]) ** 2)

    def save(self, path):
        np.savez_compressed(path, U=self.U, V=self.V, W=self.W)
