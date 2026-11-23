"""Policy-training defaults (shared by all methods) and CCA's Lagrange multiplier."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class TrainConfig:
    budget: int = 10_000_000
    batch_size: int = 256
    learning_rate: float = 4e-4
    entropy_coef: float = 4e-3
    hidden: int = 32
    beta: float = 0.1
    dual_learning_rate: float = 0.01
    dual_ema: float = 0.99
    initial_multiplier: float = 1.0
    log_every: int = 100
    threads: int = 1


class Multiplier:
    """Positive Lagrange multiplier for CCA's independence constraint; it rises while KL > beta*H(pi)."""

    def __init__(self, config):
        self.config = config
        self.log_value = math.log(config.initial_multiplier)
        self.ema = None

    @property
    def value(self):
        return math.exp(self.log_value)

    def update(self, kl, entropy):
        target = self.config.beta * entropy
        residual = kl - target
        self.ema = (
            residual
            if self.ema is None
            else self.config.dual_ema * self.ema + (1 - self.config.dual_ema) * residual
        )
        self.log_value = float(
            np.clip(
                self.log_value + self.config.dual_learning_rate * self.ema,
                math.log(1e-8),
                math.log(1e6),
            )
        )
        return residual
