"""Shared environments, networks, and rollout utilities."""

from pathlib import Path

import torch
import torch.nn as nn

from chrome_dino_game import envs  # noqa: F401 (registers the gym environment; re-exported)


DINO_RL_DIR = str(Path(__file__).resolve().parent)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    """Map a stack of grayscale frames to action logits."""

    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3072, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ValueNet(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3072, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x).squeeze(-1)


def discounted_returns(rewards: list, gamma: float) -> torch.Tensor:
    returns = torch.zeros(len(rewards), dtype=torch.float32, device=DEVICE)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


class Episode:

    def __init__(self):
        self.states: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.entropies: list[torch.Tensor] = []
        self.rewards: list[float] = []

    def add(
        self,
        state: torch.Tensor,
        log_prob: torch.Tensor,
        entropy: torch.Tensor,
        reward: float,
    ):
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        self.rewards.append(reward)

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))

    def __len__(self) -> int:
        return len(self.rewards)
