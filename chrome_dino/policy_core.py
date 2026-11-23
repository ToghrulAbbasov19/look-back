"""Policy networks and observation preprocessing."""

from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from common import DEVICE, PolicyNet, ValueNet

K = 4
IMAGE_SIZE = (128, 64)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_policy(in_channels: int, n_actions: int, orthogonal: bool = True):
    """Create the policy network used by all estimators."""
    net = PolicyNet(in_channels=in_channels, n_actions=n_actions)
    if orthogonal:
        for m in net.conv:
            if isinstance(m, nn.Conv2d):
                layer_init(m)
        layer_init(net.fc[0])
        layer_init(net.fc[2], std=0.01)
    return net


def make_value(in_channels: int, orthogonal: bool = True):
    net = ValueNet(in_channels=in_channels)
    if orthogonal:
        for m in net.conv:
            if isinstance(m, nn.Conv2d):
                layer_init(m)
        layer_init(net.fc[0])
        layer_init(net.fc[2], std=1.0)
    return net


def make_pixel_envs(n: int, game_mode: str = "train"):
    gym_envs = []
    for _ in range(n):
        e = gym.make(
            "Env-v0",
            render_mode="rgb_array",
            game_mode=game_mode,
            train_frame_limit=99999,
            disable_env_checker=True,
        )
        gym_envs.append(e)
    return gym_envs


def to_gray(obs: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(obs[..., 0]).resize(IMAGE_SIZE), dtype=np.float32)


def stack_to_tensor(stack: deque) -> torch.Tensor:
    """Normalize frames with a zero-valued background."""
    return (
        1.0 - torch.tensor(np.stack(stack), dtype=torch.float32, device=DEVICE) / 255.0
    )


def discount_weights(
    length: int, gamma: float, *, device, dtype
) -> torch.Tensor:
    """gamma**t for t = 0..length-1."""
    steps = torch.arange(length, device=device, dtype=dtype)
    return torch.as_tensor(gamma, device=device, dtype=dtype).pow(steps)
