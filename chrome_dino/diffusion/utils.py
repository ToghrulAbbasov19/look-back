# Adapted from arnaudstiegler/gameNgen-repro (Apache-2.0); modified for this project.

"""Noise-conditioning helpers for diffusion inputs."""

import torch
from diffusion.config import MAX_NOISE_LEVEL, NUM_BUCKETS


def discretize_noise_level(noise_level: float) -> int:
    size_bucket = torch.tensor(MAX_NOISE_LEVEL / NUM_BUCKETS)
    return torch.min(
        torch.round(noise_level / size_bucket), torch.tensor(NUM_BUCKETS - 1)
    ).to(torch.int32)


def get_conditioning_noise(conditioning_frames: torch.Tensor) -> torch.Tensor:
    """Sample continuous and discretized conditioning-noise levels."""
    noise_level = (
        torch.rand(conditioning_frames.shape[0], device=conditioning_frames.device)
        * MAX_NOISE_LEVEL
    )
    discretized_noise_level = discretize_noise_level(noise_level)
    return noise_level, discretized_noise_level


def add_conditioning_noise(
    conditioning_frames: torch.Tensor, noise_level: torch.Tensor
) -> torch.Tensor:
    conditional_noise = torch.randn_like(conditioning_frames) * noise_level.view(
        -1, 1, 1, 1, 1
    )
    noisy_conditioning_frames = conditioning_frames + conditional_noise
    return noisy_conditioning_frames
