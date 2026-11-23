"""Frozen diffusion and reward-model inference adapters."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from common import DEVICE
from reward_model import SCORE_MASK_ROWS, RewardNet, reward_input_from_gray


from diffusers import DDIMScheduler

from diffusion.model import load_model
from diffusion.sampling import ddim_invert, next_latent

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def to_gray_uint8(images: torch.Tensor) -> np.ndarray:
    """Convert decoded RGB tensors in [-1, 1] to uint8 grayscale images."""
    return (
        ((images.float().clamp(-1, 1) + 1) * 127.5).mean(1).round().byte().cpu().numpy()
    )


def policy_frame(gray: np.ndarray) -> np.ndarray:
    """Resize a world-model frame to the policy's 128x64 observation."""
    from PIL import Image

    return np.asarray(
        Image.fromarray(gray).resize((128, 64), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def load_reward_model(path: str) -> nn.Module:
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = RewardNet().to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False)


@torch.no_grad()
def reward_model_probs(
    model: nn.Module,
    before: np.ndarray,
    after: np.ndarray,
) -> np.ndarray:
    """Return crash, neutral, and pass probabilities."""
    pairs = np.stack(
        [
            [reward_input_from_gray(a), reward_input_from_gray(b)]
            for a, b in zip(before, after)
        ]
    )
    inputs = 1.0 - torch.from_numpy(pairs).to(DEVICE).float() / 255.0
    inputs[:, :, :SCORE_MASK_ROWS] = 0.0
    return torch.softmax(model(inputs).float(), dim=1).cpu().numpy()


class WorldModel:
    """Inference-only action-conditioned latent diffusion world model."""

    def __init__(
        self,
        model_folder: str,
        steps: int,
        cfg: float,
        dtype: str,
        vae_chunk: int,
    ) -> None:
        (
            self.unet,
            self.vae,
            self.action_embedding,
            scheduler,
            _,
            _,
        ) = load_model(model_folder, device=DEVICE, load_text_encoder=False)
        for module in (self.unet, self.vae, self.action_embedding):
            module.requires_grad_(False)
        self.scheduler = DDIMScheduler.from_config(scheduler.config)
        self.steps = steps
        self.cfg = cfg
        self.dtype = DTYPES[dtype]
        self.chunk = vae_chunk
        self.scale = self.vae.config.scaling_factor

    @torch.no_grad()
    def encode(self, images: torch.Tensor, sample: bool) -> torch.Tensor:
        """Frames in [-1, 1] -> scaled VAE latents (posterior sample or mode)."""
        latents = []
        for chunk in images.split(self.chunk):
            distribution = self.vae.encode(chunk.to(DEVICE, torch.float32)).latent_dist
            latents.append(distribution.sample() if sample else distribution.mode())
        return torch.cat(latents) * self.scale

    @torch.no_grad()
    def decode_gray(self, latents: torch.Tensor) -> np.ndarray:
        """Latents -> uint8 grayscale frames."""
        return np.concatenate(
            [
                to_gray_uint8(self.vae.decode(chunk / self.scale).sample)
                for chunk in latents.split(self.chunk)
            ]
        )

    @torch.no_grad()
    def abduct(
        self,
        frame: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """DDIM-invert a factual next latent to the noise that generates it."""
        return ddim_invert(
            unet=self.unet,
            noise_scheduler=self.scheduler,
            action_embedding=self.action_embedding,
            factual_z0=frame,
            context_latents=context,
            factual_action_seq=actions,
            device=DEVICE,
            num_inference_steps=self.steps,
            autocast_dtype=self.dtype,
        )

    @torch.no_grad()
    def predict(
        self,
        noise: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Generate the next latent from noise, context, and action."""
        return next_latent(
            unet=self.unet,
            vae=self.vae,
            noise_scheduler=self.scheduler,
            action_embedding=self.action_embedding,
            context_latents=context,
            actions=actions,
            device=DEVICE,
            num_inference_steps=self.steps,
            do_classifier_free_guidance=self.cfg != 1.0,
            guidance_scale=self.cfg,
            starting_latents=noise,
            autocast_dtype=self.dtype,
        ).float()
