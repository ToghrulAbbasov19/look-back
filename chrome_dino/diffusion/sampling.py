# Adapted from arnaudstiegler/gameNgen-repro (Apache-2.0); modified for this project.

"""DDIM generation and single-transition inversion."""

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch.amp import autocast

from diffusion.config import (
    BUFFER_SIZE,
    CFG_GUIDANCE_SCALE,
    HEIGHT,
    WIDTH,
    DEFAULT_NUM_INFERENCE_STEPS,
)
from diffusion.model import build_action_condition_channels


def encode_conditioning_frames(
    vae: AutoencoderKL, images: torch.Tensor, vae_scale_factor: int, dtype: torch.dtype
) -> torch.Tensor:
    batch_size, _, channels, height, width = images.shape
    context_frames = images[:, :BUFFER_SIZE].reshape(-1, channels, height, width)
    conditioning_frame_latents = vae.encode(
        context_frames.to(device=vae.device, dtype=vae.dtype)
    ).latent_dist.sample()
    conditioning_frame_latents = conditioning_frame_latents * vae.config.scaling_factor

    conditioning_frame_latents = conditioning_frame_latents.reshape(
        batch_size,
        BUFFER_SIZE,
        vae.config.latent_channels,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    return conditioning_frame_latents


def get_initial_noisy_latent(
    noise_scheduler: DDPMScheduler,
    batch_size: int,
    height: int,
    width: int,
    num_channels_latents: int,
    vae_scale_factor: int,
    device: torch.device,
    dtype=torch.float32,
):
    shape = (
        batch_size,
        num_channels_latents,
        int(height) // vae_scale_factor,
        int(width) // vae_scale_factor,
    )
    latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
    latents = latents * noise_scheduler.init_noise_sigma
    return latents


def next_latent(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    action_embedding: torch.nn.Embedding,
    context_latents: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    do_classifier_free_guidance: bool = True,
    guidance_scale: float = CFG_GUIDANCE_SCALE,
    skip_action_conditioning: bool = False,
    starting_latents: torch.Tensor = None,
    autocast_dtype: torch.dtype = torch.float32,
):
    batch_size = context_latents.shape[0]
    latent_height = context_latents.shape[-2]
    latent_width = context_latents.shape[-1]
    num_channels_latents = context_latents.shape[2]
    num_image_channels = (BUFFER_SIZE + 1) * num_channels_latents
    num_actions = action_embedding.num_embeddings

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    with (
        torch.no_grad(),
        autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=autocast_dtype != torch.float32,
        ),
    ):
        if starting_latents is not None:
            latents = starting_latents.to(device=device, dtype=unet.dtype)
        else:
            latents = get_initial_noisy_latent(
                noise_scheduler,
                batch_size,
                HEIGHT,
                WIDTH,
                num_channels_latents,
                vae_scale_factor,
                device,
                dtype=unet.dtype,
            )

        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = noise_scheduler.timesteps

        NULL_ACTION_ID = 3
        cond_actions = actions.to(device)
        causal_action = cond_actions[:, 0]

        if skip_action_conditioning:
            encoder_hidden_states = torch.zeros(
                (batch_size, actions.shape[1], action_embedding.embedding_dim),
                device=device,
                dtype=unet.dtype,
            )
            if do_classifier_free_guidance:
                encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
            cond_action_channels = build_action_condition_channels(
                causal_action, num_actions, latent_height, latent_width
            ).to(device=device, dtype=unet.dtype)
            uncond_action_channels = cond_action_channels
        else:
            uncond_actions = torch.full_like(cond_actions, NULL_ACTION_ID)
            cond_embeds = action_embedding(cond_actions)
            uncond_embeds = action_embedding(uncond_actions)
            if do_classifier_free_guidance:
                encoder_hidden_states = torch.cat([uncond_embeds, cond_embeds], dim=0)
            else:
                encoder_hidden_states = cond_embeds

            cond_action_channels = build_action_condition_channels(
                causal_action, num_actions, latent_height, latent_width
            ).to(device=device, dtype=unet.dtype)
            uncond_causal = torch.full_like(causal_action, NULL_ACTION_ID)
            uncond_action_channels = build_action_condition_channels(
                uncond_causal, num_actions, latent_height, latent_width
            ).to(device=device, dtype=unet.dtype)

        image_latents = torch.cat([context_latents, latents.unsqueeze(1)], dim=1)
        image_latents = image_latents.view(batch_size, -1, latent_height, latent_width)
        cond_latents = torch.cat([image_latents, cond_action_channels], dim=1)
        uncond_latents = torch.cat([image_latents, uncond_action_channels], dim=1)
        latents = cond_latents

        for _, t in enumerate(timesteps):
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([uncond_latents, cond_latents])
            else:
                latent_model_input = latents
            latent_model_input = noise_scheduler.scale_model_input(
                latent_model_input, t
            )

            class_label_batch = (
                2 * batch_size if do_classifier_free_guidance else batch_size
            )
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=None,
                class_labels=torch.zeros(class_label_batch, dtype=torch.long).to(
                    device
                ),
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            reshaped_frames = cond_latents[:, :num_image_channels].reshape(
                batch_size,
                BUFFER_SIZE + 1,
                num_channels_latents,
                latent_height,
                latent_width,
            )
            last_frame = reshaped_frames[:, -1]
            denoised_last_frame = noise_scheduler.step(
                noise_pred, t, last_frame, return_dict=False
            )[0]

            reshaped_frames[:, -1] = denoised_last_frame
            image_latents = reshaped_frames.reshape(
                batch_size, -1, latent_height, latent_width
            )
            cond_latents = torch.cat([image_latents, cond_action_channels], dim=1)
            uncond_latents = torch.cat([image_latents, uncond_action_channels], dim=1)
            latents = cond_latents

            assert torch.all(context_latents == reshaped_frames[:, :BUFFER_SIZE])

        reshaped_frames = cond_latents[:, :num_image_channels].reshape(
            batch_size,
            BUFFER_SIZE + 1,
            num_channels_latents,
            latent_height,
            latent_width,
        )
        return reshaped_frames[:, -1]


def decode_and_postprocess(
    vae: AutoencoderKL, image_processor: VaeImageProcessor, latents: torch.Tensor
) -> Image:
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = image_processor.postprocess(
        image.detach(), output_type="pil", do_denormalize=[True] * image.shape[0]
    )[0]
    return image


def run_inference_img_conditioning_with_params(
    unet,
    vae,
    noise_scheduler,
    action_embedding,
    tokenizer,
    text_encoder,
    batch,
    device,
    num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
    do_classifier_free_guidance=True,
    guidance_scale=CFG_GUIDANCE_SCALE,
    skip_action_conditioning=False,
) -> Image:
    assert batch["pixel_values"].shape[0] == 1, "Batch size must be 1"
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float32):
        actions = batch["input_ids"]
        conditioning_frames_latents = encode_conditioning_frames(
            vae,
            images=batch["pixel_values"],
            vae_scale_factor=vae_scale_factor,
            dtype=torch.float32,
        )
        new_frame = next_latent(
            unet=unet,
            vae=vae,
            noise_scheduler=noise_scheduler,
            action_embedding=action_embedding,
            context_latents=conditioning_frames_latents,
            device=device,
            actions=actions,
            skip_action_conditioning=skip_action_conditioning,
            num_inference_steps=num_inference_steps,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=guidance_scale,
        )
        image = decode_and_postprocess(
            vae=vae, image_processor=image_processor, latents=new_frame
        )
    return image


def ddim_invert(
    unet,
    noise_scheduler,
    action_embedding,
    factual_z0: torch.Tensor,
    context_latents: torch.Tensor,
    factual_action_seq: torch.Tensor,
    device: torch.device,
    num_inference_steps: int = 50,
    autocast_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Recover the DDIM latent associated with one factual transition."""
    batch_size = context_latents.shape[0]
    latent_height = context_latents.shape[-2]
    latent_width = context_latents.shape[-1]
    num_channels_latents = context_latents.shape[2]
    num_image_channels = (BUFFER_SIZE + 1) * num_channels_latents
    num_actions = action_embedding.num_embeddings

    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = list(reversed(noise_scheduler.timesteps))

    latents = factual_z0.clone()
    encoder_hidden_states = action_embedding(factual_action_seq.to(device))
    class_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    causal_action = factual_action_seq[:, 0].to(device)
    action_channels = build_action_condition_channels(
        causal_action, num_actions, latent_height, latent_width
    ).to(device=device, dtype=unet.dtype)

    image_latents = torch.cat([context_latents, latents.unsqueeze(1)], dim=1)
    image_latents = image_latents.view(batch_size, -1, latent_height, latent_width)
    latents = torch.cat([image_latents, action_channels], dim=1)

    with (
        torch.no_grad(),
        autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=autocast_dtype != torch.float32,
        ),
    ):
        for timestep, next_timestep in zip(timesteps[:-1], timesteps[1:]):
            latent_model_input = noise_scheduler.scale_model_input(latents, timestep)
            noise_pred = unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=None,
                class_labels=class_labels,
                return_dict=False,
            )[0]

            reshaped = latents[:, :num_image_channels].reshape(
                batch_size,
                BUFFER_SIZE + 1,
                num_channels_latents,
                latent_height,
                latent_width,
            )
            last_frame = reshaped[:, -1]

            alpha_t = noise_scheduler.alphas_cumprod[timestep].to(device)
            alpha_t_next = noise_scheduler.alphas_cumprod[next_timestep].to(device)
            beta_t = 1 - alpha_t

            if noise_scheduler.config.prediction_type == "v_prediction":
                pred_x0 = alpha_t**0.5 * last_frame - beta_t**0.5 * noise_pred
                pred_eps = alpha_t**0.5 * noise_pred + beta_t**0.5 * last_frame
            elif noise_scheduler.config.prediction_type == "sample":
                pred_x0 = noise_pred
                pred_eps = (last_frame - alpha_t**0.5 * pred_x0) / beta_t**0.5
            else:
                pred_x0 = (last_frame - beta_t**0.5 * noise_pred) / alpha_t**0.5
                pred_eps = noise_pred

            if getattr(noise_scheduler.config, "clip_sample", False):
                clip_range = getattr(noise_scheduler.config, "clip_sample_range", 1.0)
                pred_x0 = pred_x0.clamp(-clip_range, clip_range)

            new_last = (
                alpha_t_next**0.5 * pred_x0 + (1 - alpha_t_next) ** 0.5 * pred_eps
            )
            reshaped[:, -1] = new_last
            image_latents = reshaped.reshape(
                batch_size, -1, latent_height, latent_width
            )
            latents = torch.cat([image_latents, action_channels], dim=1)

    reshaped = latents[:, :num_image_channels].reshape(
        batch_size,
        BUFFER_SIZE + 1,
        num_channels_latents,
        latent_height,
        latent_width,
    )
    return reshaped[:, -1]
