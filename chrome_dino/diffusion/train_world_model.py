#!/usr/bin/env python
# coding=utf-8
# Adapted from arnaudstiegler/gameNgen-repro (Apache-2.0); modified for this project.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train the action-conditioned diffusion world model (8 context frames -> next frame)."""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import datasets
import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration, set_seed
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, cast_training_params
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo
from packaging import version
from safetensors.torch import load_file
from tqdm.auto import tqdm

try:
    import wandb
except ModuleNotFoundError:
    wandb = None
from diffusion.config import (
    BUFFER_SIZE,
    CFG_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    REPO_NAME,
)
from diffusion.data import EpisodeDataset, collate_fn
from diffusion.model import (
    build_action_condition_channels,
    get_model,
    save_and_maybe_upload_to_hub,
    save_model,
)
from diffusion.sampling import run_inference_img_conditioning_with_params
from diffusion.utils import add_conditioning_noise, get_conditioning_noise

logger = get_logger(__name__, log_level="INFO")

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the action-conditioned Dino diffusion world model."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ddp_backend",
        type=str,
        choices=["nccl", "gloo"],
        default="nccl",
        help="Distributed backend; reported runs used gloo.",
    )
    parser.add_argument(
        "--collision_repeat",
        type=int,
        default=1,
        help="Repeat windows whose target frame shows a collision this many times.",
    )
    parser.add_argument("--validation_steps", type=int, default=50)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global seed for initialization, data order, VAE sampling, and diffusion noise.",
    )
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=16)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--prediction_type", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='Supported: "tensorboard", "wandb", "comet_ml", "all".',
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=20000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true"
    )
    parser.add_argument("--noise_offset", type=float, default=0)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument(
        "--unet_size",
        type=str,
        default="sd",
        choices=["sd", "tiny", "small"],
        help="sd: fine-tune the pretrained SD-1.4 UNet; tiny (18M) / small (101M): UNet trained from scratch",
    )
    parser.add_argument("--load_pretrained", type=str, default=None)
    parser.add_argument(
        "--use_ema", action="store_true", help="Use EMA for the UNet weights."
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.9999, help="EMA decay rate."
    )

    parser.set_defaults(
        skip_action_conditioning=False,
        skip_image_conditioning=False,
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if not args.dataset_name:
        raise ValueError("--dataset_name must point to a parquet dataset directory.")

    return args


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.report_to == "wandb":
        import wandb

        run = wandb.init()

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[InitProcessGroupKwargs(backend=args.ddp_backend)],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            create_repo(repo_id=REPO_NAME, exist_ok=True, token=args.hub_token)

    dataset = EpisodeDataset(args.dataset_name, collision_repeat=args.collision_repeat)
    logger.info(f"dataset: {dataset.describe()}", main_process_only=True)
    action_dim = dataset.get_action_dim()
    NUM_ACTIONS = action_dim + 1

    unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder = get_model(
        action_dim,
        skip_image_conditioning=args.skip_image_conditioning,
        unet_size=args.unet_size,
    )

    print("action_dim =", action_dim)
    print("embedding size =", action_embedding.num_embeddings)

    if args.load_pretrained:
        print("loading pretrained")
        logger.info(f"Loading pretrained model from {args.load_pretrained}")
        unet.load_state_dict(
            load_file(
                os.path.join(
                    args.load_pretrained, "unet", "diffusion_pytorch_model.safetensors"
                )
            )
        )
        vae.load_state_dict(
            load_file(
                os.path.join(
                    args.load_pretrained, "vae", "diffusion_pytorch_model.safetensors"
                )
            )
        )

        with open(
            os.path.join(
                args.load_pretrained, "noise_scheduler", "scheduler_config.json"
            ),
            "r",
        ) as f:
            scheduler_config = json.load(f)
        noise_scheduler = DDIMScheduler.from_config(scheduler_config)

        action_embedding.load_state_dict(
            load_file(
                os.path.join(args.load_pretrained, "action_embedding_model.safetensors")
            )
        )

        with open(os.path.join(args.load_pretrained, "embedding_info.json")) as f:
            embedding_info = json.load(f)
        action_embedding.num_embeddings = embedding_info["num_embeddings"]
        action_embedding.embedding_dim = embedding_info["embedding_dim"]

    if args.use_ema:
        ema_unet = EMAModel(
            unet.parameters(), decay=args.ema_decay, use_ema_warmup=True
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    action_embedding.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if args.skip_action_conditioning:
        optimizer = optimizer_cls(
            unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = optimizer_cls(
            list(unet.parameters()) + list(action_embedding.parameters()),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        collate_fn=collate_fn,
        generator=data_generator,
    )

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(
            len(train_dataloader) / accelerator.num_processes
        )
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding / args.gradient_accumulation_steps
        )
        num_training_steps_for_scheduler = (
            args.num_train_epochs
            * num_update_steps_per_epoch
            * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = (
            args.max_train_steps * accelerator.num_processes
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    unet, action_embedding, optimizer, train_dataloader, lr_scheduler = (
        accelerator.prepare(
            unet, action_embedding, optimizer, train_dataloader, lr_scheduler
        )
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)
        accelerator.register_for_checkpointing(ema_unet)

    trainable_params = [
        parameter
        for module in (unet, action_embedding)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if (
            num_training_steps_for_scheduler
            != args.max_train_steps * accelerator.num_processes
        ):
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Dataset = {args.dataset_name}")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  BUFFER_SIZE = {BUFFER_SIZE}")
    logger.info(
        f"  Use EMA = {args.use_ema}"
        + (f" (decay={args.ema_decay}, warmup=True)" if args.use_ema else "")
    )
    logger.info(f"  Skip action conditioning = {args.skip_action_conditioning}")
    logger.info(f"  Skip image conditioning = {args.skip_image_conditioning}")
    logger.info(f"  Use CFG = {args.use_cfg}")
    logger.info(f"  Prediction type = {args.prediction_type}")
    logger.info(f"  Learning rate = {args.learning_rate}")
    logger.info(f"  Adam weight decay = {args.adam_weight_decay}")
    logger.info(f"  Output dir = {args.output_dir}")

    global_step = 0
    first_epoch = 0
    resume_step = 0

    resume_state_dir = os.path.join(args.output_dir, "resume_state")
    if args.resume_from_checkpoint:
        if not os.path.exists(resume_state_dir):
            accelerator.print(
                f"No resume state found at '{resume_state_dir}'. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from {resume_state_dir}")
            accelerator.load_state(resume_state_dir)
            with open(os.path.join(resume_state_dir, "global_step.json")) as f:
                global_step = json.load(f)["global_step"]

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = (
                global_step % num_update_steps_per_epoch
            ) * args.gradient_accumulation_steps
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        data_generator.manual_seed(args.seed + epoch)
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if epoch == first_epoch and step < resume_step:
                continue
            with accelerator.accumulate(unet):
                if args.skip_image_conditioning:
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    if args.noise_offset:
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1),
                            device=latents.device,
                        )

                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    ).long()

                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    concatenated_latents = noisy_latents
                else:
                    bs, buffer_len, channels, height, width = batch[
                        "pixel_values"
                    ].shape

                    folded_conditional_images = batch["pixel_values"].view(
                        bs * buffer_len, channels, height, width
                    )

                    latents = vae.encode(
                        folded_conditional_images.to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    _, latent_channels, latent_height, latent_width = latents.shape
                    latents = latents.view(
                        bs, buffer_len, latent_channels, latent_height, latent_width
                    )

                    noise = torch.randn_like(latents[:, -1:, :, :, :])

                    if args.noise_offset:
                        noise[:, -1, :, :, :] = args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[2], 1, 1),
                            device=latents.device,
                        )

                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bs,),
                        device=latents.device,
                    ).long()

                    noisy_latents = latents.clone()
                    noisy_latents[:, -1:, :, :, :] = noise_scheduler.add_noise(
                        latents[:, -1:, :, :, :], noise, timesteps
                    )

                    noise_level, discretized_noise_level = get_conditioning_noise(
                        latents[:, :-1, :, :, :]
                    )
                    noisy_latents[:, :-1, :, :, :] = add_conditioning_noise(
                        latents[:, :-1, :, :, :], noise_level
                    )

                    concatenated_latents = noisy_latents.view(
                        bs, buffer_len * latent_channels, latent_height, latent_width
                    )

                    causal_action = batch["input_ids"][:, 0]
                    action_channels = build_action_condition_channels(
                        causal_action, NUM_ACTIONS, latent_height, latent_width
                    ).to(
                        device=concatenated_latents.device,
                        dtype=concatenated_latents.dtype,
                    )
                    concatenated_latents = torch.cat(
                        [concatenated_latents, action_channels], dim=1
                    )

                if args.skip_action_conditioning:
                    encoder_hidden_states = text_encoder(
                        batch["input_ids"], return_dict=False
                    )[0]
                else:
                    encoder_hidden_states = action_embedding(batch["input_ids"])

                if args.prediction_type is not None:
                    noise_scheduler.register_to_config(
                        prediction_type=args.prediction_type
                    )

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                model_pred = unet(
                    concatenated_latents,
                    timesteps,
                    class_labels=discretized_noise_level,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]

                if not args.skip_image_conditioning:
                    target_last_frame = target[:, -1, :, :, :]
                else:
                    target_last_frame = target

                loss_unreduced = F.mse_loss(
                    model_pred.float(), target_last_frame.float(), reduction="none"
                )
                loss_unreduced = loss_unreduced.mean(
                    dim=list(range(1, len(loss_unreduced.shape)))
                )

                is_uncond_mask = batch["is_unconditional"].to(loss_unreduced.device)

                if is_uncond_mask.any():
                    uncond_loss = loss_unreduced[is_uncond_mask].mean().detach().item()
                else:
                    uncond_loss = None

                if (~is_uncond_mask).any():
                    cond_loss = loss_unreduced[~is_uncond_mask].mean().detach().item()
                else:
                    cond_loss = None

                loss = loss_unreduced.mean()

                if accelerator.is_main_process and args.report_to == "wandb":
                    step_logs = {
                        "train_loss": loss.item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                    }
                    if uncond_loss is not None:
                        step_logs["train_loss_unconditional"] = uncond_loss
                    if cond_loss is not None:
                        step_logs["train_loss_conditional"] = cond_loss
                    run.log(step_logs, step=global_step)

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(accelerator.unwrap_model(unet).parameters())
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        checkpoint_dir = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        logger.info(f"Saving clean checkpoint to {checkpoint_dir}")
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        if args.use_ema:
                            ema_unet.store(unwrapped_unet.parameters())
                            ema_unet.copy_to(unwrapped_unet.parameters())
                        save_model(
                            checkpoint_dir,
                            unet=unwrapped_unet,
                            vae=vae,
                            noise_scheduler=noise_scheduler,
                            action_embedding=accelerator.unwrap_model(action_embedding),
                        )
                        if args.use_ema:
                            ema_unet.restore(unwrapped_unet.parameters())

                    logger.info(f"Updating resume state at {resume_state_dir}")
                    accelerator.save_state(resume_state_dir)
                    if accelerator.is_main_process:
                        with open(
                            os.path.join(resume_state_dir, "global_step.json"), "w"
                        ) as f:
                            json.dump({"global_step": global_step}, f)

                log_dict = {
                    "train_loss": train_loss,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                }
                if uncond_loss is not None:
                    log_dict["train_loss_unconditional"] = uncond_loss
                if cond_loss is not None:
                    log_dict["train_loss_conditional"] = cond_loss

                accelerator.log(log_dict, step=global_step)
                train_loss = 0.0

                validation_images = []
                context_images = []
                target_images = []
                if global_step % args.validation_steps == 0:
                    accelerator.print("Generating validation image")
                    unet.eval()
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    if args.use_ema:
                        ema_unet.store(unwrapped_unet.parameters())
                        ema_unet.copy_to(unwrapped_unet.parameters())
                    if accelerator.is_main_process:
                        save_and_maybe_upload_to_hub(
                            repo_id=REPO_NAME,
                            output_dir=args.output_dir,
                            unet=unwrapped_unet,
                            vae=vae,
                            noise_scheduler=noise_scheduler,
                            action_embedding=accelerator.unwrap_model(action_embedding),
                            should_upload_to_hub=args.push_to_hub,
                        )

                        for i in range(min(2, batch["pixel_values"].shape[0])):
                            single_sample_batch = {
                                "pixel_values": batch["pixel_values"][i].unsqueeze(0),
                                "input_ids": batch["input_ids"][i].unsqueeze(0),
                            }
                            with torch.no_grad():
                                if args.skip_image_conditioning:
                                    raise NotImplementedError("Not supported anymore")
                                else:
                                    generated_image = run_inference_img_conditioning_with_params(
                                        unet=unwrapped_unet,
                                        vae=vae,
                                        noise_scheduler=noise_scheduler,
                                        action_embedding=accelerator.unwrap_model(
                                            action_embedding
                                        ),
                                        tokenizer=tokenizer,
                                        text_encoder=text_encoder,
                                        batch=single_sample_batch,
                                        device=accelerator.device,
                                        num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
                                        do_classifier_free_guidance=args.use_cfg,
                                        guidance_scale=CFG_GUIDANCE_SCALE,
                                        skip_action_conditioning=args.skip_action_conditioning,
                                    )
                                validation_images.append(generated_image)

                                context_images.append(
                                    single_sample_batch["pixel_values"][0][:BUFFER_SIZE]
                                )
                                target_images.append(
                                    single_sample_batch["pixel_values"][0][-1]
                                )

                        if args.report_to == "wandb":
                            wandb.log(
                                {
                                    "1_validation_images": [
                                        wandb.Image(img, caption=f"Generated Image {i}")
                                        for i, img in enumerate(validation_images)
                                    ],
                                    "2_target_images": [
                                        wandb.Image(
                                            target_img, caption=f"Target Image {i}"
                                        )
                                        for i, target_img in enumerate(target_images)
                                    ],
                                    "3_context_images": [
                                        wandb.Image(
                                            context_img, caption=f"Context Image {i}"
                                        )
                                        for i, context_img in enumerate(context_images)
                                    ],
                                },
                                step=global_step,
                            )
                    if args.use_ema:
                        ema_unet.restore(unwrapped_unet.parameters())
                    unet.train()

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.store(unwrapped_unet.parameters())
            ema_unet.copy_to(unwrapped_unet.parameters())
        save_and_maybe_upload_to_hub(
            repo_id=REPO_NAME,
            output_dir=args.output_dir,
            unet=unwrapped_unet,
            vae=vae,
            noise_scheduler=noise_scheduler,
            action_embedding=accelerator.unwrap_model(action_embedding),
            should_upload_to_hub=args.push_to_hub,
        )
        if args.use_ema:
            ema_unet.restore(unwrapped_unet.parameters())

    accelerator.end_training()

    if accelerator.is_main_process:
        if args.report_to == "wandb":
            wandb.finish()


if __name__ == "__main__":
    main()
