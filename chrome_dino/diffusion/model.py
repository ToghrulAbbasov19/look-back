# Adapted from arnaudstiegler/gameNgen-repro (Apache-2.0); modified for this project.

"""Construct, save, and load the diffusion world model."""

import json
import os

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from huggingface_hub import hf_hub_download, upload_folder
from safetensors.torch import save_file, load_file
from transformers import CLIPTokenizer, CLIPTextModel

from diffusion.config import BUFFER_SIZE, HEIGHT, PRETRAINED_MODEL_NAME_OR_PATH, WIDTH
from diffusion.utils import NUM_BUCKETS


def get_ft_vae_decoder(path: str | None = None) -> AutoencoderKL:
    path = path or os.environ.get("DINO_VAE_PATH")
    if not path:
        raise RuntimeError("DINO_VAE_PATH must point to the pretrained Dino VAE")
    return AutoencoderKL.from_pretrained(path)


def build_action_condition_channels(
    action_ids: torch.Tensor,
    num_actions: int,
    height: int,
    width: int,
) -> torch.Tensor:
    batch_size = action_ids.shape[0]
    one_hot = F.one_hot(action_ids.long(), num_classes=num_actions).to(
        dtype=torch.float32
    )
    channels = one_hot.view(batch_size, num_actions, 1, 1).expand(
        batch_size, num_actions, height, width
    )
    return channels.contiguous()


TINY_UNET_CONFIG = dict(
    sample_size=(HEIGHT // 8, WIDTH // 8),
    out_channels=4,
    block_out_channels=(64, 128, 256),
    layers_per_block=1,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    attention_head_dim=8,
    cross_attention_dim=768,
)

UNET_CONFIGS = {
    "tiny": TINY_UNET_CONFIG,
    "small": dict(
        TINY_UNET_CONFIG, block_out_channels=(128, 256, 512), layers_per_block=2
    ),
}


def get_model(
    action_embedding_dim: int,
    skip_image_conditioning: bool = False,
    unet_size: str = "sd",
) -> tuple[
    UNet2DConditionModel,
    AutoencoderKL,
    torch.nn.Embedding,
    DDIMScheduler,
    CLIPTokenizer,
    CLIPTextModel,
]:
    action_embedding = torch.nn.Embedding(
        num_embeddings=action_embedding_dim + 1, embedding_dim=768
    )
    torch.nn.init.normal_(action_embedding.weight, mean=0.0, std=0.02)
    num_actions = action_embedding_dim + 1

    noise_scheduler = DDIMScheduler.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="scheduler"
    )
    noise_scheduler.register_to_config(prediction_type="v_prediction")

    vae = get_ft_vae_decoder()

    if unet_size in UNET_CONFIGS:
        in_channels = (
            4 if skip_image_conditioning else 4 * (BUFFER_SIZE + 1) + num_actions
        )
        unet = UNet2DConditionModel(
            **UNET_CONFIGS[unet_size],
            in_channels=in_channels,
            num_class_embeds=NUM_BUCKETS,
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(
            PRETRAINED_MODEL_NAME_OR_PATH, subfolder="unet"
        )
        unet.register_to_config(num_class_embeds=NUM_BUCKETS)
        unet.class_embedding = torch.nn.Embedding(
            NUM_BUCKETS, unet.time_embedding.linear_2.out_features
        )

        if not skip_image_conditioning:
            num_image_channels = 4 * (BUFFER_SIZE + 1)
            new_in_channels = num_image_channels + num_actions
            new_conv_in = torch.nn.Conv2d(
                new_in_channels, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )
            torch.nn.init.xavier_uniform_(new_conv_in.weight)
            with torch.no_grad():
                new_conv_in.weight[:, num_image_channels:, :, :] = 0
            torch.nn.init.zeros_(new_conv_in.bias)

            unet.conv_in = new_conv_in
            unet.config["in_channels"] = new_in_channels

    tokenizer = CLIPTokenizer.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="text_encoder"
    )

    unet.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    return unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder


def load_embedding_info_dict(model_folder: str) -> dict:
    if os.path.exists(model_folder):
        with open(os.path.join(model_folder, "embedding_info.json"), "r") as f:
            embedding_info = json.load(f)
    else:
        file_path = hf_hub_download(
            repo_id=model_folder, filename="embedding_info.json", repo_type="model"
        )
        with open(file_path, "r") as f:
            embedding_info = json.load(f)
    return embedding_info


def load_action_embedding(
    model_folder: str, action_num_embeddings: int
) -> torch.nn.Embedding:
    action_embedding = torch.nn.Embedding(
        num_embeddings=action_num_embeddings, embedding_dim=768
    )
    if os.path.exists(model_folder):
        action_embedding.load_state_dict(
            load_file(os.path.join(model_folder, "action_embedding_model.safetensors"))
        )
    else:
        file_path = hf_hub_download(
            repo_id=model_folder,
            filename="action_embedding_model.safetensors",
            repo_type="model",
        )
        action_embedding.load_state_dict(load_file(file_path))
    return action_embedding


def load_model(
    model_folder: str,
    device: torch.device | None = None,
    load_text_encoder: bool = True,
) -> tuple[
    UNet2DConditionModel,
    AutoencoderKL,
    torch.nn.Embedding,
    DDIMScheduler,
    CLIPTokenizer,
    CLIPTextModel,
]:
    embedding_info = load_embedding_info_dict(model_folder)
    action_embedding = load_action_embedding(
        model_folder=model_folder,
        action_num_embeddings=embedding_info["num_embeddings"],
    )

    noise_scheduler = DDIMScheduler.from_pretrained(
        model_folder, subfolder="noise_scheduler"
    )

    local_vae = os.path.join(model_folder, "vae")
    vae_path = os.environ.get("DINO_VAE_PATH")
    if vae_path is None and os.path.isdir(local_vae):
        vae_path = local_vae
    vae = get_ft_vae_decoder(vae_path)
    unet = UNet2DConditionModel.from_pretrained(model_folder, subfolder="unet")

    assert noise_scheduler.config.prediction_type == "v_prediction"
    assert unet.config.num_class_embeds == NUM_BUCKETS

    tokenizer = text_encoder = None
    if load_text_encoder:
        tokenizer = CLIPTokenizer.from_pretrained(
            PRETRAINED_MODEL_NAME_OR_PATH, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            PRETRAINED_MODEL_NAME_OR_PATH, subfolder="text_encoder"
        )

    if device:
        unet = unet.to(device)
        vae = vae.to(device)
        action_embedding = action_embedding.to(device)
        if text_encoder is not None:
            text_encoder = text_encoder.to(device)

    return unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder


def save_model(
    output_dir: str,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler: DDIMScheduler,
    action_embedding: torch.nn.Embedding,
) -> None:
    unet.save_pretrained(os.path.join(output_dir, "unet"))
    vae.save_pretrained(os.path.join(output_dir, "vae"))
    noise_scheduler.save_pretrained(os.path.join(output_dir, "noise_scheduler"))
    save_file(
        action_embedding.state_dict(),
        os.path.join(output_dir, "action_embedding_model.safetensors"),
    )
    embedding_info = {
        "num_embeddings": action_embedding.num_embeddings,
        "embedding_dim": action_embedding.embedding_dim,
    }
    with open(os.path.join(output_dir, "embedding_info.json"), "w") as f:
        json.dump(embedding_info, f)


def save_and_maybe_upload_to_hub(
    repo_id: str,
    output_dir: str,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler: DDIMScheduler,
    action_embedding: torch.nn.Embedding,
    should_upload_to_hub: bool = True,
) -> None:
    save_model(output_dir, unet, vae, noise_scheduler, action_embedding)

    if should_upload_to_hub:
        try:
            upload_folder(
                repo_id=repo_id,
                folder_path=output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
        except Exception as e:
            print(f"Error uploading to hub: {e}")
