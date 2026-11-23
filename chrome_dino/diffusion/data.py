# Adapted from arnaudstiegler/gameNgen-repro (Apache-2.0); modified for this project.

"""Build episode-safe context, action, and target windows for diffusion training."""

import io
import random

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision import transforms

from diffusion.config import BUFFER_SIZE, HEIGHT, WIDTH

NULL_ACTION_ID = 3

IMG_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(
            (HEIGHT, WIDTH), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def preprocess_train(examples: dict) -> dict:
    images = [
        IMG_TRANSFORMS(Image.open(io.BytesIO(image)).convert("RGB"))
        for image in examples["frames"]
    ]
    actions = (
        torch.tensor(examples["actions"])
        if isinstance(examples["actions"], list)
        else examples["actions"]
    )
    return {"pixel_values": images, "input_ids": actions}


def collate_fn(examples: list[dict]) -> dict[str, torch.Tensor]:
    images = (
        torch.stack([torch.stack(example["pixel_values"]) for example in examples])
        .to(memory_format=torch.contiguous_format)
        .float()
    )

    image_mask = torch.rand(images.shape[0], 1, 1, 1, device=images.device) < 0.0
    image_mask = image_mask.unsqueeze(1).expand(-1, images.shape[1] - 1, -1, -1, -1)
    images[:, :-1] = torch.where(
        image_mask, torch.zeros_like(images[:, :-1]), images[:, :-1]
    )

    actions = torch.stack([example["input_ids"] for example in examples])
    action_mask = torch.rand(actions.shape[0], 1, device=actions.device) < 0.0
    actions = torch.where(
        action_mask, torch.full_like(actions, NULL_ACTION_ID), actions
    )
    return {
        "pixel_values": images,
        "input_ids": actions,
        "is_unconditional": torch.tensor(
            [example["is_unconditional"] for example in examples]
        ),
    }


class EpisodeDataset:
    def __init__(
        self, dataset_name: str, uncond_prob: float = 0.2, collision_repeat: int = 1
    ):
        raw_dataset = load_dataset(
            "parquet", data_files=f"{dataset_name.rstrip('/')}/*.parquet"
        )["train"]
        self.dataset = raw_dataset.with_transform(preprocess_train)
        self.uncond_prob = uncond_prob

        self.episode_ids = np.asarray(raw_dataset["episode_id"])
        self.step_ids = np.asarray(raw_dataset["step_id"])
        self.policy_type = np.asarray(raw_dataset["policy_type"])
        self.actions = np.asarray(raw_dataset["actions"])
        self.action_dim = int(self.actions.max())
        self.crashes = (
            np.asarray(raw_dataset["crashes"])
            if "crashes" in raw_dataset.column_names
            else np.zeros(len(self.actions), dtype=np.int64)
        )

        b = BUFFER_SIZE
        same_episode = (self.episode_ids[b:] == self.episode_ids[:-b]) & (
            self.step_ids[b:] - self.step_ids[:-b] == b
        )
        targets = np.flatnonzero(same_episode) + b
        self.collision_targets = targets[
            (self.crashes[targets - 1] > 0) | (self.crashes[targets - 2] > 0)
        ]
        self.windows = targets
        if collision_repeat > 1:
            self.windows = np.concatenate(
                [targets, np.repeat(self.collision_targets, collision_repeat - 1)]
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        target = int(self.windows[idx % len(self.windows)])
        data = self.dataset[target - BUFFER_SIZE : target + 1]
        pixel_list = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x)
            for x in data["pixel_values"]
        ]
        actions = (
            data["input_ids"]
            if isinstance(data["input_ids"], torch.Tensor)
            else torch.tensor(data["input_ids"])
        )
        # Action at t-1 conditions target frame t.
        final_actions = actions[-2:-1].clone()

        is_unconditional = random.random() < self.uncond_prob
        if is_unconditional:
            final_actions = torch.full_like(final_actions, NULL_ACTION_ID)
        return {
            "pixel_values": pixel_list,
            "input_ids": final_actions,
            "is_unconditional": is_unconditional,
        }

    def get_action_dim(self) -> int:
        return self.action_dim + 1

    def describe(self) -> str:
        targets = np.unique(self.windows)
        return (
            f"{len(self.actions)} rows, {len(np.unique(self.episode_ids))} episodes, {len(targets)} windows "
            f"({np.mean(self.policy_type[targets] == 0):.0%} expert), "
            f"{len(self.collision_targets)} collision windows ({len(self.collision_targets) / len(targets):.1%}), "
            f"{len(self.windows)} samples per epoch"
        )


def get_dataloader(
    dataset_name: str,
    batch_size: int = 1,
    num_workers: int = 1,
    shuffle: bool = False,
    collision_repeat: int = 1,
) -> torch.utils.data.DataLoader:
    dataset = EpisodeDataset(dataset_name, collision_repeat=collision_repeat)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
