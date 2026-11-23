# Adapted from arnaudstiegler/gameNgen-repro (Apache-2.0); modified for this project.

"""Shared dimensions and defaults for the Dino diffusion world model."""

BUFFER_SIZE = 8

HEIGHT = 240
WIDTH = 320

CFG_GUIDANCE_SCALE = 1.5

DEFAULT_NUM_INFERENCE_STEPS = 10

MAX_NOISE_LEVEL = 0.7
NUM_BUCKETS = 10

PRETRAINED_MODEL_NAME_OR_PATH = "CompVis/stable-diffusion-v1-4"

REPO_NAME = "dino-world-model"
