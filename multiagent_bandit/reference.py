"""Export trained REINFORCE (+ baseline) agent policies as tables, used to collect model data."""

from pathlib import Path

import numpy as np
import torch


def export_reference(checkpoint, output):
    checkpoint, output = Path(checkpoint), Path(output)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    params = saved["agents"]
    logits = (
        params["params.policy__weight"].transpose(1, 2)
        + params["params.policy__bias"][:, None, :]
    )
    probabilities = logits.softmax(-1).double().numpy()
    probabilities /= probabilities.sum(-1, keepdims=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, probabilities=probabilities)
