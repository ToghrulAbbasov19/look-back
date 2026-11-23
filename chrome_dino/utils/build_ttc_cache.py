"""Measure time-to-contact from consecutive frames in a parquet dataset."""

from __future__ import annotations

import argparse
import io
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image

from diffusion.reward_rules import FULL_W, detect_obstacles


FULL_H = 512
FRAMES_PER_DECISION = 4


def obstacle_tops(frame_bytes: bytes) -> list[tuple[float, float]]:
    frame = Image.open(io.BytesIO(frame_bytes)).convert("L").resize(
        (320, 240), Image.Resampling.BILINEAR
    )
    return [(box[0], box[2]) for box in detect_obstacles(frame)]


def measure_dino_front(frames: list[bytes], samples: int = 300) -> float:
    edges: list[int] = []
    rows = np.linspace(0, len(frames) - 1, min(samples, len(frames))).astype(int)
    for row in rows:
        gray = np.asarray(
            Image.open(io.BytesIO(frames[row])).convert("L").resize(
                (320, 240), Image.Resampling.BILINEAR
            )
        )
        dark = gray[int(0.35 * 240) : int(0.95 * 240), : int(0.2 * 320)] < 100
        columns = np.flatnonzero(dark.sum(0) >= 3)
        if len(columns):
            edges.append(int(columns.max()))
    if not edges:
        raise RuntimeError("could not locate the Dino in the dataset frames")
    return float(np.median(edges)) * (FULL_W / 320)


def compute_ttc(
    frames: list[bytes], episode_ids: np.ndarray, step_ids: np.ndarray, workers: int
) -> tuple[np.ndarray, np.ndarray, float]:
    with get_context("fork").Pool(workers) as pool:
        detections = pool.map(obstacle_tops, frames, chunksize=256)
    dino_x = measure_dino_front(frames)
    ttc = np.full(len(frames), np.inf, dtype=np.float32)
    speed = np.full(len(frames), np.nan, dtype=np.float32)
    for row in range(1, len(frames)):
        if episode_ids[row] != episode_ids[row - 1] or step_ids[row] != step_ids[row - 1] + 1:
            continue
        for left, top in detections[row]:
            shifts = [
                previous_left - left
                for previous_left, previous_top in detections[row - 1]
                if abs(previous_top - top) < 20 and 1 <= previous_left - left <= 0.25 * FULL_W
            ]
            if shifts and left > dino_x:
                movement = min(shifts)
                candidate = (left - dino_x) * FRAMES_PER_DECISION / movement
                if candidate < ttc[row]:
                    ttc[row] = candidate
                    speed[row] = movement / FRAMES_PER_DECISION
    return ttc, speed, dino_x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    pattern = str(args.dataset / "*.parquet") if args.dataset.is_dir() else str(args.dataset)
    data = load_dataset("parquet", data_files=pattern)["train"]
    if "policy_type" in data.column_names:
        data = data.select(np.flatnonzero(np.asarray(data["policy_type"]) == 0))
    data = data.with_format(None)
    ttc, speed, dino_x = compute_ttc(
        data["frames"],
        np.asarray(data["episode_id"]),
        np.asarray(data["step_id"]),
        args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, ttc)
    speed_path = args.output.with_name(f"{args.output.stem}_pxframe.npy")
    np.save(speed_path, speed)
    print(f"rows={len(ttc)} dino_front={dino_x:.1f} output={args.output}")
    print(f"speed cache={speed_path}")


if __name__ == "__main__":
    main()
