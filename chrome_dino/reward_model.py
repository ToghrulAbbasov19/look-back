"""Predict crash, neutral, or pass rewards from consecutive decision frames.

Recorded and generated frames use the same preprocessing path.
"""

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

WM_SIZE = (320, 240)
IN_SIZE = (160, 120)
CLASSES = ("crash", "nothing", "pass")


def reward_input_from_gray(gray: np.ndarray) -> np.ndarray:
    """Resize a decoded frame for the reward model."""
    return np.asarray(
        Image.fromarray(gray).resize(IN_SIZE, Image.BILINEAR), dtype=np.uint8
    )


def reward_input_from_png(path: str) -> np.ndarray:
    """Load and preprocess a recorded simulator frame."""
    rgb = np.asarray(
        Image.open(path).convert("RGB").resize(WM_SIZE, Image.BILINEAR),
        dtype=np.float32,
    )
    return reward_input_from_gray(np.round(rgb.mean(axis=2)).astype(np.uint8))


def _episode(args):
    """Load consecutive decision-frame pairs; crash takes priority over pass."""
    ep_dir, source = args
    crashes = np.load(os.path.join(ep_dir, "crashes.npy"))
    passes = np.load(os.path.join(ep_dir, "passes.npy"))
    files = sorted(glob.glob(os.path.join(ep_dir, "frames", "*.png")))
    n = len(crashes)
    assert len(files) == n + 1, (ep_dir, len(files), n)
    labels = np.where(crashes > 0, 0, np.where(passes > 0, 2, 1)).astype(np.int64)
    frames = np.stack([reward_input_from_png(f) for f in files])
    return (
        frames,
        np.arange(n),
        np.arange(1, n + 1),
        labels,
        source,
        os.path.basename(ep_dir),
    )


def prepare(dirs, cache: str, workers: int):
    jobs = [
        (e, s)
        for s, d in enumerate(dirs)
        for e in sorted(glob.glob(os.path.join(d, "episode_*")))
    ]
    t0 = time.time()
    frames, a_idx, b_idx, labels, episode, source, step = [], [], [], [], [], [], []
    offset = 0
    with Pool(workers) as pool:
        for k, (fr, a, b, lab, src, name) in enumerate(
            pool.imap(_episode, jobs, chunksize=4)
        ):
            frames.append(fr)
            a_idx.append(a + offset)
            b_idx.append(b + offset)
            labels.append(lab)
            episode.append(np.full(len(lab), k))
            source.append(np.full(len(lab), src))
            step.append(np.arange(len(lab)))
            offset += len(fr)
            if (k + 1) % 200 == 0:
                print(
                    f"  {k + 1}/{len(jobs)} episodes, {offset} frames, {time.time() - t0:.0f} s",
                    flush=True,
                )
    out = dict(
        frames=np.concatenate(frames),
        a_idx=np.concatenate(a_idx),
        b_idx=np.concatenate(b_idx),
        labels=np.concatenate(labels),
        episode=np.concatenate(episode),
        source=np.concatenate(source),
        step=np.concatenate(step),
        episode_names=np.array([os.path.basename(e) for e, _ in jobs]),
        episode_sources=np.array([s for _, s in jobs]),
    )
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **out)
    counts = np.bincount(out["labels"], minlength=3)
    print(
        f"cached {len(out['frames'])} frames, {len(out['labels'])} decisions "
        f"({', '.join(f'{c} {n}' for c, n in zip(CLASSES, counts))}) -> {cache} in {time.time() - t0:.0f} s"
    )


class RewardNet(nn.Module):
    """Classify frame pairs as crash, neutral, or pass."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 10, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.head(self.conv(x))


SCORE_MASK_ROWS = 14


def to_input(frames_u8: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Build reward-model inputs and mask the score counter."""
    x = 1.0 - torch.stack([frames_u8[a], frames_u8[b]], dim=1).float() / 255.0
    x[:, :, :SCORE_MASK_ROWS] = 0.0
    return x


def augment(x: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    blur = torch.rand(B, device=x.device) < 0.5
    if blur.any():
        x[blur] = F.avg_pool2d(x[blur], 3, stride=1, padding=1)
    x = x * (1 + 0.2 * (torch.rand(B, 1, 1, 1, device=x.device) - 0.5))
    x = x + torch.randn_like(x) * (0.03 * torch.rand(B, 1, 1, 1, device=x.device))
    return x.clamp(0, 1)


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    conf = np.zeros((3, 3), dtype=int)
    np.add.at(conf, (true, pred), 1)
    out = {
        "confusion_true_rows": conf.tolist(),
        "accuracy": float((pred == true).mean()),
    }
    for c, name in enumerate(CLASSES):
        tp, fn, fp = (
            conf[c, c],
            conf[c].sum() - conf[c, c],
            conf[:, c].sum() - conf[c, c],
        )
        out[f"{name}_recall"] = float(tp / max(tp + fn, 1))
        out[f"{name}_precision"] = float(tp / max(tp + fp, 1))
    out["crash_pass_f1"] = float(
        np.mean(
            [
                2
                * out[f"{n}_recall"]
                * out[f"{n}_precision"]
                / max(out[f"{n}_recall"] + out[f"{n}_precision"], 1e-9)
                for n in ("crash", "pass")
            ]
        )
    )
    return out


def train(cache: str, out_dir: str, epochs: int, batch: int, lr: float, val_frac: float, seed: int):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = np.load(cache)
    labels, episode, source = d["labels"], d["episode"], d["source"]
    n_ep = len(d["episode_names"])
    val_eps = set()
    # Split each source by episode.
    for s in (0, 1):
        eps = np.flatnonzero(d["episode_sources"] == s)
        val_eps |= set(
            rng.choice(
                eps, max(1, int(round(val_frac * len(eps)))), replace=False
            ).tolist()
        )
    is_val = np.isin(episode, list(val_eps))
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    print(
        f"episodes {n_ep} ({len(val_eps)} held out); decisions train {len(tr)} val {len(va)}"
    )
    print(
        "train class counts",
        dict(zip(CLASSES, np.bincount(labels[tr], minlength=3).tolist())),
        "| val",
        dict(zip(CLASSES, np.bincount(labels[va], minlength=3).tolist())),
        flush=True,
    )

    frames = torch.from_numpy(d["frames"]).to(dev)
    a_idx, b_idx = (
        torch.from_numpy(d["a_idx"]).to(dev),
        torch.from_numpy(d["b_idx"]).to(dev),
    )
    y = torch.from_numpy(labels).to(dev)
    counts = np.bincount(labels[tr], minlength=3)
    # Use square-root inverse-frequency sampling.
    sample_w = torch.as_tensor(
        (1.0 / counts[labels[tr]]) ** 0.5, dtype=torch.float, device=dev
    )
    tr_t, va_t = torch.from_numpy(tr).to(dev), torch.from_numpy(va).to(dev)

    model = RewardNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(out_dir, exist_ok=True)
    best, history = -1.0, []
    for ep in range(epochs):
        model.train()
        order = tr_t[torch.multinomial(sample_w, len(tr), replacement=True)]
        t0, tot = time.time(), 0.0
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            x = augment(to_input(frames, a_idx[idx], b_idx[idx]))
            loss = F.cross_entropy(model(x), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(va_t), 2048):
                idx = va_t[i : i + 2048]
                preds.append(
                    model(to_input(frames, a_idx[idx], b_idx[idx])).argmax(1).cpu()
                )
        pred = torch.cat(preds).numpy()
        m = metrics(pred, labels[va])
        m_src = {
            name: metrics(pred[source[va] == s], labels[va][source[va] == s])
            for s, name in ((0, "expert"), (1, "random"))
        }
        history.append(
            {
                "epoch": ep + 1,
                "train_loss": tot / len(order),
                "val": m,
                "val_by_source": m_src,
            }
        )
        print(
            f"epoch {ep + 1:2d} loss {tot / len(order):.4f}  val acc {m['accuracy']:.4f}  "
            f"crash R {m['crash_recall']:.3f} P {m['crash_precision']:.3f}  "
            f"pass R {m['pass_recall']:.3f} P {m['pass_precision']:.3f}  ({time.time() - t0:.0f} s)",
            flush=True,
        )
        if m["crash_pass_f1"] > best:
            best = m["crash_pass_f1"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_size": IN_SIZE,
                    "classes": CLASSES,
                    "score_mask_rows": SCORE_MASK_ROWS,
                    "epoch": ep + 1,
                    "val": m,
                    "val_by_source": m_src,
                    "val_episodes": sorted(int(e) for e in val_eps),
                },
                os.path.join(out_dir, "reward_model.pt"),
            )
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    best_ep = max(history, key=lambda h: h["val"]["crash_pass_f1"])
    print(
        f"\nbest epoch {best_ep['epoch']} (crash/pass F1 {best_ep['val']['crash_pass_f1']:.3f}) -> {out_dir}/reward_model.pt"
    )
    print(
        "val confusion (rows true crash/nothing/pass, cols predicted):",
        best_ep["val"]["confusion_true_rows"],
    )
    for name, mm in best_ep["val_by_source"].items():
        print(
            f"  {name}: acc {mm['accuracy']:.4f} crash R {mm['crash_recall']:.3f} P {mm['crash_precision']:.3f} "
            f"pass R {mm['pass_recall']:.3f} P {mm['pass_precision']:.3f}  confusion {mm['confusion_true_rows']}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("cmd", choices=["prepare", "train"])
    p.add_argument("--episodes", nargs="+", help="expert and random episode folders (prepare)")
    p.add_argument("--cache", default="runs/reward_model_cache/reward_data.npz")
    p.add_argument("--output", default="runs/reward_model_training")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.cmd == "prepare":
        prepare(a.episodes, a.cache, a.workers)
    else:
        train(a.cache, a.output, a.epochs, a.batch, a.lr, a.val_frac, a.seed)
