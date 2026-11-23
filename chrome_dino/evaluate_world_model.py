"""Evaluate CWM and WM generations using composition, effectiveness, FID,
CLD minimality, and obstacle-preservation metrics."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image

from common import DEVICE
from diffusion.config import BUFFER_SIZE, HEIGHT, WIDTH
from diffusion.data import IMG_TRANSFORMS
from world_model import WorldModel, to_gray_uint8


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class FactualData:
    """Recorded expert rows and the precomputed time-to-contact per row."""

    def __init__(self, dataset_path: Path, ttc_cache: Path) -> None:
        raw = load_dataset("parquet", data_files=str(dataset_path / "*.parquet"))[
            "train"
        ]
        if "policy_type" in raw.column_names:
            raw = raw.select(np.flatnonzero(np.asarray(raw["policy_type"]) == 0))
        raw = raw.with_format(None)
        self.frames = raw["frames"]
        self.actions = np.asarray(raw["actions"], dtype=np.int64)
        self.episode_ids = np.asarray(raw["episode_id"], dtype=np.int64)
        self.step_ids = np.asarray(raw["step_id"], dtype=np.int64)
        self.ttc = np.load(ttc_cache)
        if len(self.ttc) != len(self.actions):
            raise ValueError(
                f"TTC cache has {len(self.ttc)} rows, dataset has "
                f"{len(self.actions)} expert rows"
            )

    def images(self, rows: np.ndarray) -> torch.Tensor:
        flat = np.asarray(rows, dtype=np.int64).reshape(-1)
        images = [
            IMG_TRANSFORMS(Image.open(io.BytesIO(self.frames[int(r)])).convert("RGB"))
            for r in flat
        ]
        return torch.stack(images)

    def starts(self, start_ttc: float = 12.0) -> np.ndarray:
        """One-step roots with TTC in (8, 12] and no jump in the last 3 rows."""
        out: list[int] = []
        for root in range(BUFFER_SIZE, len(self.actions) - 1):
            first, last = root - BUFFER_SIZE, root + 1
            if (
                self.episode_ids[first] != self.episode_ids[last]
                or self.step_ids[last] - self.step_ids[first] != last - first
            ):
                continue
            if not (start_ttc - 4.0 < self.ttc[root] <= start_ttc):
                continue
            if 1 in self.actions[root - 3 : root]:
                continue
            out.append(root)
        return np.asarray(out, dtype=np.int64)


def valid_transitions(data: FactualData) -> np.ndarray:
    rows = np.arange(len(data.actions) - 1)
    return rows[
        (data.episode_ids[rows] == data.episode_ids[rows + 1])
        & (data.step_ids[rows + 1] == data.step_ids[rows] + 1)
    ]


def gray_tensor(data: FactualData, rows: np.ndarray, size=(64, 128)) -> torch.Tensor:
    x = data.images(rows).float()
    x = ((x.clamp(-1, 1) + 1) * 0.5).mean(1, keepdim=True)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def uint8_tensor(images: np.ndarray, size=(64, 128)) -> torch.Tensor:
    x = torch.from_numpy(images).float().unsqueeze(1) / 255.0
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ActionPredictor(nn.Module):
    """Action classifier used to compute effectiveness."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 192, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(192, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x[..., :32]).flatten(1))


class EvalVAE(nn.Module):
    """VAE used to compute the CLD metric."""

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 192, 4, 2, 1),
            nn.ReLU(),
        )
        self.mu = nn.Linear(192 * 4 * 8, latent_dim)
        self.logvar = nn.Linear(192 * 4 * 8, latent_dim)
        self.fc = nn.Linear(latent_dim, 192 * 4 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return self.mu(h), self.logvar(h).clamp(-12, 8)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(self.fc(z).view(-1, 192, 4, 8)), mu, logvar


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for label in range(3):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(scores))


def balanced_rows(rows, labels, per_class, rng):
    selected = []
    for label in range(3):
        candidates = rows[labels == label]
        selected.extend(
            rng.choice(candidates, min(per_class, len(candidates)), replace=False)
        )
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def train_evaluators(args, data: FactualData, out_dir: Path):
    rng = np.random.default_rng(args.seed)
    rows = valid_transitions(data)
    rows = np.asarray(
        [
            r
            for r in rows
            if data.episode_ids[max(0, r - 3)] == data.episode_ids[r]
            and 1 not in data.actions[max(0, r - 3) : r]
        ],
        dtype=np.int64,
    )
    train = rows[(data.episode_ids[rows] % 5) != 0]
    valid = rows[(data.episode_ids[rows] % 5) == 0]
    train = balanced_rows(
        train, data.actions[train], args.evaluator_train_per_class, rng
    )
    valid = balanced_rows(
        valid, data.actions[valid], args.evaluator_valid_per_class, rng
    )

    predictor = ActionPredictor().to(DEVICE)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=2e-3, weight_decay=1e-4)
    best_score, best_state = -1.0, None
    for epoch in range(args.predictor_epochs):
        predictor.train()
        permutation = rng.permutation(train)
        for begin in range(0, len(train), args.evaluator_batch):
            batch_rows = permutation[begin : begin + args.evaluator_batch]
            x = gray_tensor(data, batch_rows + 1).to(DEVICE)
            y = torch.as_tensor(data.actions[batch_rows], device=DEVICE)
            loss = F.cross_entropy(predictor(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        predictor.eval()
        predictions = []
        with torch.no_grad():
            for begin in range(0, len(valid), args.evaluator_batch):
                batch_rows = valid[begin : begin + args.evaluator_batch]
                predictions.append(
                    predictor(gray_tensor(data, batch_rows + 1).to(DEVICE))
                    .argmax(1)
                    .cpu()
                )
        score = macro_f1(data.actions[valid], torch.cat(predictions).numpy())
        print(
            f"predictor epoch {epoch + 1}: validation macro-F1={score:.4f}", flush=True
        )
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in predictor.state_dict().items()
            }
    assert best_state is not None
    predictor.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "validation_macro_f1": best_score,
            "train_rows": train,
            "valid_rows": valid,
        },
        out_dir / "action_predictor.pt",
    )

    vae = EvalVAE(args.vae_latent_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(args.vae_epochs):
        vae.train()
        permutation = rng.permutation(train)
        total = 0.0
        for begin in range(0, len(permutation), args.evaluator_batch):
            batch_rows = permutation[begin : begin + args.evaluator_batch]
            x = gray_tensor(data, batch_rows + 1).to(DEVICE)
            reconstruction, mu, logvar = vae(x)
            reconstruction_loss = F.binary_cross_entropy(reconstruction, x)
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
            loss = reconstruction_loss + args.vae_beta * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch_rows)
        print(f"VAE epoch {epoch + 1}: loss={total / len(permutation):.6f}", flush=True)
    torch.save(
        {
            "state_dict": vae.state_dict(),
            "latent_dim": args.vae_latent_dim,
            "beta": args.vae_beta,
            "train_rows": train,
        },
        out_dir / "unconditional_vae.pt",
    )
    return predictor.eval(), vae.eval(), best_score


def load_evaluators(directory: Path):
    predictor_checkpoint = torch.load(
        directory / "action_predictor.pt", map_location="cpu", weights_only=False
    )
    vae_checkpoint = torch.load(
        directory / "unconditional_vae.pt", map_location="cpu", weights_only=False
    )
    predictor = ActionPredictor().to(DEVICE).eval()
    predictor.load_state_dict(predictor_checkpoint["state_dict"])
    vae = EvalVAE(vae_checkpoint["latent_dim"]).to(DEVICE).eval()
    vae.load_state_dict(vae_checkpoint["state_dict"])
    return predictor, vae, float(predictor_checkpoint["validation_macro_f1"])


@torch.no_grad()
def generate(args, data: FactualData, out_dir: Path) -> None:
    rng = np.random.default_rng(args.seed)
    world = WorldModel(
        str(args.model_folder), args.num_inference_steps, 1.0, "bf16", args.vae_chunk
    )
    pool = data.starts()
    pool = pool[(data.episode_ids[pool] % 5) == 0]
    if len(pool) < args.n:
        raise ValueError(f"requested {args.n} roots but only {len(pool)} are eligible")
    roots = rng.choice(pool, size=args.n, replace=False)
    output = {
        name: []
        for name in (
            "factual",
            "cwm_composition",
            "wm_composition",
            "cwm_counterfactual",
            "wm_counterfactual",
            "factual_actions",
            "target_actions",
        )
    }
    for begin in range(0, len(roots), args.generation_batch):
        batch_roots = roots[begin : begin + args.generation_batch]
        rows = np.stack(
            [np.arange(root - BUFFER_SIZE + 1, root + 2) for root in batch_roots]
        )
        images = data.images(rows).view(
            len(batch_roots), BUFFER_SIZE + 1, 3, HEIGHT, WIDTH
        )
        context = world.encode(images[:, :BUFFER_SIZE].flatten(0, 1), sample=True)
        context = context.view(len(batch_roots), BUFFER_SIZE, *context.shape[1:])
        target = world.encode(images[:, BUFFER_SIZE], sample=False)
        factual_actions = data.actions[batch_roots]
        alternative_actions = np.asarray(
            [
                rng.choice([action for action in range(3) if action != factual])
                for factual in factual_actions
            ]
        )
        factual = torch.as_tensor(factual_actions, device=DEVICE).unsqueeze(1)
        alternative = torch.as_tensor(alternative_actions, device=DEVICE).unsqueeze(1)
        recovered = world.abduct(target, context, factual)
        fresh = torch.randn_like(target) * world.scheduler.init_noise_sigma
        output["cwm_composition"].append(
            world.decode_gray(world.predict(recovered, context, factual))
        )
        output["wm_composition"].append(
            world.decode_gray(world.predict(fresh, context, factual))
        )
        output["cwm_counterfactual"].append(
            world.decode_gray(world.predict(recovered, context, alternative))
        )
        output["wm_counterfactual"].append(
            world.decode_gray(world.predict(fresh, context, alternative))
        )
        output["factual"].append(to_gray_uint8(images[:, BUFFER_SIZE].to(DEVICE)))
        output["factual_actions"].append(factual_actions)
        output["target_actions"].append(alternative_actions)
        print(f"generated {begin + len(batch_roots)}/{len(roots)}", flush=True)
    np.savez_compressed(
        out_dir / "generated_samples.npz",
        rows=roots,
        **{name: np.concatenate(values) for name, values in output.items()},
    )


@torch.no_grad()
def predictor_f1(model, images, targets, batch_size):
    x = uint8_tensor(images)
    predictions = []
    for begin in range(0, len(x), batch_size):
        predictions.append(
            model(x[begin : begin + batch_size].to(DEVICE)).argmax(1).cpu()
        )
    return macro_f1(targets, torch.cat(predictions).numpy())


@torch.no_grad()
def encode_vae(model, images, batch_size):
    means, logvars = [], []
    for begin in range(0, len(images), batch_size):
        mean, logvar = model.encode(images[begin : begin + batch_size].to(DEVICE))
        means.append(mean.cpu().numpy())
        logvars.append(logvar.cpu().numpy())
    return np.concatenate(means), np.concatenate(logvars)


def load_official_module(repository: Path, metric: str):
    path = repository / "counterfactual_benchmark/evaluation/metrics" / f"{metric}.py"
    if not path.is_file():
        raise FileNotFoundError(f"official metric implementation not found: {path}")
    spec = importlib.util.spec_from_file_location(
        f"official_counterfactual_{metric}", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def official_fid(repository: Path, real, generated, batch_size):
    def rgbify(image, normalized=True):
        if image.shape[1] == 1:
            if normalized and torch.min(image) < -0.5:
                image = (image + 1) / 2
            image = image.repeat(1, 3, 1, 1)
        return torch.clamp(image, 0, 1)

    models = sys.modules.setdefault("models", types.ModuleType("models"))
    utils = types.ModuleType("models.utils")
    utils.rgbify = rgbify
    models.utils = utils
    sys.modules["models.utils"] = utils
    module = load_official_module(repository, "fid")
    real_x = uint8_tensor(real, size=real.shape[-2:])
    generated_x = uint8_tensor(generated, size=generated.shape[-2:])
    real_batches = [
        {"image": real_x[i : i + batch_size]} for i in range(0, len(real_x), batch_size)
    ]
    generated_batches = [
        generated_x[i : i + batch_size] for i in range(0, len(generated_x), batch_size)
    ]
    return float(module.fid(real_batches, generated_batches))


def official_cld(
    repository,
    factual_mean,
    factual_logvar,
    generated_mean,
    generated_logvar,
    factual_actions,
    target_actions,
):
    module = load_official_module(repository, "minimality")
    factual_features = np.stack([factual_mean, 0.5 * factual_logvar], axis=1)
    generated_features = np.stack([generated_mean, 0.5 * generated_logvar], axis=1)
    one_hot = np.eye(3, dtype=np.float32)
    real = list(zip(factual_features, one_hot[factual_actions]))
    generated = list(zip(generated_features, one_hot[target_actions]))
    scores, _, _ = module.minimality(
        real=real,
        generated=generated,
        interventions=["action"] * len(real),
        bins={},
        embedding="vae",
    )
    return float(np.mean(scores))


def score(args, data: FactualData, out_dir: Path, predictor, vae, predictor_f1_valid):
    samples = np.load(out_dir / "generated_samples.npz")
    factual = samples["factual"]
    cwm_cf, wm_cf = samples["cwm_counterfactual"], samples["wm_counterfactual"]
    factual_actions, target_actions = (
        samples["factual_actions"],
        samples["target_actions"],
    )

    composition = {
        "CWM": float(
            np.abs(samples["cwm_composition"].astype(np.float32) - factual).mean()
        ),
        "WM": float(
            np.abs(samples["wm_composition"].astype(np.float32) - factual).mean()
        ),
    }
    effectiveness = {
        "CWM": predictor_f1(predictor, cwm_cf, target_actions, args.evaluator_batch),
        "WM": predictor_f1(predictor, wm_cf, target_actions, args.evaluator_batch),
    }
    factual_mean, factual_logvar = encode_vae(
        vae, uint8_tensor(factual), args.evaluator_batch
    )
    cwm_mean, cwm_logvar = encode_vae(vae, uint8_tensor(cwm_cf), args.evaluator_batch)
    wm_mean, wm_logvar = encode_vae(vae, uint8_tensor(wm_cf), args.evaluator_batch)
    minimality = {
        "CWM": official_cld(
            args.benchmark_repo,
            factual_mean,
            factual_logvar,
            cwm_mean,
            cwm_logvar,
            factual_actions,
            target_actions,
        ),
        "WM": official_cld(
            args.benchmark_repo,
            factual_mean,
            factual_logvar,
            wm_mean,
            wm_logvar,
            factual_actions,
            target_actions,
        ),
    }
    previous = to_gray_uint8(data.images(samples["rows"]))

    def edge_ink(frame):
        return float((frame[120:, 250:] < 128).sum())

    event_gain = np.asarray(
        [edge_ink(after) - edge_ink(before) for before, after in zip(previous, factual)]
    )
    event_mask = event_gain > 120
    if not event_mask.any():
        raise RuntimeError("no entering-obstacle cases in the benchmark sample")
    event_region = np.s_[event_mask, 120:240, 250:320]
    obstacle = {
        "CWM": float(
            np.abs(
                cwm_cf[event_region].astype(np.float32)
                - factual[event_region].astype(np.float32)
            ).mean()
        ),
        "WM": float(
            np.abs(
                wm_cf[event_region].astype(np.float32)
                - factual[event_region].astype(np.float32)
            ).mean()
        ),
    }
    realism = {
        "CWM": official_fid(args.benchmark_repo, factual, cwm_cf, args.fid_batch),
        "WM": official_fid(args.benchmark_repo, factual, wm_cf, args.fid_batch),
    }
    result = {
        "settings": {
            "seed": args.seed,
            "generated_samples": int(len(factual)),
            "predictor_validation_macro_f1": predictor_f1_valid,
            "entering_obstacle_samples": int(event_mask.sum()),
            "num_inference_steps": args.num_inference_steps,
            "evaluator_train_per_class": args.evaluator_train_per_class,
            "evaluator_valid_per_class": args.evaluator_valid_per_class,
            "predictor_epochs": args.predictor_epochs,
            "vae_epochs": args.vae_epochs,
            "vae_latent_dim": args.vae_latent_dim,
            "vae_beta": args.vae_beta,
        },
        "composition": composition,
        "effectiveness": effectiveness,
        "realism": realism,
        "minimality": minimality,
        "obstacle_preservation_l1": obstacle,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--ttc-cache", type=Path, required=True)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path)
    parser.add_argument("--reuse-generated", action="store_true")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--generation-batch", type=int, default=8)
    parser.add_argument("--vae-chunk", type=int, default=4)
    parser.add_argument("--evaluator-batch", type=int, default=128)
    parser.add_argument("--fid-batch", type=int, default=32)
    parser.add_argument("--evaluator-train-per-class", type=int, default=3000)
    parser.add_argument("--evaluator-valid-per-class", type=int, default=1000)
    parser.add_argument("--predictor-epochs", type=int, default=5)
    parser.add_argument("--vae-epochs", type=int, default=8)
    parser.add_argument("--vae-latent-dim", type=int, default=32)
    parser.add_argument("--vae-beta", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = FactualData(args.dataset_path, args.ttc_cache)
    if args.evaluator_dir:
        predictor, vae, predictor_valid_f1 = load_evaluators(args.evaluator_dir)
    else:
        predictor, vae, predictor_valid_f1 = train_evaluators(args, data, args.out_dir)
    if not args.reuse_generated:
        generate(args, data, args.out_dir)
    score(args, data, args.out_dir, predictor, vae, predictor_valid_f1)


if __name__ == "__main__":
    main()
