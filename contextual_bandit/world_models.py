"""The WM and CWM: small Gaussian models of a + u, fit by maximum likelihood."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DensityConfig:
    kind: str = "world"
    n: int = 10
    hidden: int = 8
    next_state_center: float = 0.0
    next_state_scale: float = 1.0
    min_scale: float = 0.05


class TransitionDensity(nn.Module):
    """Gaussian over a + u: a small SiLU MLP plus a linear skip for the mean, softplus scale.

    WM inputs: context and action. CWM inputs add the real transition's context,
    action and a + u, and its mean is a correction to the real a + u. The
    noise is never an input. Normalization constants come from the training data.
    """

    def __init__(self, config: DensityConfig):
        super().__init__()
        self.config = config
        inputs = 2 if config.kind == "world" else 5
        self.net = nn.Sequential(
            nn.Linear(inputs, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, 2),
        )
        self.mean_skip = nn.Linear(inputs, 1)
        # Start from a constant marginal mean / factual a + u. This is
        # not initialization at the known simulator equation.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.zeros_(self.mean_skip.weight)
        nn.init.zeros_(self.mean_skip.bias)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def distribution_parameters(
        self,
        contexts,
        actions,
        *,
        root_contexts=None,
        root_actions=None,
        root_next_states=None,
    ):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        contexts = torch.as_tensor(contexts, device=device)
        actions = torch.as_tensor(actions, device=device)
        inputs = [
            (contexts.to(dtype) - self.config.n) / self.config.n,
            (actions.to(dtype) - self.config.n) / self.config.n,
        ]
        if self.config.kind == "counterfactual":
            roots = [
                torch.as_tensor(x, device=device)
                for x in (root_contexts, root_actions, root_next_states)
            ]
            factual_c, factual_a, factual_y = roots
            inputs += [
                (factual_c.to(dtype) - self.config.n) / self.config.n,
                (factual_a.to(dtype) - self.config.n) / self.config.n,
                (factual_y.to(dtype) - self.config.next_state_center)
                / self.config.next_state_scale,
            ]
            origin = factual_y.to(dtype)
        else:
            origin = self.config.next_state_center
        features = torch.stack(inputs, -1)
        outputs = self.net(features)
        mean = origin + self.config.next_state_scale * (
            outputs[..., 0] + self.mean_skip(features).squeeze(-1)
        )
        excess_scale = self.config.next_state_scale * F.softplus(outputs[..., 1])
        scale = (excess_scale.square() + self.config.min_scale**2).sqrt()
        return mean, scale

    def forward(self, contexts, actions, **kwargs):
        return self.distribution_parameters(contexts, actions, **kwargs)

    def negative_log_likelihood(self, contexts, actions, next_states, **kwargs):
        mean, scale = self.distribution_parameters(contexts, actions, **kwargs)
        target = torch.as_tensor(next_states, device=mean.device, dtype=mean.dtype)
        return (
            0.5 * ((target - mean) / scale).square() + scale.log() + 0.9189385332046727
        )

    @torch.no_grad()
    def sample(
        self,
        contexts,
        actions,
        generator=None,
        *,
        root_contexts=None,
        root_actions=None,
        root_next_states=None,
    ):
        mean, scale = self.distribution_parameters(
            contexts,
            actions,
            root_contexts=root_contexts,
            root_actions=root_actions,
            root_next_states=root_next_states,
        )
        noise = torch.randn(
            mean.shape, dtype=mean.dtype, device=mean.device, generator=generator
        )
        return mean + scale * noise


def load_density(path, *, device="cpu"):
    """Load a frozen final checkpoint; return (density, full metadata)."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    model = TransitionDensity(DensityConfig(**checkpoint["density_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint


def density_checkpoint(model, **metadata):
    return {
        "density_config": asdict(model.config),
        "model_state": model.state_dict(),
        "parameter_count": model.parameter_count,
        **metadata,
    }


class FrozenModel:
    """NumPy interface used for sampling during policy training."""

    def __init__(self, density, metadata):
        self.density = density.eval().requires_grad_(False)
        self.metadata = metadata
        self.kind = "wm" if density.config.kind == "world" else "cwm"
        self.n = density.config.n

    @torch.no_grad()
    def distribution(self, contexts, actions, **kwargs):
        mean, scale = self.density.distribution_parameters(contexts, actions, **kwargs)
        return mean.cpu().numpy(), scale.cpu().numpy()

    def sample(self, contexts, actions, rng, **kwargs):
        mean, scale = self.distribution(contexts, actions, **kwargs)
        return (mean + scale * rng.normal(size=mean.shape)).astype(np.float32)


def load_model(path):
    density, metadata = load_density(path)
    return FrozenModel(density, metadata)
