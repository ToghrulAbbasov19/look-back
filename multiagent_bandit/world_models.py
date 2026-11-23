"""Per-agent WM and CWM: truncated-Gaussian densities of the joint cost.

Inputs are the agent's own context and action (plus the simulated episode's
action and cost for the CWM); never the other agents' contexts or actions.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DensityConfig:
    kind: str = "world"
    n: int = 10
    num_agents: int = 8
    hidden: int = 16
    cost_center: float = 0.0
    cost_scale: float = 1.0
    delta_scale: float = 1.0
    min_scale: float = 0.05
    seed: int = 0


def _log_sub_exp(big, small):
    # clamp protects an exactly rounded zero interval in extreme tails.
    delta = (small - big).clamp(max=-torch.finfo(big.dtype).eps)
    return big + torch.log(-torch.expm1(delta))


def truncation(location, scale, upper):
    """Standardized bounds and stable log probability of physical support."""
    location, scale = location.double(), scale.double()
    a, b = -location / scale, (upper - location) / scale
    reflected = a > 0
    left = torch.where(reflected, -b, a)
    right = torch.where(reflected, -a, b)
    log_z = _log_sub_exp(torch.special.log_ndtr(right), torch.special.log_ndtr(left))
    return a, b, log_z


def truncated_moments(location, scale, upper):
    """Mean/std of the bounded distribution, rather than Gaussian location/scale."""
    a, b, log_z = truncation(location, scale, upper)
    pa = torch.exp(-0.5 * a.square() - 0.5 * math.log(2 * math.pi) - log_z)
    pb = torch.exp(-0.5 * b.square() - 0.5 * math.log(2 * math.pi) - log_z)
    offset = pa - pb
    mean = location.double() + scale.double() * offset
    variance = scale.double().square() * (1 + a * pa - b * pb - offset.square()).clamp(
        min=0.0
    )
    # A one-sided extreme tail loses variance by cancellation in 1+a*m-m².
    # When the remote bound changes mass by less than exp(-50), the inverse
    # Mills-ratio expansion gives stable boundary distance and variance.
    distance = torch.maximum(a, -b)
    extreme = (distance > 30.0) & ((b - a) * distance > 50.0)
    if bool(extreme.any()):
        t = distance[extreme].reciprocal()
        q = t.square()
        excess = t * (1 + q * (-2 + q * (10 + q * (-74 + q * (706 - 8162 * q)))))
        var = q * (1 + q * (-6 + q * (50 + q * (-518 + q * (6354 - 89782 * q)))))
        left_mean = scale.double()[extreme] * excess
        mean[extreme] = torch.where(a[extreme] > 0, left_mean, upper - left_mean)
        variance[extreme] = scale.double()[extreme].square() * var
    return mean.clamp(0.0, upper), variance.sqrt()


def _inverse_log_cdf(log_p):
    """Inverse normal CDF without underflow in small tail probabilities."""
    ordinary = torch.special.ndtri(log_p.exp().clamp(min=1e-300, max=1 - 1e-15))
    mask = log_p < -30.0
    if not bool(mask.any()):
        return ordinary
    selected = log_p[mask]
    tail = -torch.sqrt((-2 * selected).clamp(min=1.0))
    for _ in range(7):
        log_cdf = torch.special.log_ndtr(tail)
        derivative = torch.exp(
            -0.5 * tail.square() - 0.5 * math.log(2 * math.pi) - log_cdf
        )
        tail = tail - (log_cdf - selected) / derivative
    ordinary[mask] = tail
    return ordinary


def truncated_sample(location, scale, upper, uniforms):
    """Inverse-CDF sampling, stable when Gaussian location is outside support."""
    location, scale = location.double(), scale.double()
    a, b = -location / scale, (upper - location) / scale
    reflected = a > 0
    left, right = torch.where(reflected, -b, a), torch.where(reflected, -a, b)
    u = torch.as_tensor(uniforms, dtype=torch.float64, device=location.device)
    u = u.clamp(torch.finfo(u.dtype).eps, 1 - torch.finfo(u.dtype).eps)
    # Reflection reverses quantiles, so both tails remain monotonic in u.
    u = torch.where(reflected, 1 - u, u)
    p_left, p_right = torch.special.ndtr(left), torch.special.ndtr(right)
    probability = p_left + (p_right - p_left) * u
    ordinary = (
        (probability > 1e-14) & (probability < 1 - 1e-14) & ((p_right - p_left) > 1e-12)
    )
    z = torch.special.ndtri(probability.clamp(1e-15, 1 - 1e-15))
    if not bool(ordinary.all()):
        mask = ~ordinary
        log_p = torch.logaddexp(
            torch.special.log_ndtr(left[mask]) + torch.log1p(-u[mask]),
            torch.special.log_ndtr(right[mask]) + torch.log(u[mask]),
        )
        z[mask] = _inverse_log_cdf(log_p)
    z = torch.where(reflected, -z, z)
    return (location.double() + scale.double() * z).clamp(0.0, upper)


class TransitionDensity(nn.Module):
    """Batched independent 16×16 networks; first input axis is agent identity."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        inputs, h = (2 if config.kind == "world" else 5), config.hidden
        collected = {
            name: [] for name in ("w1", "b1", "w2", "b2", "wo", "bo", "ws", "bs")
        }
        for i in range(config.num_agents):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(
                    int(
                        np.random.SeedSequence([config.seed, 9401, i]).generate_state(
                            1
                        )[0]
                    )
                )
                first, second = nn.Linear(inputs, h), nn.Linear(h, h)
                values = {
                    "w1": first.weight.detach(),
                    "b1": first.bias.detach(),
                    "w2": second.weight.detach(),
                    "b2": second.bias.detach(),
                    "wo": torch.zeros(2, h),
                    "bo": torch.zeros(2),
                    "ws": torch.zeros(1, inputs),
                    "bs": torch.zeros(1),
                }
            for name, value in values.items():
                collected[name].append(value)
        self.params = nn.ParameterDict(
            {
                name: nn.Parameter(torch.stack(value))
                for name, value in collected.items()
            }
        )

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters()) // self.config.num_agents

    @property
    def upper(self):
        return float(4 * self.config.n**2 * self.config.num_agents)

    def _linear(self, x, weight, bias):
        return (
            torch.bmm(x, self.params[weight].transpose(1, 2))
            + self.params[bias][:, None, :]
        )

    def distribution_parameters(
        self,
        contexts,
        actions,
        *,
        root_contexts=None,
        root_actions=None,
        root_costs=None,
    ):
        p = self.params["w1"]
        c, a = [torch.as_tensor(x, device=p.device) for x in (contexts, actions)]
        shape = a.shape
        inputs = [
            (c.to(p.dtype) - self.config.n) / self.config.n,
            (a.to(p.dtype) - self.config.n) / self.config.n,
        ]
        if self.config.kind == "counterfactual":
            roots = [
                torch.as_tensor(x, device=p.device, dtype=p.dtype)
                for x in (root_contexts, root_actions, root_costs)
            ]
            rc, ra, rz = roots
            inputs += [
                (rc - self.config.n) / self.config.n,
                (ra - self.config.n) / self.config.n,
                (rz - self.config.cost_center) / self.config.cost_scale,
            ]
            origin, output_scale = rz, self.config.delta_scale
        else:
            origin, output_scale = self.config.cost_center, self.config.cost_scale
        features = torch.stack(inputs, -1).reshape(
            self.config.num_agents, -1, len(inputs)
        )
        h = F.silu(self._linear(features, "w1", "b1"))
        h = F.silu(self._linear(h, "w2", "b2"))
        out = self._linear(h, "wo", "bo")
        mean = origin + output_scale * (
            out[..., 0] + self._linear(features, "ws", "bs").squeeze(-1)
        ).reshape(shape)
        scale = (
            (
                output_scale**2 * F.softplus(out[..., 1]).square()
                + self.config.min_scale**2
            )
            .sqrt()
            .reshape(shape)
        )
        return mean, scale

    def forward(self, contexts, actions, **kwargs):
        return self.distribution_parameters(contexts, actions, **kwargs)

    def moments(self, contexts, actions, **kwargs):
        return truncated_moments(
            *self.distribution_parameters(contexts, actions, **kwargs), self.upper
        )

    def negative_log_likelihood(self, contexts, actions, costs, **kwargs):
        location, scale = self.distribution_parameters(contexts, actions, **kwargs)
        target = torch.as_tensor(costs, dtype=torch.float64, device=location.device)
        _, _, log_z = truncation(location, scale, self.upper)
        return (
            0.5 * ((target - location) / scale).square()
            + scale.double().log()
            + 0.5 * math.log(2 * math.pi)
            + log_z
        )

    @torch.no_grad()
    def sample(self, contexts, actions, generator=None, **kwargs):
        location, scale = self.distribution_parameters(contexts, actions, **kwargs)
        uniforms = torch.rand(
            location.shape,
            dtype=torch.float64,
            device=location.device,
            generator=generator,
        )
        return truncated_sample(location, scale, self.upper, uniforms)


def density_checkpoint(model, **metadata):
    return {
        "density_config": asdict(model.config),
        "model_state": model.state_dict(),
        "parameter_count_per_agent": model.parameter_count,
        **metadata,
    }


def load_density(path, *, device="cpu"):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    density = TransitionDensity(DensityConfig(**checkpoint["density_config"]))
    density.load_state_dict(checkpoint["model_state"])
    return density.to(device).eval().requires_grad_(False), checkpoint


class FrozenModel:
    """NumPy interface: distribution returns bounded mean/std, not location/scale."""

    def __init__(self, density, metadata):
        self.density = density.eval().requires_grad_(False)
        self.metadata = metadata
        self.kind = "wm" if density.config.kind == "world" else "cwm"
        self.n, self.num_agents = density.config.n, density.config.num_agents

    @torch.no_grad()
    def distribution(self, contexts, actions, **kwargs):
        mean, std = self.density.moments(contexts, actions, **kwargs)
        return mean.cpu().numpy(), std.cpu().numpy()

    @torch.no_grad()
    def distribution_parameters(self, contexts, actions, **kwargs):
        location, scale = self.density.distribution_parameters(
            contexts, actions, **kwargs
        )
        return location.cpu().numpy(), scale.cpu().numpy()

    @torch.no_grad()
    def sample(self, contexts, actions, rng, **kwargs):
        location, scale = self.density.distribution_parameters(
            contexts, actions, **kwargs
        )
        draws = truncated_sample(
            location, scale, self.density.upper, rng.random(tuple(location.shape))
        )
        return draws.cpu().numpy().astype(np.float32)


def load_model(path):
    return FrozenModel(*load_density(path))
