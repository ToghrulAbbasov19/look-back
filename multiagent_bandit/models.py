"""K independent agents stored as stacked parameters (leading axis = agent)."""

import numpy as np
import torch
from torch import nn

from .agent import Agent


class Agents(nn.Module):
    def __init__(
        self, num_agents, num_actions, feedback_dim, method="cca", hidden=32, seed=0
    ):
        super().__init__()
        self.method = "cca" if method == "cca" else "baseline"
        self.num_agents = num_agents
        originals = []
        # Per-agent initialization depends only on (seed, agent), so all methods start alike.
        with torch.random.fork_rng(devices=[]):
            for i in range(num_agents):
                torch.manual_seed(
                    int(np.random.SeedSequence([seed, 2, i]).generate_state(1)[0])
                )
                originals.append(Agent(num_actions, feedback_dim, self.method, hidden))
        states = [dict(a.named_parameters()) for a in originals]
        self.params = nn.ParameterDict(
            {
                name.replace(".", "__"): nn.Parameter(
                    torch.stack([d[name].detach() for d in states])
                )
                for name in states[0]
            }
        )

    def linear(self, x, name, freeze=False):
        weight = self.params[name.replace(".", "__") + "__weight"]
        bias = self.params[name.replace(".", "__") + "__bias"]
        if freeze:
            weight, bias = weight.detach(), bias.detach()
        return torch.bmm(x, weight.transpose(1, 2)) + bias[:, None, :]

    def policy(self, context):
        return self.linear(context, "policy")

    def features(self, context, feedback):
        x = torch.cat((context, feedback), dim=-1)
        x = self.linear(x, "encoder.net.0").relu()
        return self.linear(x, "encoder.net.2").relu()

    def classifier(self, x, freeze=False):
        for i in range(3):
            x = self.linear(x, f"classifier.layers.{i}", freeze)
            if i < 2:
                x = x.relu()
        return x

    def baseline(self, context, feedback=None):
        x = (
            torch.cat((context, self.features(context, feedback)), -1)
            if self.method == "cca"
            else context
        )
        return self.linear(x, "value").squeeze(-1)

    def losses(self, context, actions, rewards, feedback, entropy_coef=0.004):
        logp = self.policy(context).log_softmax(-1)
        p = logp.exp()
        entropy = -(p * logp).sum(-1).mean(-1)
        zero = rewards.new_zeros(self.num_agents)
        if self.method == "cca":
            value_inputs = torch.cat((context, self.features(context, feedback)), -1)
            value = self.linear(value_inputs, "value").squeeze(-1)
            posterior = self.classifier(value_inputs.detach()).log_softmax(-1)
            supervised = -posterior.gather(-1, actions[..., None]).squeeze(-1).mean(-1)
            frozen = self.classifier(value_inputs, freeze=True).log_softmax(-1)
            independence = (p.detach() * (logp.detach() - frozen)).sum(-1).mean(-1)
        else:
            value = self.baseline(context)
            supervised, independence = zero, zero
        advantage = rewards - value
        pg = -(
            logp.gather(-1, actions[..., None]).squeeze(-1) * advantage.detach()
        ).mean(-1)
        return {
            "policy": pg - entropy_coef * entropy,
            "value": advantage.square().mean(-1),
            "supervised": supervised,
            "independence": independence,
            "entropy": entropy,
            "advantage_mean": advantage.detach().mean(-1),
            "advantage_variance": advantage.detach().var(dim=-1, unbiased=False),
        }
