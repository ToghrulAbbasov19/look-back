"""One agent: policy plus a state baseline (REINFORCE + baseline) or CCA's hindsight baseline."""

import torch
from torch import nn
from torch.nn import functional as F


class HindsightEncoder(nn.Module):
    """Encodes [one_hot(context), feedback] with two ReLU layers.

    The sampled action and the reward are never direct inputs.
    """

    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, context, feedback):
        return self.net(torch.cat((context, feedback), dim=-1))


class HindsightClassifier(nn.Module):
    def __init__(self, input_dim, num_actions, hidden=32):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden),
                nn.Linear(hidden, hidden),
                nn.Linear(hidden, num_actions),
            ]
        )

    def forward(self, inputs, *, freeze_weights=False):
        for i, layer in enumerate(self.layers):
            weight = layer.weight.detach() if freeze_weights else layer.weight
            bias = layer.bias.detach() if freeze_weights else layer.bias
            inputs = F.linear(inputs, weight, bias)
            if i != len(self.layers) - 1:
                inputs = F.relu(inputs)
        return inputs


class Agent(nn.Module):
    def __init__(self, num_actions, feedback_dim, method="cca", hidden=32):
        super().__init__()
        self.method = method
        # At the decision there is no feedback; a [context, zero] linear
        # policy is equivalent to this context-only parameterization.
        self.policy = nn.Linear(num_actions, num_actions)
        if method == "cca":
            self.encoder = HindsightEncoder(num_actions + feedback_dim, hidden)
            self.value = nn.Linear(num_actions + hidden, 1)
            self.classifier = HindsightClassifier(
                num_actions + hidden, num_actions, hidden
            )
        else:
            self.value = nn.Linear(num_actions, 1)

    def baseline(self, context, feedback=None):
        if self.method == "baseline":
            return self.value(context).squeeze(-1)
        features = self.encoder(context, feedback)
        return self.value(torch.cat((context, features), dim=-1)).squeeze(-1)

    def losses(self, context, actions, reward, feedback, entropy_coef=4e-3):
        """Losses on one on-policy batch.

        Gradients: policy loss -> policy; MSE -> value and encoder;
        cross-entropy -> classifier; KL(pi || h) -> encoder.
        """
        log_policy = self.policy(context).log_softmax(-1)
        policy = log_policy.exp()
        entropy = -(policy * log_policy).sum(-1).mean()
        zero = reward.new_zeros(())
        if self.method == "cca":
            features = self.encoder(context, feedback)
            value_inputs = torch.cat((context, features), dim=-1)
            value = self.value(value_inputs).squeeze(-1)
            posterior = self.classifier(value_inputs.detach())
            supervised = F.cross_entropy(posterior, actions)
            frozen_posterior = self.classifier(
                value_inputs, freeze_weights=True
            ).log_softmax(-1)
            independence = (
                (policy.detach() * (log_policy.detach() - frozen_posterior))
                .sum(-1)
                .mean()
            )
        else:
            value = self.value(context).squeeze(-1)
            supervised, independence = zero, zero
        advantage = reward - value
        pg = -(
            log_policy.gather(-1, actions[:, None]).squeeze(-1) * advantage.detach()
        ).mean()
        return {
            "policy": pg - entropy_coef * entropy,
            "value": advantage.square().mean(),
            "supervised": supervised,
            "independence": independence,
            "entropy": entropy,
            "advantage_mean": advantage.detach().mean(),
            "advantage_variance": advantage.detach().var(unbiased=False),
        }
