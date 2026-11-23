"""Independent mathematical, simulator-accounting, and training checks."""

from dataclasses import asdict
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.polynomial.hermite import hermgauss
import torch

from ..config import TrainConfig
from ..environment import ContextualBandit, BanditConfig
from ..estimators import group_cost, leave_one_out, sample_branches, score_loss
from ..models import Agent
from ..train import run


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_score_expectation_and_common_noise_variance(self):
        """Exhaustive actions and Gaussian polynomial integration; no MC tolerance."""
        p = np.array([0.2, 0.5, 0.3])
        actions = np.arange(3)
        context = 1
        score = np.eye(3) - p
        q = -((context - actions) ** 2)
        truth = (p[:, None] * q[:, None] * score).sum(0)
        nodes, weights = hermgauss(4)
        weights = weights / np.sqrt(np.pi)
        moments = {}
        for method in ("reinforce", "rloo", "rlb"):
            mean, second = np.zeros(3), 0.0
            for a, b in itertools.product(actions, repeat=2):
                for u, w in enumerate(weights):
                    for v, z in enumerate(weights):
                        eta = np.sqrt(2) * 10 * nodes[u]
                        eta_other = (
                            eta if method == "rlb" else np.sqrt(2) * 10 * nodes[v]
                        )
                        reward = (
                            np.array(
                                [
                                    -((context - a - eta) ** 2),
                                    -((context - b - eta_other) ** 2),
                                ]
                            )
                            + 10**2
                        )
                        advantages = (
                            reward if method == "reinforce" else reward - reward[::-1]
                        )
                        g = (advantages[:, None] * score[[a, b]]).mean(0)
                        probability = p[a] * p[b] * w * z
                        mean += probability * g
                        second += probability * (g @ g)
            np.testing.assert_allclose(mean, truth, rtol=1e-11, atol=1e-11)
            moments[method] = second - mean @ mean
        self.assertLess(moments["rlb"], moments["rloo"])
        self.assertLess(moments["rlb"], moments["reinforce"])

    def test_all_branch_scores_and_stop_gradient(self):
        torch.manual_seed(9)
        agent = Agent(3, 2, "baseline").double()
        context = torch.eye(3, dtype=torch.double)[[0, 2]]
        actions = torch.tensor([[0, 2, 1], [2, 0, 2]])
        reward = torch.tensor(
            [[-2.0, 4.0, -1.0], [3.0, -8.0, 2.0]],
            dtype=torch.double,
            requires_grad=True,
        )
        losses = score_loss(
            agent, context, actions, reward, branching=True, entropy_coef=0.0
        )
        losses["policy"].backward()
        p = agent.policy(context).softmax(-1).detach().numpy()
        advantage = leave_one_out(reward.detach()).numpy()
        expected = np.zeros((3, 3))
        for group, c in enumerate((0, 2)):
            for branch in range(3):
                expected[:, c] -= (
                    advantage[group, branch]
                    * (np.eye(3)[actions[group, branch]] - p[group])
                    / 6
                )
        np.testing.assert_allclose(agent.policy.weight.grad, expected, atol=1e-14)
        np.testing.assert_allclose(agent.policy.bias.grad, expected.sum(-1), atol=1e-14)
        self.assertIsNone(reward.grad)
        self.assertIsNone(agent.value.weight.grad)

    def test_coupling_and_zero_noise_equivalence(self):
        contexts = np.array([0, 1, 2] * 20)
        probabilities = np.full((len(contexts), 3), 1 / 3)
        for sigma in (0.0, 10.0):
            env = ContextualBandit(BanditConfig(n=1, noise_std=sigma), 0)
            batches = [
                sample_branches(
                    env, probabilities, contexts, np.random.default_rng(7), method, 4
                )
                for method in ("rloo", "rlb")
            ]
            np.testing.assert_array_equal(batches[0].actions, batches[1].actions)
            noise = batches[1].next_states - (batches[1].actions - 1)
            np.testing.assert_allclose(
                noise, np.broadcast_to(noise[:, :1], noise.shape), atol=3e-6
            )
            self.assertTrue(
                np.any(batches[1].actions[:, 1:] != batches[1].actions[:, :1])
            )
            if sigma == 0:
                np.testing.assert_array_equal(batches[0].rewards, batches[1].rewards)

    def test_real_and_learned_simulator_accounting(self):
        class CountingEnv(ContextualBandit):
            calls = 0

            def transition(self, c, a, w):
                self.calls += np.asarray(a).size
                return super().transition(c, a, w)

        class ConstantModel:
            def sample(self, c, a, rng, **kwargs):
                self.inputs = (c, a, kwargs)
                return np.zeros_like(a, dtype=np.float32)

        contexts = np.array([0, 1, 2])
        p = np.full((3, 3), 1 / 3)
        for method in ("rloo", "rlb", "rloo_wm", "rlb_cwm"):
            env = CountingEnv(BanditConfig(n=1), 0)
            model = ConstantModel()
            batch = sample_branches(
                env, p, contexts, np.random.default_rng(8), method, 4, model
            )
            learned = method.endswith(("wm", "cwm"))
            self.assertEqual(env.calls, 3 if learned else 12)
            self.assertEqual(batch.environment_interactions, env.calls)
            self.assertEqual(batch.model_samples, 9 if learned else 0)
            self.assertEqual(group_cost(method, 4, "all"), 4)
            self.assertEqual(group_cost(method, 4, "real"), 1 if learned else 4)
            if method == "rlb_cwm":
                self.assertEqual(
                    set(model.inputs[-1]),
                    {"root_contexts", "root_actions", "root_next_states"},
                )

    def test_validation_and_reproducible_budget(self):
        for method in ("reinforce", "reinforce_baseline", "cca", "rloo", "rlb"):
            with tempfile.TemporaryDirectory() as temp:
                job = {
                    "method": method,
                    "seed": 9,
                    "M": 4,
                    "budget_mode": "all",
                    "environment": asdict(
                        BanditConfig(n=1, feedback_dim=2, noise_std=2)
                    ),
                    "training": asdict(
                        TrainConfig(budget=96, batch_size=32, log_every=2)
                    ),
                }
                results = []
                for i in range(2):
                    result = run({**job, "output": str(Path(temp) / str(i))})
                    result.pop("seconds")
                    results.append(result)
                self.assertEqual(results[0], results[1])
                self.assertEqual(results[0]["environment_interactions"], 96)
                self.assertEqual(results[0]["charged_interactions"], 96)
                self.assertEqual(results[0]["updates"], 3)
                self.assertEqual(results[0]["model_samples"], 0)

    def test_learned_training_budgets_and_frozen_generator(self):
        from ..world_models import DensityConfig, TransitionDensity, density_checkpoint

        with tempfile.TemporaryDirectory() as temp:
            for method, kind in (("rloo_wm", "world"), ("rlb_cwm", "counterfactual")):
                density = TransitionDensity(DensityConfig(kind=kind, n=2, hidden=2))
                checkpoint = Path(temp) / f"{kind}.pt"
                torch.save(density_checkpoint(density), checkpoint)
                original = checkpoint.read_bytes()
                for budget_mode in ("all", "real"):
                    result = run(
                        {
                            "method": method,
                            "seed": 9,
                            "M": 4,
                            "budget_mode": budget_mode,
                            "model_checkpoint": str(checkpoint),
                            "environment": asdict(
                                BanditConfig(n=2, feedback_dim=2, noise_std=2)
                            ),
                            "training": asdict(
                                TrainConfig(budget=96, batch_size=32, log_every=2)
                            ),
                            "output": str(Path(temp) / f"{kind}_{budget_mode}"),
                        }
                    )
                    self.assertEqual(result["charged_interactions"], 96)
                    self.assertEqual(result["updates"], 3)
                    self.assertEqual(
                        result["environment_interactions"],
                        24 if budget_mode == "all" else 96,
                    )
                    self.assertEqual(
                        result["model_samples"], 72 if budget_mode == "all" else 288
                    )
                    self.assertEqual(
                        result["environment_interactions"] + result["model_samples"],
                        96 if budget_mode == "all" else 384,
                    )
                    self.assertEqual(checkpoint.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
