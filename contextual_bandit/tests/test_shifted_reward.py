"""Checks specific to applying a reward shift during training, not just plotting."""

import unittest
import numpy as np
from numpy.polynomial.hermite import hermgauss
import torch

from ..environment import ContextualBandit, BanditConfig
from ..estimators import leave_one_out, sample_branches
from ..models import Agent
from ..train import evaluate


class ShiftedRewardTests(unittest.TestCase):
    def test_expected_payoff_and_optimum_by_gaussian_quadrature(self):
        nodes, weights = hermgauss(4)
        contexts, actions = np.meshgrid(np.arange(21), np.arange(21), indexing="ij")
        for sigma in (0.0, 0.1, 10.0, 30.0, 60.0, 100.0):
            env = ContextualBandit(BanditConfig(noise_std=sigma), 0)
            expected = np.zeros((21, 21))
            for node, weight in zip(nodes, weights / np.sqrt(np.pi)):
                state = env.transition(
                    contexts, actions, np.full((21, 21), node * np.sqrt(2) * sigma)
                )
                expected += weight * env.reward(state)
            np.testing.assert_allclose(
                expected,
                -((contexts - actions) ** 2),
                atol=max(4e-4, 8 * np.finfo(np.float32).eps * (400 + sigma**2)),
                rtol=2e-6,
            )
            np.testing.assert_array_equal(
                env.expected_payoff(), -((contexts - actions) ** 2)
            )
            np.testing.assert_array_equal(
                env.expected_payoff().argmax(1), np.arange(21)
            )
            np.testing.assert_array_equal(
                env.expected_payoff().diagonal(), np.zeros(21)
            )

    def test_shift_applied_to_every_real_and_generated_branch(self):
        class FixedModel:
            def sample(self, c, a, rng, **kwargs):
                return (a - 10).astype(np.float32) + 0.5

        contexts = np.array([0, 10, 20])
        p = np.full((3, 21), 1 / 21)
        env = ContextualBandit(BanditConfig(noise_std=30.0), 0)
        for method in ("rloo", "rlb", "rloo_wm", "rlb_cwm"):
            batch = sample_branches(
                env, p, contexts, np.random.default_rng(19), method, 4, FixedModel()
            )
            unshifted = -(((contexts[:, None] - 10) - batch.next_states) ** 2)
            np.testing.assert_allclose(
                batch.rewards, unshifted.astype(np.float32) + 900.0, rtol=0, atol=0
            )
            root_noise = batch.next_states[:, 0] - (batch.actions[:, 0] - 10)
            rewards, _ = env.outcome(contexts, batch.actions[:, 0], root_noise)
            np.testing.assert_allclose(
                rewards, batch.rewards[:, 0], atol=0.002, rtol=2e-6
            )

    def test_leave_one_out_cancels_constant_but_plain_scores_change(self):
        rewards = torch.tensor([[-4.0, -9.0, -16.0, -25.0]], dtype=torch.float64)
        torch.testing.assert_close(
            leave_one_out(rewards + 900.0), leave_one_out(rewards), atol=1e-12, rtol=0
        )
        scores = torch.tensor(
            [[0.5, -0.5], [0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]], dtype=torch.float64
        )
        self.assertFalse(
            torch.equal(
                (rewards.T * scores).mean(0), ((rewards + 900.0).T * scores).mean(0)
            )
        )

    def test_evaluation_is_actual_shifted_expected_reward(self):
        agent = Agent(21, 32, "baseline")
        env = ContextualBandit(BanditConfig(noise_std=30.0), 0)
        stats = evaluate(agent, torch.eye(21), env)
        with torch.no_grad():
            p = agent.policy(torch.eye(21)).softmax(-1).double().numpy()
        p /= p.sum(-1, keepdims=True)
        self.assertAlmostEqual(
            stats["expected_return"], float((p * env.expected_payoff()).sum(1).mean())
        )
        self.assertEqual(stats["expected_return"], -stats["regret"])


if __name__ == "__main__":
    unittest.main()
