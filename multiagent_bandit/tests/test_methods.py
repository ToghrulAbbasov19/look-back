"""Exact gradients on small cases, branch coupling, interaction counts, and training checks."""

from dataclasses import asdict
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ..agent import Agent
from ..environment import MultiAgentBandit, MultiAgentConfig
from ..models import Agents
from ..sampling import (
    BRANCH_METHODS,
    budget_schedule,
    categorical,
    alternative_actions,
    group_cost,
    loo_advantages,
    sample_branches,
)
from ..train import TrainConfig, losses_for_batch, probabilities_all_contexts, run


def full_score(probabilities, context, action):
    score = np.eye(len(probabilities))[action] - probabilities[context]
    weight = np.zeros_like(probabilities)
    weight[context] = score
    return np.r_[weight.ravel(), score]


class CountingEnvironment(MultiAgentBandit):
    def __init__(self, config, seed):
        super().__init__(config, seed)
        self.evaluations = []

    def reward(self, contexts, actions):
        self.evaluations.append((contexts.copy(), actions.copy()))
        return super().reward(contexts, actions)


class RecordingModel:
    """Only consumes the documented observed arguments; records each draw."""

    def __init__(self, kind, cfg):
        self.kind, self.n, self.num_agents = kind, cfg.n, cfg.num_agents
        self.calls = []

    def sample(self, contexts, actions, rng, **observed):
        self.calls.append((contexts.copy(), actions.copy(), observed))
        return rng.uniform(0.0, 10.0, size=contexts.shape)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = MultiAgentConfig(3, n=1, feedback_dim=4)
        self.p = np.random.default_rng(12).dirichlet(np.ones(3), size=(3, 3))

    def sample(self, method, groups=19, M=4):
        env = CountingEnvironment(self.cfg, 8)
        batch = sample_branches(
            env,
            self.p,
            groups,
            method,
            M,
            np.random.default_rng(21),
            np.random.default_rng(47),
        )
        return env, batch

    def test_exact_environment_accounting_including_shared_root(self):
        for method in ("reinforce", "reinforce_baseline", "cca", "rloo", "rlb"):
            env, batch = self.sample(method)
            expected = 19 * group_cost(method, 3, 4, num_actions=3)
            self.assertEqual(batch.environment_interactions, expected)
            actual_calls = sum(len(c) for c, a in env.evaluations)
            self.assertEqual(actual_calls, expected)
            self.assertEqual(batch.factual_environment_interactions, 19)
            self.assertEqual(
                batch.actions.shape[-1], 4 if method in ("rloo", "rlb") else 1
            )

    def test_rlb_fixes_other_agents_and_cancels_their_payoffs(self):
        env, batch = self.sample("rlb")
        cs, ac = [x.reshape(3, 19, 3, 3) for x in env.evaluations[1]]
        for i, (contexts, actions) in enumerate(zip(cs, ac)):
            np.testing.assert_array_equal(
                contexts, np.repeat(batch.root_contexts[:, None, :], 3, axis=1)
            )
            other = [j for j in range(3) if j != i]
            np.testing.assert_array_equal(
                actions[:, :, other],
                np.repeat(batch.root_actions[:, None, other], 3, axis=1),
            )
        own_rewards = -((batch.contexts - batch.actions).astype(float) ** 2)
        np.testing.assert_allclose(
            loo_advantages(batch.rewards), loo_advantages(own_rewards), atol=1e-6
        )

    def test_rloo_resamples_other_agents_but_keeps_agent_context(self):
        env, batch = self.sample("rloo")
        cs, ac = [x.reshape(3, 19, 3, 3) for x in env.evaluations[1]]
        for i, (contexts, actions) in enumerate(zip(cs, ac)):
            np.testing.assert_array_equal(
                contexts[:, :, i], np.repeat(batch.root_contexts[:, None, i], 3, axis=1)
            )
            others = [j for j in range(3) if j != i]
            self.assertTrue(
                np.any(contexts[:, :, others] != batch.root_contexts[:, None, others])
            )

    def test_learned_branches_only_receive_the_agents_own_information(self):
        for method, kind in (("rloo_wm", "wm"), ("rlb_cwm", "cwm")):
            env, model = (
                CountingEnvironment(self.cfg, 8),
                RecordingModel(kind, self.cfg),
            )
            batch = sample_branches(
                env,
                self.p,
                19,
                method,
                4,
                np.random.default_rng(21),
                np.random.default_rng(47),
                model=model,
                model_rng=np.random.default_rng(53),
            )
            self.assertEqual(len(env.evaluations), 1)
            self.assertEqual(batch.environment_interactions, 19)
            self.assertEqual(batch.model_samples, 19 * 3 * 3)
            self.assertEqual(len(model.calls), 1)
            contexts, actions, observed = model.calls[0]
            np.testing.assert_array_equal(contexts, batch.contexts[:, :, 1:])
            np.testing.assert_array_equal(actions, batch.actions[:, :, 1:])
            if kind == "wm":
                self.assertEqual(observed, {})
            else:
                self.assertEqual(
                    set(observed), {"root_contexts", "root_actions", "root_costs"}
                )
                np.testing.assert_array_equal(
                    observed["root_costs"],
                    np.broadcast_to(-batch.rewards[:, :, :1], (3, 19, 3)),
                )
                np.testing.assert_array_equal(
                    observed["root_actions"],
                    np.broadcast_to(batch.root_actions.T[:, :, None], (3, 19, 3)),
                )
            # Independent generative draws, rather than copying a factual or
            # shared predicted value across all alternative actions.
            self.assertTrue(np.all(np.var(batch.rewards[:, :, 1:], axis=-1) > 0))
            self.assertTrue(np.any(actions != batch.actions[:, :, :1]))

    def test_policy_action_streams_pair_across_simulator_and_learned_methods(self):
        actions = {}
        for method in BRANCH_METHODS:
            model = (
                RecordingModel("wm" if method == "rloo_wm" else "cwm", self.cfg)
                if "_" in method
                else None
            )
            root_rng, branch_rng = np.random.default_rng(21), np.random.default_rng(47)
            nuisance_rng, model_rng = (
                np.random.default_rng(51),
                np.random.default_rng(53),
            )
            batches = [
                sample_branches(
                    MultiAgentBandit(self.cfg, 8),
                    self.p,
                    19,
                    method,
                    4,
                    root_rng,
                    branch_rng,
                    model=model,
                    model_rng=model_rng,
                    nuisance_rng=nuisance_rng,
                )
                for _ in range(2)
            ]
            actions[method] = [b.actions for b in batches]
        for method in BRANCH_METHODS[1:]:
            for actual, expected in zip(actions[method], actions["rloo"]):
                np.testing.assert_array_equal(actual, expected)

    def test_cumulative_schedule_has_exact_group_totals_and_bounded_rounding(self):
        for K in (8, 20, 50, 60):
            cost, budget, batch_size = 1 + K * 3, 10_000_000, 256
            groups = list(budget_schedule(budget, batch_size, cost))
            self.assertEqual(sum(groups), budget // cost)
            self.assertEqual(len(groups), 39063)
            cumulative = np.cumsum(groups) * cost
            targets = np.minimum(np.arange(1, len(groups) + 1) * batch_size, budget)
            self.assertTrue(
                np.all((targets - cumulative >= 0) & (targets - cumulative < cost))
            )
            self.assertLess(budget - sum(groups) * cost, cost)
        groups = list(budget_schedule(10_000, 256, 1201))
        self.assertEqual(sum(groups), 10_000 // 1201)

    def test_precomputed_cdf_matches_expanded_sampling_including_boundaries(self):
        rng = np.random.default_rng(89)
        for S in (3, 21):
            for dtype in (np.float32, np.float64):
                probabilities = rng.dirichlet(np.ones(S), size=(4, S)).astype(dtype)
                probabilities[0, 0] = 0.0
                probabilities[0, 0, -1] = 1.0
                contexts = rng.integers(S, size=(4, 71, 4))
                contexts[0, :2] = 0
                uniforms = rng.random(contexts.shape)
                uniforms[0, 0] = 0.0
                uniforms[0, 1] = np.nextafter(1.0, 0.0)
                identity = np.arange(4)[:, None, None]
                # Probe equality with interior cumulative thresholds as well
                # as zero-mass plateaus and the largest permitted uniform.
                uniforms[:, 2] = probabilities.cumsum(-1)[
                    identity[:, 0], contexts[:, 2], 1
                ]
                expected = categorical(probabilities[identity, contexts], uniforms)
                np.testing.assert_array_equal(
                    alternative_actions(probabilities, contexts, uniforms), expected
                )

    def test_feedback_noise_does_not_change_reward(self):
        env = MultiAgentBandit(self.cfg, 3)
        c = np.array([[0, 1, 2], [2, 1, 0]])
        a = np.array([[2, 2, 1], [0, 1, 0]])
        r0, f0 = env.outcome(c, a, np.zeros((2, 3, 4)))
        r1, f1 = env.outcome(c, a, np.ones((2, 3, 4)))
        np.testing.assert_array_equal(r0, [-6.0, -4.0])
        np.testing.assert_array_equal(r0, r1)
        np.testing.assert_allclose(f1 - f0, 1.0, atol=1e-6)


class ExactEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.p = np.random.default_rng(17).dirichlet(np.ones(3), size=(2, 3))
        self.q = -((np.arange(3)[:, None] - np.arange(3)[None, :]).astype(float) ** 2)
        v = (self.p[0] * self.q).sum(-1)
        weight = self.p[0] * (self.q - v[:, None]) / 3
        self.truth = np.r_[weight.ravel(), weight.sum(0)]

    def agent_moments(self, method):
        p, q = self.p, self.q
        expected = np.zeros(12)
        second = 0.0
        # All contexts/actions enumerated; M=2. Other agents are integrated
        # under policies fixed for this coordinate calculation.
        for c, a0, a1, d0, b0 in itertools.product(range(3), repeat=5):
            tail = (
                [(d0, b0)] if method == "rlb" else itertools.product(range(3), repeat=2)
            )
            for d1, b1 in tail:
                mass = p[0, c, a0] * p[0, c, a1] * p[1, d0, b0] / 9
                if method == "rloo":
                    mass *= p[1, d1, b1] / 3
                rewards = np.array([q[c, a0] + q[d0, b0], q[c, a1] + q[d1, b1]])
                scores = np.array([full_score(p[0], c, a0), full_score(p[0], c, a1)])
                gradient = (loo_advantages(rewards)[:, None] * scores).mean(0)
                expected += mass * gradient
                second += mass * np.dot(gradient, gradient)
        return expected, second - np.dot(expected, expected)

    def test_rloo_rlb_unbiased_and_rlb_reduces_exact_variance(self):
        ordinary, ordinary_variance = self.agent_moments("rloo")
        counterfactual, counterfactual_variance = self.agent_moments("rlb")
        np.testing.assert_allclose(ordinary, self.truth, atol=2e-14)
        np.testing.assert_allclose(counterfactual, self.truth, atol=2e-14)
        self.assertLess(counterfactual_variance, ordinary_variance)

    def test_reinforce_and_action_independent_baseline_unbiased(self):
        p, q = self.p, self.q
        means = [np.zeros(12), np.zeros(12)]
        baseline = np.array([4.0, -3.0, -25.0])
        for c, d, a, b in itertools.product(range(3), repeat=4):
            mass = p[0, c, a] * p[1, d, b] / 9
            score = full_score(p[0], c, a)
            reward = q[c, a] + q[d, b]
            means[0] += mass * reward * score
            means[1] += mass * (reward - baseline[c]) * score
        for mean in means:
            np.testing.assert_allclose(mean, self.truth, atol=2e-14)


class LearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_baseline_and_cca_match_independent_copied_agents(self):
        torch.manual_seed(7)
        for method in ("reinforce_baseline", "cca"):
            combined = Agents(2, 3, 4, method, seed=9).double()
            x = torch.eye(3, dtype=torch.float64)[torch.randint(3, (2, 5))]
            a = torch.randint(3, (2, 5))
            r, f = (
                torch.randn(2, 5, dtype=torch.float64),
                torch.randn(2, 5, 4, dtype=torch.float64),
            )
            actual = combined.losses(x, a, r, f)
            total = sum(
                actual[key].sum()
                for key in ("policy", "value", "supervised", "independence")
            )
            total.backward()
            for i in range(2):
                single = Agent(3, 4, "cca" if method == "cca" else "baseline").double()
                single.load_state_dict(
                    {
                        key: combined.params[key.replace(".", "__")][i].detach()
                        for key in single.state_dict()
                    }
                )
                expected = single.losses(x[i], a[i], r[i], f[i])
                sum(
                    expected[key]
                    for key in ("policy", "value", "supervised", "independence")
                ).backward()
                for key in actual:
                    torch.testing.assert_close(actual[key][i], expected[key])
                for key, parameter in single.named_parameters():
                    torch.testing.assert_close(
                        combined.params[key.replace(".", "__")].grad[i], parameter.grad
                    )

    def test_plain_and_branching_losses_never_train_value_head(self):
        env = MultiAgentBandit(MultiAgentConfig(2, n=1, feedback_dim=4), 3)
        p = np.full((2, 3, 3), 1 / 3)
        for method in ("reinforce", "rloo", "rlb"):
            agents = Agents(2, 3, 4, method)
            batch = sample_branches(
                env, p, 5, method, 4, np.random.default_rng(7), np.random.default_rng(8)
            )
            losses = losses_for_batch(agents, batch, method, torch.eye(3), None, 0.0)
            losses["policy"].sum().backward()
            self.assertIsNone(agents.params["value__weight"].grad)
            self.assertIsNone(agents.params["value__bias"].grad)
            self.assertGreater(
                float(agents.params["policy__weight"].grad.abs().sum()), 0.0
            )

    def test_actor_loss_gradient_matches_all_branch_score_formula(self):
        env = MultiAgentBandit(MultiAgentConfig(2, n=1, feedback_dim=4), 3)
        eye = torch.eye(3, dtype=torch.float64)
        for method in ("reinforce", "rloo", "rlb"):
            agents = Agents(2, 3, 4, method, seed=11).double()
            probabilities = probabilities_all_contexts(agents, eye)
            batch = sample_branches(
                env,
                probabilities,
                7,
                method,
                4,
                np.random.default_rng(7),
                np.random.default_rng(8),
            )
            loss = losses_for_batch(agents, batch, method, eye, None, 0.0)["policy"]
            loss.sum().backward()
            advantages = (
                batch.rewards
                if method == "reinforce"
                else loo_advantages(batch.rewards)
            )
            for i in range(2):
                terms = [
                    full_score(probabilities[i], c, a) * float(v)
                    for c, a, v in zip(
                        batch.contexts[i].ravel(),
                        batch.actions[i].ravel(),
                        advantages[i].ravel(),
                    )
                ]
                expected = -np.mean(terms, axis=0)
                actual = np.r_[
                    agents.params["policy__weight"].grad[i].numpy().T.ravel(),
                    agents.params["policy__bias"].grad[i].numpy(),
                ]
                np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_sparse_policy_objective_matches_expanded_objective_and_gradients(self):
        env = MultiAgentBandit(MultiAgentConfig(8, n=10), 3)
        for dtype in (torch.float32, torch.float64):
            eye = torch.eye(21, dtype=dtype)
            for method in ("reinforce", "rloo", "rlb"):
                dense, sparse = [
                    Agents(8, 21, 32, method, seed=11).to(dtype) for _ in range(2)
                ]
                probabilities = probabilities_all_contexts(dense, eye)
                batch = sample_branches(
                    env,
                    probabilities,
                    19,
                    method,
                    4,
                    np.random.default_rng(7),
                    np.random.default_rng(8),
                )
                contexts = eye[torch.from_numpy(batch.contexts.reshape(8, -1))]
                actions = torch.from_numpy(batch.actions.reshape(8, -1))
                advantages = torch.from_numpy(
                    (
                        batch.rewards
                        if method == "reinforce"
                        else loo_advantages(batch.rewards)
                    ).reshape(8, -1)
                )
                logp = dense.policy(contexts).log_softmax(-1)
                entropy = -(logp.exp() * logp).sum(-1).mean(-1)
                dense_loss = (
                    -(
                        logp.gather(-1, actions[..., None]).squeeze(-1) * advantages
                    ).mean(-1)
                    - 0.004 * entropy
                )
                sparse_loss = losses_for_batch(sparse, batch, method, eye, None, 0.004)[
                    "policy"
                ]
                torch.testing.assert_close(dense_loss, sparse_loss, rtol=0, atol=0)
                dense_loss.sum().backward()
                sparse_loss.sum().backward()
                a = torch.cat(
                    [p.grad.ravel() for p in dense.parameters() if p.grad is not None]
                )
                b = torch.cat(
                    [p.grad.ravel() for p in sparse.parameters() if p.grad is not None]
                )
                self.assertLess(
                    float((a - b).norm() / a.norm()),
                    4e-7 if dtype == torch.float32 else 1e-14,
                )

    def test_training_budget_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as folder:
            results = {}
            for method in ("reinforce", "reinforce_baseline", "cca", "rloo", "rlb"):
                output = Path(folder) / method
                job = {
                    "method": method,
                    "M": 2,
                    "seed": 6,
                    "environment": asdict(MultiAgentConfig(2, n=1, feedback_dim=4)),
                    "training": asdict(
                        TrainConfig(budget=41, batch_size=10, log_every=2)
                    ),
                    "output": str(output),
                }
                result = run(job)
                self.assertEqual(result["environment_interactions"], 41)
                self.assertEqual(result["charged_interactions"], 41)
                self.assertEqual(
                    result["unused_tail_interactions"],
                    2 if method in ("rloo", "rlb") else 0,
                )
                self.assertAlmostEqual(
                    result["joint_expected_return"], 2 * result["expected_return"]
                )
                rows = [
                    json.loads(line)
                    for line in (output / "progress.jsonl").read_text().splitlines()
                ]
                self.assertEqual(rows[-1]["environment_interactions"], 41)
                self.assertTrue((output / "config.json").exists())
                results[method] = result
                with self.assertRaises(FileExistsError):
                    run(job)
            replay = run({**job, "output": str(Path(folder) / "replay")})
            for key in replay.keys() - {"seconds"}:
                self.assertEqual(replay[key], results["rlb"][key])

    def test_learned_training_counts_both_budgets_and_freezes_model(self):
        from ..world_models import DensityConfig, TransitionDensity, density_checkpoint
        import hashlib

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for method, kind in (("rloo_wm", "world"), ("rlb_cwm", "counterfactual")):
                model = TransitionDensity(
                    DensityConfig(
                        kind=kind,
                        n=1,
                        num_agents=2,
                        hidden=4,
                        cost_center=3.0,
                        cost_scale=2.0,
                        delta_scale=1.0,
                    )
                )
                checkpoint = folder / (method + ".pt")
                torch.save(
                    density_checkpoint(
                        model, training={"offline_environment_interactions": 300}
                    ),
                    checkpoint,
                )
                before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                for budget_mode in ("all", "real"):
                    output = folder / (method + "_" + budget_mode)
                    result = run(
                        {
                            "method": method,
                            "M": 2,
                            "seed": 3,
                            "budget_mode": budget_mode,
                            "model_checkpoint": str(checkpoint),
                            "output": str(output),
                            "environment": asdict(
                                MultiAgentConfig(2, n=1, feedback_dim=4)
                            ),
                            "training": asdict(
                                TrainConfig(budget=41, batch_size=10, log_every=2)
                            ),
                        }
                    )
                    self.assertEqual(result["charged_interactions"], 41)
                    self.assertEqual(
                        result["environment_interactions"],
                        15 if budget_mode == "all" else 41,
                    )
                    self.assertEqual(
                        result["model_samples"], 26 if budget_mode == "all" else 82
                    )
                    self.assertEqual(
                        result["unused_tail_interactions"],
                        2 if budget_mode == "all" else 0,
                    )
                self.assertEqual(
                    hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before
                )
                self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
