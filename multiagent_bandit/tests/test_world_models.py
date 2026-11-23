"""Paired-data, density-normalization, independent-agent and checkpoint tests."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from ..offline import (
    collect_dataset,
    expand_pairs,
    load_dataset,
    train_density,
    true_marginal_moments,
)
from ..world_models import (
    DensityConfig,
    TransitionDensity,
    density_checkpoint,
    load_model,
    truncated_moments,
    truncated_sample,
    truncation,
)


class TruncatedDistributionTests(unittest.TestCase):
    def test_quadrature_moments_density_and_sample_quantiles(self):
        # This independent grid integration includes very small retained mass
        # (location=-50) and reflection into the right tail (location=55).
        x = np.linspace(0.0, 5.0, 100001)
        uniforms = torch.linspace(0.00005, 0.99995, 10000, dtype=torch.float64)
        for location, scale in (
            (2.0, 0.8),
            (-0.5, 0.4),
            (5.5, 0.4),
            (-50.0, 1.0),
            (55.0, 1.0),
            (0.0, 0.05),
        ):
            log_weight = -0.5 * ((x - location) / scale) ** 2
            weight = np.exp(log_weight - log_weight.max())
            integral = np.sum((weight[1:] + weight[:-1]) * np.diff(x) / 2)
            density = weight / integral
            numerical_mean = np.sum(
                ((x * density)[1:] + (x * density)[:-1]) * np.diff(x) / 2
            )
            centered = (x - numerical_mean) ** 2 * density
            numerical_var = np.sum((centered[1:] + centered[:-1]) * np.diff(x) / 2)
            loc, std = (
                torch.tensor(location, dtype=torch.float64),
                torch.tensor(scale, dtype=torch.float64),
            )
            mean, predicted_std = truncated_moments(loc, std, 5.0)
            self.assertAlmostEqual(float(mean), numerical_mean, delta=2e-7)
            self.assertAlmostEqual(float(predicted_std**2), numerical_var, delta=2e-7)
            _, _, log_z = truncation(loc, std, 5.0)
            actual = np.exp(
                log_weight - np.log(scale) - 0.5 * np.log(2 * np.pi) - float(log_z)
            )
            actual_mass = np.sum((actual[1:] + actual[:-1]) * np.diff(x) / 2)
            self.assertAlmostEqual(actual_mass, 1.0, delta=6e-7)
            samples = truncated_sample(
                loc.expand_as(uniforms), std.expand_as(uniforms), 5.0, uniforms
            )
            self.assertTrue(bool(torch.isfinite(samples).all()))
            self.assertTrue(bool(((samples >= 0) & (samples <= 5)).all()))
            self.assertTrue(bool((samples[1:] >= samples[:-1]).all()))
            self.assertAlmostEqual(float(samples.mean()), numerical_mean, delta=6e-5)
            self.assertAlmostEqual(
                float(samples.var(unbiased=False)), numerical_var, delta=3e-4
            )

    def test_nll_derivatives_match_finite_differences(self):
        for location in (-20.0, -0.5, 2.0, 5.5, 25.0):
            value = torch.tensor(location, dtype=torch.float64, requires_grad=True)
            scale = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)

            def loss(loc, std):
                return (
                    0.5 * ((1.0 - loc) / std) ** 2
                    + std.log()
                    + truncation(loc, std, 5.0)[2]
                )

            result = loss(value, scale)
            result.backward()
            eps = 1e-5
            finite = (
                loss(value.detach() + eps, scale.detach())
                - loss(value.detach() - eps, scale.detach())
            ) / (2 * eps)
            self.assertAlmostEqual(float(value.grad), float(finite), delta=1e-6)
            self.assertTrue(torch.isfinite(scale.grad))

    def test_extreme_tail_variance_stays_positive(self):
        for distance in (100.0, 300.0, 1000.0, 3000.0, 10000.0):
            for location in (-distance, 5.0 + distance):
                loc, scale = (
                    torch.tensor(location, dtype=torch.float64),
                    torch.tensor(1.0, dtype=torch.float64),
                )
                mean, std = truncated_moments(loc, scale, 5.0)
                self.assertGreater(float(std), 0.0)
                self.assertAlmostEqual(float(std) * distance, 1.0, delta=0.0004)
                boundary_distance = mean if location < 0 else 5.0 - mean
                self.assertAlmostEqual(
                    float(boundary_distance) * distance, 1.0, delta=0.0003
                )


class MultiagentDensityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_parameter_budgets_shapes_and_independence(self):
        for kind, count in (("world", 357), ("counterfactual", 408)):
            model = TransitionDensity(
                DensityConfig(
                    kind=kind,
                    num_agents=3,
                    cost_center=120.0,
                    cost_scale=40.0,
                    delta_scale=20.0,
                )
            )
            self.assertEqual(model.parameter_count, count)
            self.assertLessEqual(count, 462)
            c = torch.tensor([[0, 4], [3, 18], [10, 20]])
            a = torch.tensor([[2, 9], [5, 16], [4, 14]])
            kwargs = (
                {}
                if kind == "world"
                else dict(
                    root_contexts=c, root_actions=a, root_costs=torch.full_like(c, 50.0)
                )
            )
            loss = model.negative_log_likelihood(
                c, a, torch.full_like(c, 60.0), **kwargs
            )
            loss[1].sum().backward()
            for parameter in model.parameters():
                self.assertTrue(
                    torch.equal(parameter.grad[0], torch.zeros_like(parameter.grad[0]))
                )
                self.assertTrue(
                    torch.equal(parameter.grad[2], torch.zeros_like(parameter.grad[2]))
                )
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "model.pt"
                torch.save(
                    density_checkpoint(
                        model, training={"offline_environment_interactions": 123}
                    ),
                    path,
                )
                frozen = load_model(path)
                self.assertEqual(frozen.num_agents, 3)
                self.assertEqual(frozen.n, 10)
                means, stds = frozen.distribution(c, a, **kwargs)
                samples = frozen.sample(c, a, np.random.default_rng(19), **kwargs)
                self.assertEqual(means.shape, (3, 2))
                self.assertEqual(stds.shape, (3, 2))
                self.assertEqual(samples.shape, (3, 2))
                self.assertTrue(
                    all(not p.requires_grad for p in frozen.density.parameters())
                )

    def test_initialization_prefix_and_validation(self):
        small = TransitionDensity(DensityConfig(num_agents=1, seed=12))
        large = TransitionDensity(DensityConfig(num_agents=3, seed=12))
        for key, parameter in small.params.items():
            self.assertTrue(torch.equal(parameter[0], large.params[key][0]))


class PairedOfflineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_dataset_sharing_split_and_exact_costs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.npz"
            probabilities = np.full((3, 5, 5), 0.2)
            np.savez(reference, probabilities=probabilities)
            kwargs = dict(
                seed=3,
                n=2,
                train_pairs=13,
                validation_pairs=7,
                test_pairs=9,
            )
            metadata = collect_dataset(root / "data", reference, **kwargs)
            repeated = collect_dataset(root / "repeat", reference, **kwargs)
            arrays, _ = load_dataset(root / "data")
            repeated_arrays, _ = load_dataset(root / "repeat")
            for key in arrays:
                np.testing.assert_array_equal(arrays[key], repeated_arrays[key])
            self.assertEqual(metadata["environment_interactions"], 29 * 4)
            self.assertEqual(repeated["environment_interactions"], 29 * 4)
            root_cost = ((arrays["contexts"] - arrays["root_actions"]) ** 2).sum(-1)
            np.testing.assert_array_equal(root_cost, arrays["root_costs"])
            for i in range(3):
                alternative = arrays["root_actions"].copy()
                alternative[:, i] = arrays["alternative_actions"][:, i]
                np.testing.assert_array_equal(
                    ((arrays["contexts"] - alternative) ** 2).sum(-1),
                    arrays["alternative_costs"][:, i],
                )
            group_sets = []
            for split in range(3):
                world = expand_pairs(arrays, split, "world")
                counter = expand_pairs(arrays, split, "counterfactual")
                self.assertEqual(
                    set(world), {"contexts", "actions", "costs", "root_group_ids"}
                )
                self.assertTrue(torch.equal(world["costs"], counter["costs"]))
                count = world["costs"].shape[-1] // 2
                self.assertTrue(
                    torch.equal(
                        counter["costs"][:, :count], counter["root_costs"][:, count:]
                    )
                )
                self.assertTrue(
                    torch.equal(
                        counter["actions"][:, count:],
                        counter["root_actions"][:, :count],
                    )
                )
                group_sets.append(set(world["root_group_ids"].tolist()))
            self.assertFalse(
                group_sets[0] & group_sets[1]
                or group_sets[0] & group_sets[2]
                or group_sets[1] & group_sets[2]
            )
            means, variances = true_marginal_moments(probabilities)
            self.assertTrue(np.allclose(means, 4.0))
            self.assertTrue(np.all(variances > 0))

    def test_short_training_checkpoint_metrics_and_frozen_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.npz"
            np.savez(reference, probabilities=np.full((2, 5, 5), 0.2))
            collect_dataset(
                root / "data",
                reference,
                seed=1,
                n=2,
                train_pairs=32,
                validation_pairs=16,
                test_pairs=16,
            )
            for kind in ("world", "counterfactual"):
                run = train_density(
                    root / "data",
                    root / kind,
                    kind=kind,
                    seed=1,
                    steps=3,
                    batch_size=8,
                    hidden=2,
                    log_every=2,
                )
                self.assertEqual(run["offline_environment_interactions"], 64 * 3)
                self.assertEqual(run["optimizer_steps"], 3)
                frozen = load_model(root / kind / "model.pt")
                self.assertEqual(frozen.metadata["training"]["optimizer_steps"], 3)


if __name__ == "__main__":
    unittest.main()
