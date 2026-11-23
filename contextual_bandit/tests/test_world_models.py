"""Meaningful checks of paired data, likelihoods, and frozen model interfaces."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ..offline import (
    collect_dataset,
    evaluate_density,
    expand_pairs,
    load_dataset,
    train_density,
)
from ..world_models import (
    DensityConfig,
    TransitionDensity,
    density_checkpoint,
    load_model,
)


class WorldModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def reference(self, path):
        probabilities = 0.1 / 21 + 0.9 * np.eye(21)
        np.savez_compressed(path, probabilities=probabilities)
        return path

    def test_pair_sharing_and_split_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref = self.reference(root / "reference.npz")
            metadata = collect_dataset(
                root / "data",
                ref,
                seed=17,
                noise_std=30,
                train_pairs=30,
                validation_pairs=11,
                test_pairs=13,
            )
            arrays, loaded = load_dataset(root / "data")
            np.testing.assert_allclose(
                arrays["next_states"][:, 0] - arrays["actions"][:, 0],
                arrays["next_states"][:, 1] - arrays["actions"][:, 1],
                atol=8e-6,
            )
            self.assertEqual(metadata["environment_interactions"], 108)
            self.assertNotIn("noise", arrays)
            seen = []
            for split in range(3):
                wm = expand_pairs(arrays, split, "world")
                cwm = expand_pairs(arrays, split, "counterfactual")
                self.assertEqual(
                    len(wm["contexts"]), 2 * np.sum(arrays["split"] == split)
                )
                self.assertTrue(torch.equal(wm["next_states"], cwm["next_states"]))
                half = len(cwm["actions"]) // 2
                self.assertTrue(
                    torch.equal(cwm["root_actions"][:half], cwm["actions"][half:])
                )
                self.assertTrue(
                    torch.equal(cwm["root_next_states"][:half], cwm["next_states"][half:])
                )
                self.assertTrue(
                    torch.equal(cwm["root_actions"][half:], cwm["actions"][:half])
                )
                seen.append(set(wm["pair_ids"].tolist()))
            self.assertFalse(
                seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2]
            )

    def test_fixed_dataset_reproducibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref = self.reference(root / "reference.npz")
            datasets = []
            for name in ("one", "two"):
                collect_dataset(
                    root / name,
                    ref,
                    seed=3,
                    train_pairs=10,
                    validation_pairs=3,
                    test_pairs=4,
                )
                datasets.append(load_dataset(root / name)[0])
            for key in datasets[0]:
                np.testing.assert_array_equal(datasets[0][key], datasets[1][key])

    def test_small_models_do_not_start_at_true_equations(self):
        c, a = torch.tensor([2, 14]), torch.tensor([0, 20])
        wm = TransitionDensity(
            DensityConfig(kind="world", next_state_center=3.0, next_state_scale=7.0)
        )
        cwm = TransitionDensity(DensityConfig(kind="counterfactual", next_state_scale=7.0))
        self.assertEqual(wm.parameter_count, 117)
        self.assertEqual(cwm.parameter_count, 144)
        self.assertLess(cwm.parameter_count, 462)
        mean, _ = wm(c, a)
        torch.testing.assert_close(mean, torch.full((2,), 3.0))
        roots = dict(
            root_contexts=c,
            root_actions=torch.tensor([10, 10]),
            root_next_states=torch.tensor([31.0, -23.0]),
        )
        mean, _ = cwm(c, a, **roots)
        torch.testing.assert_close(mean, roots["root_next_states"])
        self.assertFalse(
            torch.equal(mean, roots["root_next_states"] + a - roots["root_actions"])
        )

    def test_normal_likelihood_and_positive_physical_floor(self):
        model = TransitionDensity(DensityConfig(next_state_scale=30.0, min_scale=0.05))
        c = torch.tensor([0, 10, 20])
        y = torch.tensor([-2.0, 3.0, 0.0])
        mean, scale = model(c, c)
        reference = -torch.distributions.Normal(mean, scale).log_prob(y)
        torch.testing.assert_close(model.negative_log_likelihood(c, c, y), reference)
        with torch.no_grad():
            model.net[-1].bias[1] = -100.0
        _, scale = model(c, c)
        torch.testing.assert_close(scale, torch.full((3,), 0.05))
        loss = model.negative_log_likelihood(c, c, y).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )

    def test_frozen_loader_sampler_and_distribution_adapter(self):
        with tempfile.TemporaryDirectory() as temp:
            model = TransitionDensity(
                DensityConfig(kind="counterfactual", next_state_scale=4.0)
            )
            checkpoint = Path(temp) / "model.pt"
            torch.save(density_checkpoint(model), checkpoint)
            frozen = load_model(checkpoint)
            self.assertEqual(frozen.kind, "cwm")
            self.assertFalse(frozen.density.training)
            self.assertTrue(
                all(not p.requires_grad for p in frozen.density.parameters())
            )
            c = np.zeros(4096, dtype=np.int64)
            kwargs = dict(
                root_contexts=c, root_actions=c, root_next_states=np.ones(4096) * 9.0
            )
            mean, scale = frozen.distribution(c, c, **kwargs)
            first = frozen.sample(c, c, np.random.default_rng(4), **kwargs)
            second = frozen.sample(c, c, np.random.default_rng(4), **kwargs)
            np.testing.assert_array_equal(first, second)
            self.assertLess(abs(first.mean() - mean.mean()), 0.15)
            self.assertLess(abs(first.std() - scale.mean()), 0.15)

    def test_quality_metrics_use_correct_conditional_truth(self):
        model = TransitionDensity(
            DensityConfig(kind="counterfactual", next_state_scale=1.0)
        )
        examples = {
            "contexts": torch.tensor([10, 10]),
            "actions": torch.tensor([11, 12]),
            "root_contexts": torch.tensor([10, 10]),
            "root_actions": torch.tensor([10, 10]),
            "root_next_states": torch.tensor([3.0, 4.0]),
            "next_states": torch.tensor([4.0, 6.0]),
        }
        values = evaluate_density(model, examples)
        self.assertAlmostEqual(values["conditional_mean_rmse"], np.sqrt(2.5), places=6)

    def test_short_mle_training_final_checkpoint_and_common_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref = self.reference(root / "reference.npz")
            collect_dataset(
                root / "data",
                ref,
                seed=1,
                noise_std=10.0,
                train_pairs=128,
                validation_pairs=32,
                test_pairs=32,
            )
            for kind in ("world", "counterfactual"):
                train_density(
                    root / "data",
                    root / kind,
                    kind=kind,
                    seed=1,
                    steps=100,
                    batch_size=64,
                    log_every=100,
                )
                self.assertTrue((root / kind / "learning_curve.csv").exists())
                checkpoint = load_model(root / kind / "model.pt")
                self.assertEqual(
                    checkpoint.metadata["training"]["optimizer_steps"], 100
                )
            with (root / "counterfactual" / "learning_curve.csv").open() as handle:
                curve = list(csv.DictReader(handle))
            self.assertLess(
                float(curve[-1]["conditional_mean_rmse"]),
                float(curve[0]["conditional_mean_rmse"]),
            )


if __name__ == "__main__":
    unittest.main()
