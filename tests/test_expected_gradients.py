import unittest
from pathlib import Path

import torch
from captum.attr import IntegratedGradients

from methods.ExpectedGradients.ExpectedGradients_benchmarking import (
    PROJECT_ROOT,
    ExpectedGradients,
    _resolve_project_path,
)


class TinyLinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [0.5, -0.2, 0.3, 0.1],
                        [-0.4, 0.6, 0.2, -0.1],
                        [0.2, 0.1, -0.5, 0.7],
                    ]
                )
            )
            self.linear.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

    def forward(self, inputs):
        return self.linear(inputs.reshape(inputs.shape[0], -1))


class ExpectedGradientsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = TinyLinearClassifier().eval()
        self.inputs = torch.tensor(
            [
                [[[0.2, 0.7], [-0.1, 0.4]]],
                [[[0.8, -0.3], [0.5, 0.2]]],
            ],
            dtype=torch.float32,
        )
        self.targets = torch.tensor([0, 2])
        self.baseline_pool = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.1, -0.2], [0.3, 0.4]]],
                [[[-0.4, 0.5], [0.2, -0.1]]],
            ],
            dtype=torch.float32,
        )

    def make_method(self, **overrides):
        params = {
            "model": self.model,
            "baseline_pool": self.baseline_pool,
            "num_baselines": 3,
            "steps_per_baseline": 8,
            "method": "gausslegendre",
            "internal_batch_size": 16,
            "baseline_batch_size": 2,
            "sample_with_replacement": False,
        }
        params.update(overrides)
        return ExpectedGradients(**params)

    def test_shape_finiteness_and_metadata(self):
        method = self.make_method()
        attributions, metadata = method.attribute(
            self.inputs, self.targets, seed=17
        )

        self.assertEqual(attributions.shape, self.inputs.shape)
        self.assertTrue(torch.isfinite(attributions).all())
        self.assertEqual(metadata["num_baselines"], 3)
        self.assertEqual(metadata["steps_per_baseline"], 8)
        self.assertEqual(metadata["total_integration_steps"], 24)
        self.assertEqual(len(metadata["baseline_indices"]), 3)

    def test_fixed_seed_is_deterministic(self):
        method = self.make_method(num_baselines=2)
        first, first_meta = method.attribute(
            self.inputs, self.targets, seed=123
        )
        second, second_meta = method.attribute(
            self.inputs, self.targets, seed=123
        )

        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(
            first_meta["baseline_indices"],
            second_meta["baseline_indices"],
        )

    def test_one_reference_matches_integrated_gradients(self):
        one_reference = self.baseline_pool[1:2]
        method = self.make_method(
            baseline_pool=one_reference,
            num_baselines=1,
            baseline_batch_size=1,
        )
        expected_gradients, _ = method.attribute(
            self.inputs, self.targets, seed=5
        )

        references = one_reference.expand_as(self.inputs)
        ordinary_ig = IntegratedGradients(self.model).attribute(
            self.inputs,
            baselines=references,
            target=self.targets,
            n_steps=8,
            method="gausslegendre",
            internal_batch_size=16,
        ).to(expected_gradients.dtype)
        torch.testing.assert_close(
            expected_gradients,
            ordinary_ig,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_batching_does_not_change_each_sample(self):
        method = self.make_method(num_baselines=2)
        batched, _ = method.attribute(
            self.inputs, self.targets, seed=99
        )

        separate = []
        for index in range(self.inputs.shape[0]):
            attr, _ = method.attribute(
                self.inputs[index : index + 1],
                self.targets[index : index + 1],
                seed=99,
            )
            separate.append(attr)
        separate = torch.cat(separate, dim=0)

        torch.testing.assert_close(
            batched,
            separate,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_linear_model_matches_analytic_expectation(self):
        method = self.make_method(
            num_baselines=3,
            baseline_batch_size=1,
        )
        attributions, _ = method.attribute(
            self.inputs,
            self.targets,
            seed=31,
        )

        mean_reference = self.baseline_pool.mean(dim=0)
        target_weights = self.model.linear.weight[self.targets].reshape_as(
            self.inputs
        )
        expected = (self.inputs - mean_reference) * target_weights

        torch.testing.assert_close(
            attributions,
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_invalid_reference_count_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_method(
                num_baselines=self.baseline_pool.shape[0] + 1,
                sample_with_replacement=False,
            )

    def test_relative_paths_are_resolved_from_project_root(self):
        resolved = Path(_resolve_project_path("./data/ImageNet"))
        self.assertEqual(
            resolved,
            (PROJECT_ROOT / "data/ImageNet").resolve(),
        )

    def test_absolute_paths_are_preserved(self):
        absolute = (PROJECT_ROOT / "data/ImageNet").resolve()
        self.assertEqual(
            Path(_resolve_project_path(absolute)),
            absolute,
        )


if __name__ == "__main__":
    unittest.main()
