import importlib
import unittest
from unittest.mock import patch

import torch

from FisherRaoIG.BinaryFisherRaoIntegratedGradients import (
    FisherRaoIntegratedGradients2Class,
)
from FisherRaoIG.FisherRaoIntegratedGradients import (
    FisherRaoIntegratedGradients,
)
from FisherRaoIG.FR_utils import compute_waypoints, spherical_loss


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


class BinaryFisherRaoTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyLinearClassifier().eval()
        self.inputs = torch.tensor(
            [
                [[[0.2, 0.7], [-0.1, 0.4]]],
                [[[0.8, -0.3], [0.5, 0.2]]],
            ],
            dtype=torch.float32,
        )
        self.targets = torch.tensor([0, 2])

    def make_method(self):
        def target_logit(inputs, targets):
            return self.model(inputs).gather(
                1, targets[:, None]
            ).squeeze(1)

        return FisherRaoIntegratedGradients2Class(
            model=self.model,
            model_forward=target_logit,
            target_idx=self.targets,
        )

    def test_binary_logodds_matches_collapsed_softmax(self):
        logits = self.model(self.inputs)
        probabilities = torch.softmax(logits, dim=1)
        p_t = probabilities.gather(1, self.targets[:, None]).squeeze(1)
        expected = torch.log(p_t / (1.0 - p_t))
        actual = self.make_method()._logodds_one_vs_rest_from_logits(
            logits, self.targets
        )
        torch.testing.assert_close(actual, expected)

    def test_slerp_has_correct_endpoints_and_normalization(self):
        p_t0 = torch.tensor([0.8, 0.55])
        waypoints, mask, n_steps = self.make_method()._binary_slerp_waypoints(
            p_t0=p_t0,
            num_classes=5,
            kl_target=0.01,
        )

        expected_start = torch.stack(
            (p_t0.sqrt(), (1.0 - p_t0).sqrt()), dim=1
        )
        torch.testing.assert_close(waypoints[:, 0], expected_start)
        for batch_index, count in enumerate(n_steps.tolist()):
            expected_end = torch.tensor(
                [1.0 / 5.0, 4.0 / 5.0]
            ).sqrt()
            torch.testing.assert_close(
                waypoints[batch_index, count - 1], expected_end
            )
            valid = waypoints[batch_index, mask[batch_index]]
            torch.testing.assert_close(
                valid.square().sum(dim=1),
                torch.ones(valid.shape[0]),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_closed_form_matches_dense_rank_one_solve(self):
        torch.manual_seed(4)
        u = torch.randn(3, 2, 2)
        fisher_scalar = torch.tensor([0.2, 0.1, 0.24])
        beta = torch.tensor([-0.3, 0.5, 0.12])
        lam = 0.07
        actual, denominator = self.make_method()._closed_form_direction(
            u, fisher_scalar, beta, lam
        )

        expected = []
        for index in range(u.shape[0]):
            u_flat = u[index].flatten()
            matrix = (
                fisher_scalar[index] * torch.outer(u_flat, u_flat)
                + lam * torch.eye(u_flat.numel())
            )
            expected.append(
                torch.linalg.solve(matrix, beta[index] * u_flat).view_as(u[index])
            )
            self.assertGreater(denominator[index].item(), lam)
        torch.testing.assert_close(actual, torch.stack(expected))

    def test_attribution_is_finite_complete_and_targets_next_waypoint(self):
        attrs, stats = self.make_method().attribute(
            self.inputs,
            kl_target=0.05,
            delta_euc=1.0,
            eta_max=1.0,
            lam=0.1,
            show_progress=False,
        )

        self.assertEqual(attrs.shape, self.inputs.shape)
        self.assertTrue(torch.isfinite(attrs).all())
        self.assertLess(max(stats["completeness_delta"]), 1e-5)
        self.assertEqual(len(stats["waypoint"]), stats["total_steps"])
        self.assertEqual(
            stats["total_steps"], max(stats["n_steps"]) - 1
        )

        with torch.no_grad():
            start_probability = torch.softmax(
                self.model(self.inputs), dim=1
            ).gather(1, self.targets[:, None]).squeeze(1)
            start_sqrt = torch.stack(
                (start_probability.sqrt(), (1.0 - start_probability).sqrt()),
                dim=1,
            )
            first_target = torch.tensor(stats["waypoint"][0])
        self.assertFalse(torch.allclose(first_target, start_sqrt))

        required_trace_keys = {
            "p_t",
            "target_logodds",
            "waypoint",
            "beta",
            "gradient_norm",
            "closed_form_denominator",
            "active_constraint",
        }
        self.assertTrue(required_trace_keys.issubset(stats))

    def test_binary_is_selectable_from_canonical_fringe(self):
        def target_logit(inputs, targets):
            return self.model(inputs).gather(
                1, targets[:, None]
            ).squeeze(1)

        method = FisherRaoIntegratedGradients(
            model=self.model,
            model_forward=target_logit,
            target_idx=self.targets,
            cg_max_iters=3,
        )
        attrs, delta = method.attribute(
            self.inputs,
            kl_target=0.05,
            fisher=True,
            binary=True,
            smoothing=False,
            delta_euc=1.0,
            eta_max=1.0,
            lambda_ratio=0.1,
        )

        self.assertEqual(attrs.shape, self.inputs.shape)
        self.assertTrue(torch.isfinite(attrs).all())
        self.assertLess(delta, 1e-5)
        self.assertFalse(method.last_binary_stats["smoothing"])

    def test_categorical_fringe_targets_the_next_waypoint(self):
        def target_logit(inputs, targets):
            return self.model(inputs).gather(
                1, targets[:, None]
            ).squeeze(1)

        method = FisherRaoIntegratedGradients(
            model=self.model,
            model_forward=target_logit,
            target_idx=self.targets,
            cg_max_iters=3,
        )
        with torch.no_grad():
            p_start = torch.softmax(self.model(self.inputs), dim=1)
            expected_waypoints, _, _ = compute_waypoints(
                p_start=p_start,
                kl_target=0.05,
                num_classes=p_start.shape[1],
            )

        observed_targets = []

        def record_spherical_loss(current_sqrt, target_sqrt):
            observed_targets.append(target_sqrt.detach().clone())
            return spherical_loss(current_sqrt, target_sqrt)

        fringe_module = importlib.import_module(
            "FisherRaoIG.FisherRaoIntegratedGradients"
        )
        with patch.object(
            fringe_module,
            "spherical_loss",
            side_effect=record_spherical_loss,
        ):
            attrs, _ = method.attribute(
                self.inputs,
                kl_target=0.05,
                fisher=False,
                delta_euc=1.0,
                eta_max=0.1,
            )

        self.assertTrue(torch.isfinite(attrs).all())
        self.assertGreater(len(observed_targets), 0)
        torch.testing.assert_close(observed_targets[0], expected_waypoints[:, 1])
        self.assertFalse(
            torch.allclose(observed_targets[0], expected_waypoints[:, 0])
        )

if __name__ == "__main__":
    unittest.main()
