import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmarks.equal_budget.core import (
    BinaryFringeAttributor,
    CandidateSpec,
    IGAttributor,
    SmoothGradAttributor,
    build_disjoint_split,
    candidate_summary_from_results,
    closest_budget_candidates,
    closest_runtime_candidates,
    paired_bootstrap_mean,
)
from benchmarks.equal_budget.analyse_equal_budget import validate_comparison


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
            self.linear.bias.zero_()

    def forward(self, inputs):
        return self.linear(inputs.reshape(inputs.shape[0], -1))


class EqualBudgetTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyLinearClassifier().eval()
        self.inputs = torch.tensor(
            [[[[0.2, 0.7], [-0.1, 0.4]]]], dtype=torch.float32
        )
        self.targets = torch.tensor([0])

    def test_split_is_deterministic_and_disjoint(self):
        paths = [f"/tmp/equal-budget-image-{index}.jpg" for index in range(30)]
        first = build_disjoint_split(paths, 8, 4, 10, seed=19)
        second = build_disjoint_split(paths, 8, 4, 10, seed=19)
        self.assertEqual(first, second)
        baseline = set(first["baseline_paths"])
        warmup = set(first["warmup_paths"])
        evaluation = set(first["evaluation_paths"])
        self.assertFalse(baseline & warmup)
        self.assertFalse(baseline & evaluation)
        self.assertFalse(warmup & evaluation)
        self.assertEqual(first["overlap_count"], 0)

    def test_split_rejects_oversubscription(self):
        with self.assertRaises(ValueError):
            build_disjoint_split(
                [f"/tmp/image-{index}.jpg" for index in range(5)],
                baseline_pool_size=3,
                warmup_size=2,
                evaluation_size=1,
                seed=1,
            )

    def test_ig_completeness_for_linear_model(self):
        spec = CandidateSpec(
            "ig",
            "ig_test",
            {"n_steps": 8, "integration_method": "gausslegendre"},
        )
        method = IGAttributor(self.model, spec)
        attrs, metadata = method.attribute(self.inputs, self.targets, seed=1)
        absolute, relative = method.completeness(
            self.inputs, self.targets, attrs, metadata
        )
        self.assertLess(absolute, 1e-6)
        self.assertLess(relative, 1e-5)
        self.assertEqual(
            method.operation_budget(metadata)["gradient_evaluations"], 8
        )
        self.assertEqual(
            method.operation_budget(metadata)[
                "autograd_equivalent_evaluations"
            ],
            8,
        )

    def test_chunked_smoothgrad_is_chunk_size_invariant(self):
        common = {
            "n_samples": 6,
            "stdev_spread": 0.1,
            "magnitude": False,
        }
        small = SmoothGradAttributor(
            self.model,
            CandidateSpec(
                "smoothgrad",
                "sg_small",
                {**common, "sample_batch_size": 2},
            ),
        )
        large = SmoothGradAttributor(
            self.model,
            CandidateSpec(
                "smoothgrad",
                "sg_large",
                {**common, "sample_batch_size": 6},
            ),
        )
        attrs_small, _ = small.attribute(
            self.inputs, self.targets, seed=123
        )
        attrs_large, _ = large.attribute(
            self.inputs, self.targets, seed=123
        )
        torch.testing.assert_close(attrs_small, attrs_large)

    def test_binary_fringe_uses_smoothed_target_vs_rest_geometry(self):
        method = BinaryFringeAttributor(
            self.model,
            CandidateSpec(
                "fringe_binary",
                "binary_test",
                {
                    "kl_target": 0.05,
                    "delta_euc": 1.0,
                    "eta_max": 1.0,
                    "lambda_ratio": 0.1,
                    "smoothing": True,
                    # The tiny 2x2 fixture is too small for the production 5x5
                    # Gaussian preconditioner; gamma regularization remains on.
                    "use_sobolev_preconditioner": False,
                    "gamma_step": 0.01,
                    "gamma_prior": 0.001,
                    "cg_max_iters": 3,
                    "show_progress": False,
                },
            ),
        )
        attrs, metadata = method.attribute(self.inputs, self.targets, seed=1)

        self.assertTrue(torch.isfinite(attrs).all())
        self.assertEqual(metadata["geometry"], "binary_target_vs_rest")
        self.assertTrue(metadata["smoothing"])
        self.assertTrue(metadata["regularizer_inverse_iters"])
        budget = method.operation_budget(metadata)
        self.assertEqual(budget["fisher_matvec_evaluations"], 0.0)
        self.assertGreater(budget["regularizer_inverse_iterations"], 0.0)

    def test_runtime_matching_uses_measured_log_distance(self):
        summary = pd.DataFrame(
            [
                ["m", "ig", "ig_32", 10.0],
                ["m", "ig", "ig_64", 20.0],
                ["m", "fringe", "fr_fast", 18.0],
                ["m", "fringe", "fr_slow", 55.0],
            ],
            columns=[
                "model",
                "method",
                "candidate_id",
                "median_attribution_time_ms",
            ],
        )
        selected = closest_runtime_candidates(
            summary, "ig", "ig_32", [2, 5]
        )
        fringe = selected[selected["method"] == "fringe"].sort_values(
            "budget_multiple"
        )
        self.assertEqual(
            fringe["candidate_id"].tolist(), ["fr_fast", "fr_slow"]
        )

    def test_generic_budget_matching_skips_unavailable_models(self):
        summary = pd.DataFrame(
            [
                ["gpu", "ig", "ig_32", 100.0],
                ["gpu", "fringe", "fr", 220.0],
                ["cpu", "ig", "ig_32", np.nan],
                ["cpu", "fringe", "fr", np.nan],
            ],
            columns=["model", "method", "candidate_id", "peak"],
        )
        selected = closest_budget_candidates(
            summary,
            "ig",
            "ig_32",
            [2],
            value_column="peak",
            budget_name="memory",
        )
        self.assertEqual(set(selected["model"]), {"gpu"})
        fringe = selected[selected["method"] == "fringe"].iloc[0]
        self.assertEqual(fringe["candidate_id"], "fr")

    def test_paired_bootstrap_is_deterministic(self):
        differences = [1.0, 2.0, 3.0, 4.0]
        first = paired_bootstrap_mean(
            differences, seed=7, n_resamples=1000
        )
        second = paired_bootstrap_mean(
            differences, seed=7, n_resamples=1000
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], 2.5)
        self.assertLessEqual(first["ci_low"], first["mean"])
        self.assertGreaterEqual(first["ci_high"], first["mean"])

    def test_candidate_id_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            CandidateSpec("ig", "../escape", {"n_steps": 8})

    def test_candidate_index_is_rebuilt_from_completed_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "resnet18" / "ig" / "ig_8"
            output.mkdir(parents=True)
            (output / "candidate_config.json").write_text(
                json.dumps(
                    {
                        "model": "resnet18",
                        "method": "ig",
                        "candidate_id": "ig_8",
                        "is_full": False,
                    }
                )
            )
            (output / "systems.json").write_text(
                json.dumps(
                    {
                        "initialization_time_ms": 3.0,
                        "batched_throughput": {"images_per_second": 12.0},
                    }
                )
            )
            pd.DataFrame(
                {
                    "attribution_time_ms": [10.0, 14.0],
                    "completeness_delta": [0.1, 0.3],
                }
            ).to_parquet(output / "per_image.parquet", index=False)

            summary = candidate_summary_from_results(directory)
            self.assertEqual(len(summary), 1)
            row = summary.iloc[0]
            self.assertEqual(row["candidate_id"], "ig_8")
            self.assertEqual(row["median_attribution_time_ms"], 12.0)
            self.assertEqual(row["mean_completeness_delta"], 0.2)
            self.assertEqual(row["throughput_images_per_second"], 12.0)

    def test_comparison_rejects_mixed_gpu_models(self):
        summary = pd.DataFrame(
            [
                ["m", "ig", "ig_32", "split", "GPU A"],
                ["m", "fringe", "fringe_default", "split", "GPU B"],
            ],
            columns=[
                "model",
                "method",
                "candidate_id",
                "evaluation_sha256",
                "gpu_name",
            ],
        )
        with self.assertRaisesRegex(ValueError, "Mixed GPU models"):
            validate_comparison(summary)

    def test_comparison_rejects_mixed_evaluation_splits(self):
        summary = pd.DataFrame(
            [
                ["m", "ig", "ig_32", "split-a", "GPU A"],
                ["m", "fringe", "fringe_default", "split-b", "GPU A"],
            ],
            columns=[
                "model",
                "method",
                "candidate_id",
                "evaluation_sha256",
                "gpu_name",
            ],
        )
        with self.assertRaisesRegex(ValueError, "different evaluation splits"):
            validate_comparison(summary)


if __name__ == "__main__":
    unittest.main()
