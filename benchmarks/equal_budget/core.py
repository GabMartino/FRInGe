from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from captum.attr import IntegratedGradients

from methods.ExpectedGradients.ExpectedGradients_benchmarking import ExpectedGradients
from methods.FisherRaoIG.FisherRaoIntegratedGradientsLogs import (
    FisherRaoIntegratedGradients,
)
from methods.FisherRaoIG.BinaryFisherRaoIntegratedGradients import (
    BinaryFisherRaoIntegratedGradients,
)
from evaluation.metrics.Infidelity import InfidelityScorer
from evaluation.metrics.InsDelAUC import CausalMetricScorer
from evaluation.metrics.MAS import BatchMASMetric
from evaluation.metrics.Sparseness import SparsenessScorer
from common.utils import load_image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
METHOD_DISPLAY_NAMES = {
    "ig": "Integrated Gradients",
    "fringe": "FRInGe — full Fisher",
    "fringe_binary": "FRInGe — binary target-vs-rest",
    "expected_gradients": "Expected Gradients",
    "smoothgrad": "SmoothGrad",
    "ig_smoothed_map": "IG — smoothed attribution map",
}


def method_display_name(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(str(method), str(method))


def candidate_display_name(
    method: str,
    candidate_id: str,
    params: Mapping[str, Any],
) -> str:
    """Return a human-facing label that exposes the actual compute controls."""
    method = str(method)
    if method in {"ig", "ig_smoothed_map"} and "n_steps" in params:
        return f"steps={int(params['n_steps'])}"
    if method == "expected_gradients":
        references = int(params.get("num_baselines", 0))
        steps = int(params.get("steps_per_baseline", 0))
        return f"references={references} × steps={steps}"
    if method == "smoothgrad" and "n_samples" in params:
        return f"samples={int(params['n_samples'])}"
    if method in {"fringe", "fringe_binary"}:
        kl_target = float(params["kl_target"])
        iterations = int(params.get("cg_max_iters", 20))
        solver = "CG" if method == "fringe" else "inverse"
        return f"KL={kl_target:.4g} · {solver}≤{iterations}"
    return str(candidate_id)


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def stable_path_hash(paths: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(Path(p).resolve()) for p in paths))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def list_image_files(directory: str | Path) -> List[str]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    paths = sorted(
        str(path.resolve())
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No image files found under: {root}")
    return paths


def build_disjoint_split(
    paths: Sequence[str],
    baseline_pool_size: int,
    warmup_size: int,
    evaluation_size: Optional[int],
    seed: int,
) -> Dict[str, Any]:
    """Create a deterministic, mutually disjoint reference/warmup/eval split."""
    unique = sorted({str(Path(path).resolve()) for path in paths})
    if len(unique) != len(paths):
        raise ValueError("Dataset path list contains duplicates.")
    if baseline_pool_size < 1:
        raise ValueError("baseline_pool_size must be >= 1.")
    if warmup_size < 1:
        raise ValueError("warmup_size must be >= 1.")

    if evaluation_size is None or int(evaluation_size) <= 0:
        evaluation_size = len(unique) - baseline_pool_size - warmup_size
    evaluation_size = int(evaluation_size)
    required = baseline_pool_size + warmup_size + evaluation_size
    if required > len(unique):
        raise ValueError(
            f"Split requires {required} images but dataset has {len(unique)}."
        )

    shuffled = unique.copy()
    random.Random(int(seed)).shuffle(shuffled)
    baseline = sorted(shuffled[:baseline_pool_size])
    warmup = sorted(
        shuffled[baseline_pool_size : baseline_pool_size + warmup_size]
    )
    evaluation = sorted(shuffled[baseline_pool_size + warmup_size : required])

    sets = [set(baseline), set(warmup), set(evaluation)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Internal error: generated split is not disjoint.")

    return {
        "seed": int(seed),
        "dataset_size": len(unique),
        "baseline_pool_size": len(baseline),
        "warmup_size": len(warmup),
        "evaluation_size": len(evaluation),
        "baseline_pool_sha256": stable_path_hash(baseline),
        "warmup_sha256": stable_path_hash(warmup),
        "evaluation_sha256": stable_path_hash(evaluation),
        "baseline_paths": baseline,
        "warmup_paths": warmup,
        "evaluation_paths": evaluation,
        "overlap_count": 0,
    }


@dataclass(frozen=True)
class CandidateSpec:
    method: str
    candidate_id: str
    params: Dict[str, Any]
    is_full: bool = False

    def __post_init__(self) -> None:
        if not self.method or not self.candidate_id:
            raise ValueError("method and candidate_id must be non-empty.")
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        if any(ch not in allowed for ch in self.candidate_id):
            raise ValueError(
                f"Unsafe candidate id {self.candidate_id!r}; use letters, "
                "numbers, '-', '_' or '.'."
            )


class AttributionCandidate:
    def __init__(self, model: torch.nn.Module, spec: CandidateSpec):
        self.model = model.eval()
        self.spec = spec
        self.device = next(model.parameters()).device

    def attribute(
        self, inputs: torch.Tensor, targets: torch.Tensor, seed: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def completeness(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        attributions: torch.Tensor,
        metadata: Mapping[str, Any],
    ) -> Tuple[float, float]:
        return float("nan"), float("nan")

    def operation_budget(self, metadata: Mapping[str, Any]) -> Dict[str, float]:
        return {}


def _target_logits(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return model(inputs).gather(1, targets[:, None]).squeeze(1)


def _completeness_errors(
    attributions: torch.Tensor,
    score_input: torch.Tensor,
    score_reference: torch.Tensor,
) -> Tuple[float, float]:
    attr_sum = attributions.reshape(attributions.shape[0], -1).sum(dim=1)
    score_delta = score_input - score_reference
    absolute = (attr_sum - score_delta).abs()
    relative = absolute / score_delta.abs().clamp_min(1e-9)
    return float(absolute.mean().item()), float(relative.mean().item())


class IGAttributor(AttributionCandidate):
    def __init__(self, model: torch.nn.Module, spec: CandidateSpec):
        super().__init__(model, spec)
        self.n_steps = int(spec.params["n_steps"])
        self.method = str(spec.params.get("integration_method", "gausslegendre"))
        self.internal_batch_size = spec.params.get("internal_batch_size")
        self.ig = IntegratedGradients(self.model)

    def attribute(self, inputs, targets, seed):
        baseline = torch.zeros_like(inputs)
        attrs = self.ig.attribute(
            inputs,
            baselines=baseline,
            target=targets,
            n_steps=self.n_steps,
            method=self.method,
            internal_batch_size=self.internal_batch_size,
        )
        return attrs.detach(), {"n_steps": self.n_steps}

    def completeness(self, inputs, targets, attributions, metadata):
        with torch.no_grad():
            score_input = _target_logits(self.model, inputs, targets)
            score_reference = _target_logits(
                self.model, torch.zeros_like(inputs), targets
            )
        return _completeness_errors(
            attributions, score_input, score_reference
        )

    def operation_budget(self, metadata):
        return {
            "gradient_evaluations": float(self.n_steps),
            "autograd_equivalent_evaluations": float(self.n_steps),
        }


class SmoothedIGAttributor(IGAttributor):
    """IG followed by a spatial Gaussian filter on the final attribution map."""

    def __init__(self, model: torch.nn.Module, spec: CandidateSpec):
        super().__init__(model, spec)
        kernel = spec.params.get("kernel_size", [5, 5])
        sigma = spec.params.get("sigma", [2.0, 2.0])
        self.kernel_size = [int(v) for v in kernel]
        self.sigma = [float(v) for v in sigma]

    def attribute(self, inputs, targets, seed):
        attrs, metadata = super().attribute(inputs, targets, seed)
        attrs = TF.gaussian_blur(
            attrs, kernel_size=self.kernel_size, sigma=self.sigma
        )
        metadata.update(
            {
                "smoothing_control": "final_attribution_gaussian_blur",
                "kernel_size": self.kernel_size,
                "sigma": self.sigma,
            }
        )
        return attrs.detach(), metadata


class ExpectedGradientsAttributor(AttributionCandidate):
    def __init__(
        self,
        model: torch.nn.Module,
        spec: CandidateSpec,
        baseline_pool: torch.Tensor,
    ):
        super().__init__(model, spec)
        self.baseline_pool = baseline_pool.detach().cpu()
        self.impl = ExpectedGradients(
            model=model,
            baseline_pool=self.baseline_pool,
            num_baselines=int(spec.params["num_baselines"]),
            steps_per_baseline=int(spec.params["steps_per_baseline"]),
            method=str(spec.params.get("integration_method", "gausslegendre")),
            internal_batch_size=spec.params.get("internal_batch_size"),
            baseline_batch_size=spec.params.get("baseline_batch_size"),
            sample_with_replacement=bool(
                spec.params.get("sample_with_replacement", False)
            ),
        )

    def attribute(self, inputs, targets, seed):
        return self.impl.attribute(inputs, targets, seed=seed)

    def completeness(self, inputs, targets, attributions, metadata):
        indices = torch.as_tensor(
            metadata["baseline_indices"], dtype=torch.long
        )
        refs = self.baseline_pool.index_select(0, indices).to(
            self.device, dtype=inputs.dtype
        )
        expanded_targets = targets[:1].expand(refs.shape[0])
        with torch.no_grad():
            score_input = _target_logits(self.model, inputs, targets)
            score_reference = _target_logits(
                self.model, refs, expanded_targets
            ).mean().view(1)
        return _completeness_errors(
            attributions, score_input, score_reference
        )

    def operation_budget(self, metadata):
        return {
            "gradient_evaluations": float(
                metadata["total_integration_steps"]
            ),
            "autograd_equivalent_evaluations": float(
                metadata["total_integration_steps"]
            ),
            "reference_samples": float(metadata["num_baselines"]),
        }


class SmoothGradAttributor(AttributionCandidate):
    def __init__(self, model: torch.nn.Module, spec: CandidateSpec):
        super().__init__(model, spec)
        self.n_samples = int(spec.params["n_samples"])
        self.stdev_spread = float(spec.params.get("stdev_spread", 0.15))
        self.magnitude = bool(spec.params.get("magnitude", False))
        self.sample_batch_size = int(
            spec.params.get("sample_batch_size", self.n_samples)
        )
        if self.sample_batch_size < 1:
            raise ValueError("sample_batch_size must be >= 1.")

    def attribute(self, inputs, targets, seed):
        batch_size, channels, height, width = inputs.shape
        x_max = inputs.amax(dim=(1, 2, 3), keepdim=True)
        x_min = inputs.amin(dim=(1, 2, 3), keepdim=True)
        noise_std = self.stdev_spread * (x_max - x_min).clamp_min(1e-6)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        if self.magnitude:
            accumulator = torch.zeros_like(inputs)
        else:
            accumulator = torch.zeros_like(inputs)

        completed = 0
        while completed < self.n_samples:
            count = min(self.sample_batch_size, self.n_samples - completed)
            expanded = (
                inputs[:, None]
                .expand(batch_size, count, channels, height, width)
                .reshape(batch_size * count, channels, height, width)
                .detach()
                .requires_grad_(True)
            )
            expanded_targets = (
                targets[:, None].expand(batch_size, count).reshape(-1)
            )
            expanded_std = (
                noise_std[:, None]
                .expand(batch_size, count, 1, 1, 1)
                .reshape(batch_size * count, 1, 1, 1)
            )
            noise = torch.randn(
                expanded.shape,
                generator=generator,
                device=self.device,
                dtype=expanded.dtype,
            )
            noisy = expanded + noise * expanded_std
            scores = _target_logits(self.model, noisy, expanded_targets)
            grads = torch.autograd.grad(scores.sum(), expanded)[0].reshape(
                batch_size, count, channels, height, width
            )
            if self.magnitude:
                accumulator += (grads ** 2).sum(dim=1)
            else:
                accumulator += grads.sum(dim=1)
            completed += count

        if self.magnitude:
            attrs = torch.sqrt(accumulator / self.n_samples + 1e-12)
        else:
            attrs = accumulator / self.n_samples
        return attrs.detach(), {
            "n_samples": self.n_samples,
            "sample_batch_size": self.sample_batch_size,
        }

    def operation_budget(self, metadata):
        return {
            "gradient_evaluations": float(self.n_samples),
            "autograd_equivalent_evaluations": float(self.n_samples),
            "noise_samples": float(self.n_samples),
        }


class FringeAttributor(AttributionCandidate):
    """Standard multiclass FRInGe."""

    def attribute(self, inputs, targets, seed):
        params = dict(self.spec.params)

        def model_forward(x, target):
            return _target_logits(self.model, x, target)

        impl = FisherRaoIntegratedGradients(
            model=self.model,
            model_forward=model_forward,
            target_idx=targets,
            cg_max_iters=int(params.pop("cg_max_iters", 20)),
            cg_rtol=float(params.pop("cg_rtol", 1e-3)),
            cg_atol=float(params.pop("cg_atol", 1e-5)),
        )
        attrs, stats = impl.attribute(
            x=inputs,
            kl_target=float(params["kl_target"]),
            fisher=bool(params.get("fisher", True)),
            delta_euc=float(params.get("delta_euc", 1000.0)),
            eta_max=float(params.get("eta_max", 1000.0)),
            use_sobolev_preconditioner=bool(
                params.get("use_sobolev_preconditioner", True)
            ),
            lambda_ratio=float(params.get("lambda_ratio", 1.0)),
            smoothing=bool(params.get("smoothing", True)),
            gamma_step=float(params.get("gamma_step", 0.01)),
            gamma_prior=float(params.get("gamma_prior", 0.001)),
        )
        stats["geometry"] = "full_multiclass"
        return attrs.detach(), stats

    def completeness(self, inputs, targets, attributions, metadata):
        absolute = np.nanmean(
            np.asarray(metadata.get("completeness_delta", [np.nan]), dtype=float)
        )
        relative = np.nanmean(
            np.asarray(
                metadata.get("completeness_relative_error", [np.nan]),
                dtype=float,
            )
        )
        return float(absolute), float(relative)

    def operation_budget(self, metadata):
        cg = np.asarray(metadata.get("cg_iters", []), dtype=float)
        waypoints = float(metadata.get("total_steps", np.nan))
        cg_iterations = float(np.nansum(cg)) if cg.size else 0.0
        lambda_ratio = float(self.spec.params.get("lambda_ratio", 1.0))
        warm_start_matvecs = (
            max(0.0, waypoints - 1.0) if lambda_ratio != 0 else 0.0
        )
        # PCG evaluates A once per iteration and once for its final residual;
        # warm starts add an initial A(x0), and telemetry adds Gv once per
        # waypoint. Each Fisher matvec consists of one JVP and one VJP.
        fisher_matvecs = (
            cg_iterations + 2.0 * waypoints + warm_start_matvecs
        )
        reverse_gradients = 2.0 * waypoints + 1.0
        return {
            "waypoints": waypoints,
            "cg_iterations": cg_iterations,
            "fisher_matvec_evaluations": fisher_matvecs,
            "gradient_evaluations": reverse_gradients,
            "autograd_equivalent_evaluations": (
                reverse_gradients + 2.0 * fisher_matvecs
            ),
        }


class BinaryFringeAttributor(FringeAttributor):
    """FRInGe with only its Fisher geometry collapsed to target-vs-rest."""

    def attribute(self, inputs, targets, seed):
        params = dict(self.spec.params)
        inverse_iters = int(params.pop("cg_max_iters", 20))
        inverse_rtol = float(params.pop("cg_rtol", 1e-3))
        inverse_atol = float(params.pop("cg_atol", 1e-5))

        def model_forward(x, target):
            return _target_logits(self.model, x, target)

        impl = BinaryFisherRaoIntegratedGradients(
            model=self.model,
            model_forward=model_forward,
            target_idx=targets,
        )
        attrs, stats = impl.attribute(
            x=inputs,
            kl_target=float(params["kl_target"]),
            delta_euc=float(params.get("delta_euc", 1000.0)),
            eta_max=float(params.get("eta_max", 1000.0)),
            lam=float(params.get("lambda_ratio", 1.0)),
            smoothing=bool(params.get("smoothing", True)),
            use_sobolev_preconditioner=bool(
                params.get("use_sobolev_preconditioner", True)
            ),
            blur_kernel_size=int(params.get("blur_kernel_size", 5)),
            blur_sigma=float(params.get("blur_sigma", 2.0)),
            gamma_step=float(params.get("gamma_step", 0.01)),
            gamma_prior=float(params.get("gamma_prior", 0.001)),
            Ainv_iters=inverse_iters,
            Ainv_rtol=inverse_rtol,
            Ainv_atol=inverse_atol,
            safety_buffer=float(params.get("safety_buffer", 0.1)),
            show_progress=bool(params.get("show_progress", False)),
        )
        stats["geometry"] = "binary_target_vs_rest"
        return attrs.detach(), stats

    def operation_budget(self, metadata):
        inverse = np.asarray(
            metadata.get("regularizer_inverse_iters", []), dtype=float
        )
        waypoints = float(metadata.get("total_steps", np.nan))
        inverse_iterations = float(np.nansum(inverse)) if inverse.size else 0.0
        # Every binary waypoint uses one target-logodds gradient and one target
        # logit gradient; the initial target-logit gradient is computed once.
        reverse_gradients = 2.0 * waypoints + 1.0
        return {
            "waypoints": waypoints,
            "regularizer_inverse_iterations": inverse_iterations,
            "fisher_matvec_evaluations": 0.0,
            "gradient_evaluations": reverse_gradients,
            "autograd_equivalent_evaluations": reverse_gradients,
        }


def make_candidate(
    model: torch.nn.Module,
    spec: CandidateSpec,
    baseline_pool: Optional[torch.Tensor] = None,
) -> AttributionCandidate:
    registry = {
        "ig": IGAttributor,
        "ig_smoothed_map": SmoothedIGAttributor,
        "smoothgrad": SmoothGradAttributor,
        "fringe": FringeAttributor,
        "fringe_binary": BinaryFringeAttributor,
    }
    if spec.method == "expected_gradients":
        if baseline_pool is None:
            raise ValueError("Expected Gradients requires a baseline pool.")
        return ExpectedGradientsAttributor(model, spec, baseline_pool)
    if spec.method not in registry:
        raise ValueError(f"Unknown equal-budget method: {spec.method}")
    return registry[spec.method](model, spec)


class QualityEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: Mapping[str, Any],
    ):
        self.model = model.eval()
        self.device = device
        self.enabled = bool(config.get("enabled", True))
        self.causal_baselines = list(
            config.get("causal_baselines", ["blur", "black", "white"])
        )
        self.causal_steps = int(config.get("causal_steps", 100))
        self.causal_chunk_size = int(config.get("causal_chunk_size", 1))
        self.mas_baselines = list(config.get("mas_baselines", ["blur"]))
        self.mas_step_size = int(config.get("mas_step_size", 224))
        self.mas_klen = int(config.get("mas_klen", 15))
        self.mas_nsig = float(config.get("mas_nsig", 3.0))
        self.mas_batch_size = int(config.get("mas_batch_size", 32))
        self.infid_n = int(config.get("infidelity_perturbations", 50))
        self.infid_noise = float(config.get("infidelity_noise_scale", 0.02))
        self.infid_chunk = int(config.get("infidelity_chunk_size", 10))
        self.causal = CausalMetricScorer(
            model, steps=self.causal_steps, device=device
        )
        self.infidelity = InfidelityScorer(model, device=device)
        self.sparseness = SparsenessScorer()

    def evaluate(
        self,
        inputs: torch.Tensor,
        attrs: torch.Tensor,
        targets: torch.Tensor,
        seed: int,
    ) -> Dict[str, float]:
        if not self.enabled:
            return {}
        result: Dict[str, float] = {}
        for baseline in self.causal_baselines:
            for normalized, label in [(True, "normalized"), (False, "raw")]:
                values = self.causal.score(
                    inputs,
                    attrs,
                    baseline_mode=str(baseline),
                    chunk_size=self.causal_chunk_size,
                    normalize=normalized,
                )
                result[f"insertion_auc_{label}_{baseline}"] = float(
                    values["insertion_auc"][0].item()
                )
                result[f"deletion_auc_{label}_{baseline}"] = float(
                    values["deletion_auc"][0].item()
                )

        for baseline in self.mas_baselines:
            scorer = BatchMASMetric(
                self.model,
                step_size=self.mas_step_size,
                baseline_type=str(baseline),
                klen=self.mas_klen,
                nsig=self.mas_nsig,
                device=self.device,
            )
            values = scorer.score_batch(
                inputs, attrs, inference_batch_size=self.mas_batch_size
            )
            valid = bool(values["valid_mask"][0].item())
            result[f"mas_valid_{baseline}"] = float(valid)
            result[f"mas_insertion_{baseline}"] = (
                float(values["insertion"][0].item()) if valid else float("nan")
            )
            result[f"mas_deletion_{baseline}"] = (
                float(values["deletion"][0].item()) if valid else float("nan")
            )

        result["infidelity"] = float(
            self.infidelity.score(
                inputs,
                attrs,
                targets,
                n_perturbations=self.infid_n,
                noise_scale=self.infid_noise,
                seed=int(seed),
                max_examples_per_pass=self.infid_chunk,
            )[0].item()
        )
        result["sparseness"] = float(self.sparseness.score(attrs)[0].item())
        return result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rss_bytes() -> int:
    # Linux ru_maxrss is KiB; macOS reports bytes.
    scale = 1 if platform.system() == "Darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


def measured_attribute(
    candidate: AttributionCandidate,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, float]]:
    device = inputs.device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        _sync(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_allocated = float("nan")
        baseline_reserved = float("nan")

    rss_before = _rss_bytes()
    _sync(device)
    start = time.perf_counter_ns()
    attrs, metadata = candidate.attribute(inputs, targets, seed=seed)
    # Captum's Gauss-Legendre weights may promote IG outputs to float64.
    # The repository metrics and all native methods operate in the input/model
    # dtype, so normalize at the attribution boundary and include that cost.
    attrs = attrs.to(device=device, dtype=inputs.dtype)
    _sync(device)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    rss_after = _rss_bytes()

    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        incremental_allocated = max(0, peak_allocated - baseline_allocated)
        incremental_reserved = max(0, peak_reserved - baseline_reserved)
    else:
        peak_allocated = peak_reserved = float("nan")
        incremental_allocated = incremental_reserved = float("nan")

    system = {
        "attribution_time_ms": float(elapsed_ms),
        "peak_gpu_allocated_bytes": float(peak_allocated),
        "peak_gpu_reserved_bytes": float(peak_reserved),
        "incremental_gpu_allocated_bytes": float(incremental_allocated),
        "incremental_gpu_reserved_bytes": float(incremental_reserved),
        "cpu_max_rss_before_bytes": float(rss_before),
        "cpu_max_rss_after_bytes": float(rss_after),
        "cpu_max_rss_growth_bytes": float(max(0, rss_after - rss_before)),
    }
    return attrs, metadata, system


def warmup_candidate(
    candidate: AttributionCandidate,
    warmup_paths: Sequence[str],
    transform,
    device: torch.device,
    runs: int,
    seed: int,
) -> None:
    if runs < 1:
        return
    for index in range(runs):
        path = warmup_paths[index % len(warmup_paths)]
        inputs = load_image(path, transform, device=device)
        with torch.no_grad():
            targets = candidate.model(inputs).argmax(dim=1)
        candidate.attribute(inputs, targets, seed=seed + index)
        _sync(device)
        del inputs


def measure_batch_throughput(
    candidate: AttributionCandidate,
    paths: Sequence[str],
    transform,
    device: torch.device,
    batch_size: int,
    runs: int,
    seed: int,
) -> Dict[str, float]:
    """Measure an explicitly reported batched attribution throughput point."""
    if batch_size < 2 or runs < 1:
        return {}
    selected = [paths[index % len(paths)] for index in range(batch_size)]
    inputs = torch.cat(
        [load_image(path, transform, device=device) for path in selected],
        dim=0,
    )
    with torch.no_grad():
        targets = candidate.model(inputs).argmax(dim=1)
    measurements = []
    for run_index in range(runs):
        attrs, _, systems = measured_attribute(
            candidate,
            inputs,
            targets,
            seed=seed + run_index,
        )
        measurements.append(systems)
        del attrs
    times = np.asarray(
        [item["attribution_time_ms"] for item in measurements], dtype=float
    )
    peaks = np.asarray(
        [item["peak_gpu_allocated_bytes"] for item in measurements],
        dtype=float,
    )
    finite_peaks = peaks[np.isfinite(peaks)]
    median_batch_ms = float(np.median(times))
    return {
        "throughput_batch_size": float(batch_size),
        "throughput_runs": float(runs),
        "median_batch_time_ms": median_batch_ms,
        "median_time_per_image_ms": median_batch_ms / batch_size,
        "images_per_second": 1000.0 * batch_size / median_batch_ms,
        "median_peak_gpu_allocated_bytes": (
            float(np.median(finite_peaks))
            if finite_peaks.size
            else float("nan")
        ),
    }


def summarize_numeric_rows(frame: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"n_images": int(len(frame))}
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = frame[column].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        summary[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(values.median()),
            "p25": float(values.quantile(0.25)),
            "p75": float(values.quantile(0.75)),
            "min": float(values.min()),
            "max": float(values.max()),
            "count": int(values.count()),
        }
    return summary


def candidate_summary_from_results(results_root: str | Path) -> pd.DataFrame:
    """Rebuild the candidate index from completed, self-describing outputs."""
    rows: List[Dict[str, Any]] = []
    root = Path(results_root)
    for config_path in sorted(root.rglob("candidate_config.json")):
        output = config_path.parent
        per_image = output / "per_image.parquet"
        if not per_image.exists():
            continue
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        frame = pd.read_parquet(per_image)
        if frame.empty:
            continue
        systems: Dict[str, Any] = {}
        systems_path = output / "systems.json"
        if systems_path.exists():
            with systems_path.open(encoding="utf-8") as handle:
                systems = json.load(handle)
        row: Dict[str, Any] = {
            "model": str(config["model"]),
            "method": str(config["method"]),
            "candidate_id": str(config["candidate_id"]),
            "method_label": method_display_name(str(config["method"])),
            "candidate_label": candidate_display_name(
                str(config["method"]),
                str(config["candidate_id"]),
                config.get("params", {}),
            ),
            "is_full": bool(config.get("is_full", False)),
            "evaluation_sha256": config.get("evaluation_sha256"),
            "n_images": int(len(frame)),
            "output_dir": str(output.resolve()),
            "median_attribution_time_ms": float(
                frame["attribution_time_ms"].median()
            ),
            "mean_attribution_time_ms": float(
                frame["attribution_time_ms"].mean()
            ),
            "initialization_time_ms": float(
                systems.get("initialization_time_ms", float("nan"))
            ),
            "gpu_name": systems.get("gpu_name"),
            "gpu_total_memory_bytes": systems.get(
                "gpu_total_memory_bytes"
            ),
            "torch_version": systems.get("torch"),
            "cuda_version": systems.get("cuda_version"),
        }
        for key, value in systems.get("batched_throughput", {}).items():
            row[f"throughput_{key}"] = value
        numeric = frame.select_dtypes(include=[np.number])
        for column in numeric.columns:
            values = numeric[column].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            row[f"mean_{column}"] = (
                float(values.mean()) if not values.empty else float("nan")
            )
            row[f"median_{column}"] = (
                float(values.median()) if not values.empty else float("nan")
            )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["model", "method", "median_attribution_time_ms"]
    )


def run_candidate(
    candidate: AttributionCandidate,
    eval_paths: Sequence[str],
    transform,
    output_dir: str | Path,
    quality: QualityEvaluator,
    device: torch.device,
    seed: int,
    save_attributions: bool,
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    attrs_dir = output / "attributions"
    if save_attributions:
        attrs_dir.mkdir(exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for image_index, path in enumerate(eval_paths):
        inputs = load_image(path, transform, device=device)
        with torch.no_grad():
            logits = candidate.model(inputs)
            targets = logits.argmax(dim=1)
            target_logit = _target_logits(
                candidate.model, inputs, targets
            ).item()
            target_probability = torch.softmax(logits, dim=1).gather(
                1, targets[:, None]
            )[0, 0].item()

        image_seed = int(seed + image_index)
        attrs, metadata, systems = measured_attribute(
            candidate, inputs, targets, seed=image_seed
        )
        completeness, completeness_relative = candidate.completeness(
            inputs, targets, attrs, metadata
        )
        metrics = quality.evaluate(
            inputs, attrs, targets, seed=image_seed
        )
        operations = candidate.operation_budget(metadata)

        endpoint = np.asarray(
            metadata.get("endpoint_error_kl", [float("nan")]), dtype=float
        )
        finite_endpoint = endpoint[np.isfinite(endpoint)]
        endpoint_kl = (
            float(finite_endpoint.mean())
            if finite_endpoint.size
            else float("nan")
        )
        image_id = hashlib.sha256(
            str(Path(path).resolve()).encode("utf-8")
        ).hexdigest()[:16]
        attr_path = None
        if save_attributions:
            attr_path = attrs_dir / f"{image_id}.pt"
            torch.save(
                {
                    "image_path": str(Path(path).resolve()),
                    "target": int(targets[0].item()),
                    "attribution": attrs.detach().cpu().to(torch.float16),
                    "metadata": json_safe(metadata),
                },
                attr_path,
            )

        row: Dict[str, Any] = {
            "image_index": image_index,
            "image_id": image_id,
            "image_path": str(Path(path).resolve()),
            "method": candidate.spec.method,
            "candidate_id": candidate.spec.candidate_id,
            "is_full": candidate.spec.is_full,
            "seed": image_seed,
            "target_class": int(targets[0].item()),
            "target_logit": float(target_logit),
            "target_probability": float(target_probability),
            "completeness_delta": completeness,
            "completeness_relative_error": completeness_relative,
            "endpoint_error_kl": endpoint_kl,
            "attribution_path": (
                str(attr_path.relative_to(output)) if attr_path else None
            ),
            **systems,
            **{f"ops_{key}": value for key, value in operations.items()},
            **metrics,
        }
        rows.append(row)
        del inputs, attrs

    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "per_image.parquet", index=False)
    frame.to_csv(output / "per_image.csv", index=False)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summarize_numeric_rows(frame)), handle, indent=2)
    return frame


def system_manifest(device: torch.device) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        manifest["gpu_name"] = props.name
        manifest["gpu_total_memory_bytes"] = props.total_memory
    return manifest


def paired_bootstrap_mean(
    differences: Sequence[float],
    seed: int = 42,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> Dict[str, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(
        values, size=(int(n_resamples), values.size), replace=True
    ).mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha)),
    }


def closest_runtime_candidates(
    candidate_summary: pd.DataFrame,
    anchor_method: str,
    anchor_candidate: str,
    multiples: Sequence[float],
) -> pd.DataFrame:
    required = {
        "model",
        "method",
        "candidate_id",
        "median_attribution_time_ms",
    }
    missing = required - set(candidate_summary.columns)
    if missing:
        raise ValueError(f"Candidate summary is missing columns: {sorted(missing)}")

    rows: List[Dict[str, Any]] = []
    for model, model_frame in candidate_summary.groupby("model"):
        anchor = model_frame[
            (model_frame["method"] == anchor_method)
            & (model_frame["candidate_id"] == anchor_candidate)
        ]
        if len(anchor) != 1:
            raise ValueError(
                f"Expected exactly one anchor {anchor_method}/{anchor_candidate} "
                f"for {model}, found {len(anchor)}."
            )
        anchor_ms = float(anchor.iloc[0]["median_attribution_time_ms"])
        for multiple in multiples:
            target_ms = anchor_ms * float(multiple)
            for method, method_frame in model_frame.groupby("method"):
                valid = method_frame[
                    method_frame["median_attribution_time_ms"] > 0
                ].copy()
                if valid.empty:
                    continue
                valid["runtime_log_distance"] = np.abs(
                    np.log(
                        valid["median_attribution_time_ms"].astype(float)
                        / target_ms
                    )
                )
                selected = valid.sort_values(
                    ["runtime_log_distance", "candidate_id"]
                ).iloc[0]
                rows.append(
                    {
                        "model": model,
                        "budget_multiple": float(multiple),
                        "anchor_method": anchor_method,
                        "anchor_candidate": anchor_candidate,
                        "anchor_median_ms": anchor_ms,
                        "target_runtime_ms": target_ms,
                        "method": method,
                        "candidate_id": selected["candidate_id"],
                        "median_attribution_time_ms": float(
                            selected["median_attribution_time_ms"]
                        ),
                        "achieved_budget_multiple": float(
                            selected["median_attribution_time_ms"] / anchor_ms
                        ),
                        "runtime_log_distance": float(
                            selected["runtime_log_distance"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def closest_budget_candidates(
    candidate_summary: pd.DataFrame,
    anchor_method: str,
    anchor_candidate: str,
    multiples: Sequence[float],
    value_column: str,
    budget_name: str,
) -> pd.DataFrame:
    """Select each method's closest measured candidate for a generic budget."""
    required = {"model", "method", "candidate_id", value_column}
    missing = required - set(candidate_summary.columns)
    if missing:
        raise ValueError(f"Candidate summary is missing columns: {sorted(missing)}")

    rows: List[Dict[str, Any]] = []
    for model, model_frame in candidate_summary.groupby("model"):
        anchor = model_frame[
            (model_frame["method"] == anchor_method)
            & (model_frame["candidate_id"] == anchor_candidate)
        ]
        if len(anchor) != 1:
            raise ValueError(
                f"Expected exactly one anchor {anchor_method}/{anchor_candidate} "
                f"for {model}, found {len(anchor)}."
            )
        anchor_value = float(anchor.iloc[0][value_column])
        if not math.isfinite(anchor_value) or anchor_value <= 0:
            continue
        for multiple in multiples:
            target = anchor_value * float(multiple)
            for method, method_frame in model_frame.groupby("method"):
                valid = method_frame[
                    np.isfinite(method_frame[value_column].astype(float))
                    & (method_frame[value_column].astype(float) > 0)
                ].copy()
                if valid.empty:
                    continue
                valid["budget_log_distance"] = np.abs(
                    np.log(valid[value_column].astype(float) / target)
                )
                selected = valid.sort_values(
                    ["budget_log_distance", "candidate_id"]
                ).iloc[0]
                selected_value = float(selected[value_column])
                rows.append(
                    {
                        "model": model,
                        "budget_type": budget_name,
                        "budget_multiple": float(multiple),
                        "anchor_method": anchor_method,
                        "anchor_candidate": anchor_candidate,
                        "anchor_value": anchor_value,
                        "target_value": target,
                        "method": method,
                        "candidate_id": selected["candidate_id"],
                        "selected_value": selected_value,
                        "achieved_budget_multiple": selected_value
                        / anchor_value,
                        "budget_log_distance": float(
                            selected["budget_log_distance"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def flatten_candidates(methods_config: Mapping[str, Any]) -> List[CandidateSpec]:
    specs: List[CandidateSpec] = []
    for method, method_config in methods_config.items():
        config = dict(method_config)
        if not bool(config.get("enabled", True)):
            continue
        defaults = dict(config.get("defaults", {}))
        candidates = list(config.get("candidates", []))
        for candidate in candidates:
            candidate = dict(candidate)
            candidate_id = str(candidate.pop("id"))
            is_full = bool(candidate.pop("is_full", False))
            specs.append(
                CandidateSpec(
                    method=str(method),
                    candidate_id=candidate_id,
                    params={**defaults, **candidate},
                    is_full=is_full,
                )
            )
    if not specs:
        raise ValueError("No enabled candidates found.")
    return specs


def load_baseline_pool(
    baseline_paths: Sequence[str], transform
) -> torch.Tensor:
    return torch.cat(
        [load_image(path, transform, device="cpu") for path in baseline_paths],
        dim=0,
    )
