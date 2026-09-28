"""Measured equal-budget benchmarking for attribution methods."""

from .core import (
    CandidateSpec,
    build_disjoint_split,
    closest_budget_candidates,
    closest_runtime_candidates,
    paired_bootstrap_mean,
)

__all__ = [
    "CandidateSpec",
    "build_disjoint_split",
    "closest_budget_candidates",
    "closest_runtime_candidates",
    "paired_bootstrap_mean",
]
