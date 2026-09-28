#!/usr/bin/env python3
"""Analyse one Slurm run under ``results/comparisons/EqualBudgetComparison``.

The benchmark writes one directory per Slurm array submission:

    results/comparisons/EqualBudgetComparison/<run-id>/jobs/<model>_<method>/...

This entry point discovers those runs, checks that the expected IG, full
multiclass FRInGe, and binary FRInGe candidate grids are complete, and then
delegates the statistical analysis to
``benchmarks.equal_budget.analyse_equal_budget``.
Runs are never merged because runtime comparisons across separate scheduler
allocations or GPU types are not necessarily valid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

from benchmarks.equal_budget.analyse_equal_budget import analyse
from benchmarks.equal_budget.core import (
    candidate_summary_from_results,
    flatten_candidates,
    method_display_name,
)


FOCUSED_METHODS = ("ig", "fringe", "fringe_binary")
METHOD_COLORS = {
    "ig": "#4c78a8",
    "fringe": "#e45756",
    "fringe_binary": "#54a24b",
}
GPU_MARKERS = ("o", "s", "^", "D", "P", "X")
QUALITY_METRICS = (
    ("mean_mas_insertion_blur", "MAS insertion AUC ↑"),
    ("mean_mas_deletion_blur", "MAS deletion AUC ↓"),
    (
        "mean_insertion_auc_normalized_blur",
        "Normalized insertion AUC ↑",
    ),
    (
        "mean_deletion_auc_normalized_blur",
        "Normalized deletion AUC ↓",
    ),
    ("mean_infidelity", "Infidelity ↓"),
    ("mean_sparseness", "Sparseness ↑"),
)


def _run_sort_key(path: Path) -> tuple[int, int, str]:
    """Prefer numeric Slurm IDs, then modification time, then directory name."""
    try:
        numeric_id = int(path.name)
    except ValueError:
        numeric_id = -1
    return numeric_id, path.stat().st_mtime_ns, path.name


def discover_runs(comparison_root: Path) -> List[Path]:
    """Return run roots containing a ``jobs`` directory."""
    if (comparison_root / "jobs").is_dir():
        return [comparison_root]
    if not comparison_root.is_dir():
        return []
    return sorted(
        (
            child
            for child in comparison_root.iterdir()
            if child.is_dir() and (child / "jobs").is_dir()
        ),
        key=_run_sort_key,
    )


def select_run(comparison_root: Path, run_id: str | None) -> Path:
    if run_id:
        selected = comparison_root / str(run_id)
        if not (selected / "jobs").is_dir():
            raise FileNotFoundError(
                f"Run {run_id!r} has no jobs directory under {comparison_root}."
            )
        return selected.resolve()

    runs = discover_runs(comparison_root)
    if not runs:
        raise FileNotFoundError(
            f"No equal-budget run directories found under {comparison_root}."
        )
    return runs[-1].resolve()


def expected_grid(
    config_path: Path,
    methods: Sequence[str],
) -> tuple[List[str], Dict[str, set[str]]]:
    cfg = OmegaConf.load(config_path)
    models = [str(model) for model in cfg.avail_models]
    method_config = OmegaConf.to_container(cfg.methods, resolve=True)
    specs = flatten_candidates(method_config)
    expected: Dict[str, set[str]] = defaultdict(set)
    for spec in specs:
        if spec.method in methods:
            expected[spec.method].add(spec.candidate_id)
    return models, dict(expected)


def coverage_table(
    summary: pd.DataFrame,
    models: Iterable[str],
    expected: Mapping[str, set[str]],
) -> pd.DataFrame:
    rows = []
    for model in models:
        model_frame = summary[summary["model"] == model]
        for method, expected_ids in expected.items():
            completed_ids = set(
                model_frame.loc[
                    model_frame["method"] == method, "candidate_id"
                ].astype(str)
            )
            missing = sorted(expected_ids - completed_ids)
            unexpected = sorted(completed_ids - expected_ids)
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "method_label": method_display_name(method),
                    "completed": len(completed_ids & expected_ids),
                    "expected": len(expected_ids),
                    "status": "complete" if not missing else "incomplete",
                    "missing_candidates": ",".join(missing),
                    "unexpected_candidates": ",".join(unexpected),
                }
            )
    return pd.DataFrame(rows)


def plot_coverage(coverage: pd.DataFrame, output_root: Path) -> None:
    """Plot completed versus expected candidates for every model and method."""
    labels = [
        f"{row.model}\n{row.method_label}"
        for row in coverage.itertuples(index=False)
    ]
    x = np.arange(len(coverage))
    expected = coverage["expected"].to_numpy(float)
    completed = coverage["completed"].to_numpy(float)

    fig, ax = plt.subplots(
        figsize=(max(10.0, 1.15 * len(coverage)), 5.5)
    )
    ax.bar(x, expected, color="#d9dde3", label="Expected")
    ax.bar(x, completed, color="#3976af", label="Completed")
    for xpos, done, total in zip(x, completed, expected):
        ax.text(
            xpos,
            done + 0.08,
            f"{int(done)}/{int(total)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Candidate outputs")
    ax.set_title("Equal-budget benchmark coverage")
    ax.set_ylim(0, max(expected, default=1.0) + 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "candidate_coverage.png", dpi=200)
    fig.savefig(output_root / "candidate_coverage.pdf")
    plt.close(fig)


def plot_partial_runtimes(
    summary: pd.DataFrame,
    output_root: Path,
) -> None:
    """Plot every completed candidate, even while the array is incomplete."""
    methods = list(FOCUSED_METHODS)
    available_models = sorted(summary["model"].astype(str).unique())
    if not available_models:
        return

    fig, axes = plt.subplots(
        len(available_models),
        1,
        figsize=(11, max(4.5, 4.2 * len(available_models))),
        squeeze=False,
    )
    gpu_names = sorted(
        summary.get("gpu_name", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .unique()
    )
    for axis, model in zip(axes[:, 0], available_models):
        model_frame = summary[summary["model"] == model]
        plotted_runtimes: List[float] = []
        for method_index, method in enumerate(methods):
            method_frame = model_frame[
                model_frame["method"] == method
            ].sort_values("median_attribution_time_ms")
            if method_frame.empty:
                continue
            offsets = np.linspace(
                -0.22, 0.22, num=len(method_frame)
            )
            xs = method_index + offsets
            ys = method_frame["median_attribution_time_ms"].to_numpy(float)
            plotted_runtimes.extend(ys.tolist())
            axis.scatter(
                xs,
                ys,
                s=48,
                color=METHOD_COLORS[method],
                label=method,
                zorder=3,
            )
            for xpos, ypos, candidate_id in zip(
                xs,
                ys,
                method_frame.get(
                    "candidate_label", method_frame["candidate_id"]
                ).astype(str),
            ):
                axis.annotate(
                    candidate_id,
                    (xpos, ypos),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                    rotation=18,
                )
        axis.set_yscale("log")
        if plotted_runtimes:
            axis.set_ylim(
                min(plotted_runtimes) / 1.5,
                max(plotted_runtimes) * 2.0,
            )
        axis.set_xlim(-0.5, len(methods) - 0.35)
        axis.set_xticks(
            range(len(methods)),
            [method_display_name(method) for method in methods],
        )
        axis.set_ylabel("Median attribution time (ms, log)")
        axis.set_title(model)
        axis.grid(axis="y", which="both", alpha=0.25)
    axes[-1, 0].set_xlabel("Method")
    hardware_note = (
        " | GPUs: " + ", ".join(gpu_names)
        if gpu_names
        else ""
    )
    fig.suptitle(
        "Completed equal-budget candidates (partial results included)"
        + hardware_note
    )
    fig.tight_layout()
    fig.savefig(output_root / "candidate_runtime_partial.png", dpi=200)
    fig.savefig(output_root / "candidate_runtime_partial.pdf")
    plt.close(fig)


def _gpu_marker_map(summary: pd.DataFrame) -> Dict[str, str]:
    gpu_names = sorted(
        summary.get("gpu_name", pd.Series(dtype=str))
        .fillna("unknown GPU")
        .astype(str)
        .unique()
    )
    return {
        gpu: GPU_MARKERS[index % len(GPU_MARKERS)]
        for index, gpu in enumerate(gpu_names)
    }


def _plot_candidate_points(
    axis,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    gpu_markers: Mapping[str, str],
    annotate: bool = True,
) -> None:
    """Draw method-coloured, hardware-shaped points with candidate labels."""
    clean = frame.dropna(subset=[x_column, y_column])
    for point_index, (_, row) in enumerate(clean.iterrows()):
        method = str(row["method"])
        gpu_name = str(row.get("gpu_name") or "unknown GPU")
        xpos = float(row[x_column])
        ypos = float(row[y_column])
        axis.scatter(
            xpos,
            ypos,
            s=48,
            color=METHOD_COLORS.get(method, "#777777"),
            marker=gpu_markers[gpu_name],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        if annotate:
            candidate_label = str(
                row.get("candidate_label") or row["candidate_id"]
            )
            vertical_offset = 4 if point_index % 2 == 0 else -9
            axis.annotate(
                candidate_label,
                (xpos, ypos),
                xytext=(3, vertical_offset),
                textcoords="offset points",
                fontsize=6,
                alpha=0.9,
            )


def _figure_legends(
    fig,
    summary: pd.DataFrame,
    gpu_markers: Mapping[str, str],
) -> None:
    present_methods = [
        method
        for method in FOCUSED_METHODS
        if method in set(summary["method"].astype(str))
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=METHOD_COLORS[method],
            label=method_display_name(method),
            markersize=7,
        )
        for method in present_methods
    ]
    gpu_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markerfacecolor="#777777",
            markeredgecolor="white",
            color="#777777",
            label=gpu,
            markersize=7,
        )
        for gpu, marker in gpu_markers.items()
    ]
    fig.legend(
        handles=method_handles + gpu_handles,
        loc="lower center",
        ncol=max(1, min(6, len(method_handles) + len(gpu_handles))),
        fontsize=8,
        frameon=False,
    )


def plot_partial_memory(
    summary: pd.DataFrame,
    output_root: Path,
) -> None:
    """Plot peak and attribution-incremental CUDA memory against latency."""
    memory_columns = (
        ("median_peak_gpu_allocated_bytes", "Peak allocated CUDA memory (GiB)"),
        (
            "median_incremental_gpu_allocated_bytes",
            "Incremental attribution memory (GiB)",
        ),
    )
    available = [
        item for item in memory_columns if item[0] in summary.columns
    ]
    models = sorted(summary["model"].astype(str).unique())
    if not available or not models:
        return

    gpu_markers = _gpu_marker_map(summary)
    fig, axes = plt.subplots(
        len(models),
        len(available),
        figsize=(7.2 * len(available), max(4.8, 4.3 * len(models))),
        squeeze=False,
    )
    for row_index, model in enumerate(models):
        model_frame = summary[summary["model"] == model].copy()
        for column_index, (memory_column, ylabel) in enumerate(available):
            axis = axes[row_index, column_index]
            plot_frame = model_frame.copy()
            plot_frame["_memory_gib"] = (
                plot_frame[memory_column].astype(float) / (1024.0 ** 3)
            )
            _plot_candidate_points(
                axis,
                plot_frame,
                "median_attribution_time_ms",
                "_memory_gib",
                gpu_markers,
                annotate=False,
            )
            axis.set_xscale("log")
            axis.set_xlabel("Median attribution time (ms, log)")
            axis.set_ylabel(ylabel)
            axis.set_title(f"{model} — {ylabel}")
            axis.grid(alpha=0.25)
    fig.suptitle("Memory footprint versus attribution latency (partial results)")
    _figure_legends(fig, summary, gpu_markers)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(output_root / "memory_vs_time_partial.png", dpi=200)
    fig.savefig(output_root / "memory_vs_time_partial.pdf")
    plt.close(fig)


def plot_partial_quality_time(
    summary: pd.DataFrame,
    output_root: Path,
) -> None:
    """Plot attribution quality metrics against measured attribution latency."""
    available = [
        item
        for item in QUALITY_METRICS
        if item[0] in summary.columns and summary[item[0]].notna().any()
    ]
    models = sorted(summary["model"].astype(str).unique())
    if not available or not models:
        return

    gpu_markers = _gpu_marker_map(summary)
    columns = 3
    rows_per_model = int(np.ceil(len(available) / columns))
    total_rows = len(models) * rows_per_model
    fig, axes = plt.subplots(
        total_rows,
        columns,
        figsize=(18, max(5.0, 4.0 * total_rows)),
        squeeze=False,
    )

    for model_index, model in enumerate(models):
        model_frame = summary[summary["model"] == model]
        base_row = model_index * rows_per_model
        for metric_index, (metric, label) in enumerate(available):
            axis = axes[
                base_row + metric_index // columns,
                metric_index % columns,
            ]
            _plot_candidate_points(
                axis,
                model_frame,
                "median_attribution_time_ms",
                metric,
                gpu_markers,
            )
            axis.set_xscale("log")
            axis.set_xlabel("Median attribution time (ms, log)")
            axis.set_ylabel(label)
            axis.set_title(f"{model} — {label}")
            axis.grid(alpha=0.25)

        for unused_index in range(len(available), rows_per_model * columns):
            axes[
                base_row + unused_index // columns,
                unused_index % columns,
            ].set_visible(False)

    fig.suptitle(
        "Attribution quality versus latency (partial results; arrows show "
        "preferred direction)"
    )
    _figure_legends(fig, summary, gpu_markers)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(output_root / "quality_vs_time_partial.png", dpi=200)
    fig.savefig(output_root / "quality_vs_time_partial.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover and analyse IG versus full and binary FRInGe results "
            "from the equal-budget Slurm array."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "comparisons" / "EqualBudgetComparison",
        help="Parent directory containing <run-id>/jobs directories.",
    )
    parser.add_argument(
        "--run-id",
        help="Slurm array job ID to analyse; defaults to the latest run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "EqualBudget_config.yaml",
        help="Configuration used to determine expected models and candidates.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Models to include, for example: --models resnet18 resnet50. "
            "Defaults to every model in the benchmark configuration."
        ),
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help=(
            "Print candidate coverage and create partial plots without running "
            "runtime matching or bootstrap analysis."
        ),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail instead of analysing the completed subset of an active run.",
    )
    parser.add_argument(
        "--budget-multiples",
        nargs="+",
        type=float,
        default=[1, 2, 5, 10],
    )
    parser.add_argument("--anchor-candidate", default="ig_32")
    parser.add_argument("--reference-method", default="fringe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison_root = args.results_root.expanduser().resolve()
    run_root = select_run(comparison_root, args.run_id)
    jobs_root = run_root / "jobs"

    summary = candidate_summary_from_results(jobs_root)
    if summary.empty:
        raise FileNotFoundError(
            f"No completed candidates were found under {jobs_root}."
        )

    models, expected = expected_grid(
        args.config.expanduser().resolve(), FOCUSED_METHODS
    )
    if args.models:
        requested_models = [str(model) for model in args.models]
        unknown_models = sorted(set(requested_models) - set(models))
        if unknown_models:
            raise ValueError(
                "Requested models are not in the benchmark configuration: "
                + ", ".join(unknown_models)
            )
        models = requested_models
        summary = summary[summary["model"].isin(models)].copy()
        if summary.empty:
            raise FileNotFoundError(
                "No completed candidates were found for the requested models: "
                + ", ".join(models)
            )
    coverage = coverage_table(summary, models, expected)
    run_root.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(run_root / "analysis_coverage.csv", index=False)
    summary.to_csv(run_root / "partial_candidate_summary.csv", index=False)
    plot_coverage(coverage, run_root)
    plot_partial_runtimes(summary, run_root)
    plot_partial_memory(summary, run_root)
    plot_partial_quality_time(summary, run_root)

    print(f"Selected run: {run_root.name}")
    print(f"Completed candidate outputs: {len(summary)}")
    print(coverage.to_string(index=False))

    if args.status_only:
        print(f"\nPartial plots written to: {run_root}")
        return 0

    incomplete = coverage[coverage["status"] != "complete"]
    if args.require_complete and not incomplete.empty:
        raise RuntimeError(
            "The selected run is incomplete. Wait for the Slurm array to "
            "finish, inspect failed tasks, or use --status-only to monitor it."
        )

    anchor_mask = (
        (summary["method"] == "ig")
        & (summary["candidate_id"] == args.anchor_candidate)
    )
    analysable_models = sorted(
        summary.loc[anchor_mask, "model"].astype(str).unique()
    )
    excluded_models = sorted(set(models) - set(analysable_models))
    if not analysable_models:
        print(
            "\nNo model has a completed IG anchor "
            f"({args.anchor_candidate}); partial plots were created, but "
            "runtime matching cannot run yet."
        )
        return 0

    if not incomplete.empty:
        print(
            "\nWARNING: the run is incomplete. Statistical outputs use only "
            "currently completed candidates and may change as jobs finish."
        )

    analyse(
        results_root=run_root,
        anchor_method="ig",
        anchor_candidate=args.anchor_candidate,
        budget_multiples=args.budget_multiples,
        reference_method=args.reference_method,
        seed=args.seed,
        n_resamples=args.bootstrap_resamples,
        models=analysable_models,
    )

    manifest_path = run_root / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "coverage_complete": bool(incomplete.empty),
            "partial_results": bool(not incomplete.empty),
            "analysed_models": analysable_models,
            "excluded_models_without_anchor": excluded_models,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nAnalysis complete: {run_root}")
    print(f"Main table: {run_root / 'equal_budget_table.md'}")
    print(f"Report:     {run_root / 'EQUAL_BUDGET_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
