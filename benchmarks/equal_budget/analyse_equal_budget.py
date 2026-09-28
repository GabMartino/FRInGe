from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.equal_budget.core import (
    candidate_summary_from_results,
    closest_budget_candidates,
    closest_runtime_candidates,
    paired_bootstrap_mean,
)


QUALITY_DIRECTIONS = {
    "insertion_auc_normalized_blur": "higher",
    "deletion_auc_normalized_blur": "lower",
    "insertion_auc_raw_blur": "higher",
    "deletion_auc_raw_blur": "lower",
    "mas_insertion_blur": "higher",
    "mas_deletion_blur": "lower",
    "infidelity": "lower",
    "sparseness": "higher",
    "completeness_delta": "lower",
    "completeness_relative_error": "lower",
    "endpoint_error_kl": "lower",
}


def _candidate_frame(row: pd.Series) -> pd.DataFrame:
    return pd.read_parquet(Path(row["output_dir"]) / "per_image.parquet")


def _attach_display_labels(
    frame: pd.DataFrame, summary: pd.DataFrame
) -> pd.DataFrame:
    if frame.empty:
        return frame
    label_columns = [
        column
        for column in ("method_label", "candidate_label")
        if column in summary.columns
    ]
    if not label_columns:
        return frame
    labels = summary[
        ["model", "method", "candidate_id", *label_columns]
    ].drop_duplicates()
    return frame.merge(
        labels,
        on=["model", "method", "candidate_id"],
        how="left",
        validate="many_to_one",
    )


def _selected_quality(selection: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    merged = selection.merge(
        summary,
        on=["model", "method", "candidate_id"],
        how="left",
        suffixes=("", "_candidate"),
    )
    return merged


def paired_statistics(
    selection: pd.DataFrame,
    summary: pd.DataFrame,
    reference_method: str,
    metrics: Sequence[str],
    seed: int,
    n_resamples: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    summary_lookup = summary.set_index(["model", "method", "candidate_id"])

    for (model, multiple), budget in selection.groupby(
        ["model", "budget_multiple"]
    ):
        reference_rows = budget[budget["method"] == reference_method]
        if len(reference_rows) != 1:
            continue
        reference_selected = reference_rows.iloc[0]
        reference_info = summary_lookup.loc[
            (
                model,
                reference_method,
                reference_selected["candidate_id"],
            )
        ]
        reference = _candidate_frame(reference_info)

        for _, selected in budget.iterrows():
            info = summary_lookup.loc[
                (model, selected["method"], selected["candidate_id"])
            ]
            candidate = _candidate_frame(info)
            paired = candidate.merge(
                reference,
                on="image_id",
                suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            for metric in metrics:
                left = f"{metric}_candidate"
                right = f"{metric}_reference"
                if left not in paired or right not in paired:
                    continue
                difference = (
                    paired[left].to_numpy(float)
                    - paired[right].to_numpy(float)
                )
                stats = paired_bootstrap_mean(
                    difference,
                    seed=seed,
                    n_resamples=n_resamples,
                )
                direction = QUALITY_DIRECTIONS.get(metric, "unspecified")
                advantage = (
                    -stats["mean"] if direction == "lower" else stats["mean"]
                )
                rows.append(
                    {
                        "model": model,
                        "budget_multiple": multiple,
                        "method": selected["method"],
                        "candidate_id": selected["candidate_id"],
                        "method_label": selected.get(
                            "method_label", selected["method"]
                        ),
                        "candidate_label": selected.get(
                            "candidate_label", selected["candidate_id"]
                        ),
                        "reference_method": reference_method,
                        "reference_candidate_id": reference_selected[
                            "candidate_id"
                        ],
                        "metric": metric,
                        "preferred_direction": direction,
                        "paired_difference_candidate_minus_reference": stats[
                            "mean"
                        ],
                        "candidate_advantage": advantage,
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "n_pairs": stats["n"],
                    }
                )
    return pd.DataFrame(rows)


def _plot_pareto(
    summary: pd.DataFrame, output: Path, metric: str, direction: str
) -> None:
    quality_column = f"mean_{metric}"
    if quality_column not in summary:
        return
    clean = summary.dropna(
        subset=["median_attribution_time_ms", quality_column]
    )
    if clean.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, frame in clean.groupby("method"):
        display_method = (
            str(frame["method_label"].iloc[0])
            if "method_label" in frame
            else str(method)
        )
        ax.scatter(
            frame["median_attribution_time_ms"],
            frame[quality_column],
            label=display_method,
            alpha=0.85,
        )
        for _, row in frame.iterrows():
            ax.annotate(
                row.get("candidate_label", row["candidate_id"]),
                (row["median_attribution_time_ms"], row[quality_column]),
                fontsize=6,
                alpha=0.8,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Median attribution runtime per image (ms, log scale)")
    ax.set_ylabel(f"{metric} ({direction} is better)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / f"pareto_{metric}.png", dpi=200)
    fig.savefig(output / f"pareto_{metric}.pdf")
    plt.close(fig)


def _plot_memory_quality(
    summary: pd.DataFrame, output: Path, metric: str, direction: str
) -> None:
    quality_column = f"mean_{metric}"
    memory_column = "median_peak_gpu_allocated_bytes"
    if quality_column not in summary or memory_column not in summary:
        return
    clean = summary.dropna(subset=[memory_column, quality_column])
    clean = clean[clean[memory_column] > 0]
    if clean.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, frame in clean.groupby("method"):
        display_method = (
            str(frame["method_label"].iloc[0])
            if "method_label" in frame
            else str(method)
        )
        ax.scatter(
            frame[memory_column] / (1024 ** 3),
            frame[quality_column],
            label=display_method,
            alpha=0.85,
        )
    ax.set_xlabel("Peak CUDA allocated memory (GiB)")
    ax.set_ylabel(f"{metric} ({direction} is better)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / f"memory_vs_{metric}.png", dpi=200)
    fig.savefig(output / f"memory_vs_{metric}.pdf")
    plt.close(fig)


def fringe_gain_retention(
    summary: pd.DataFrame,
    anchor_method: str,
    anchor_candidate: str,
) -> pd.DataFrame:
    """Quantify gain retention for full and binary FRInGe candidate grids."""
    rows: List[Dict[str, object]] = []
    full_mask = summary["is_full"].map(
        lambda value: str(value).lower() in {"true", "1", "yes"}
    )
    for model, model_frame in summary.groupby("model"):
        anchor = model_frame[
            (model_frame["method"] == anchor_method)
            & (model_frame["candidate_id"] == anchor_candidate)
        ]
        if len(anchor) != 1:
            continue
        for fringe_method in ("fringe", "fringe_binary"):
            full = model_frame[
                (model_frame["method"] == fringe_method)
                & full_mask.loc[model_frame.index]
            ]
            fringe = model_frame[model_frame["method"] == fringe_method]
            if len(full) != 1:
                continue
            for metric, direction in QUALITY_DIRECTIONS.items():
                column = f"mean_{metric}"
                if column not in summary:
                    continue
                anchor_value = float(anchor.iloc[0][column])
                full_value = float(full.iloc[0][column])
                sign = -1.0 if direction == "lower" else 1.0
                full_gain = sign * (full_value - anchor_value)
                for _, candidate in fringe.iterrows():
                    candidate_value = float(candidate[column])
                    candidate_gain = sign * (candidate_value - anchor_value)
                    retention = (
                        candidate_gain / full_gain
                        if math.isfinite(full_gain) and abs(full_gain) > 1e-12
                        else float("nan")
                    )
                    rows.append(
                        {
                            "model": model,
                            "method": fringe_method,
                            "metric": metric,
                            "preferred_direction": direction,
                            "candidate_id": candidate["candidate_id"],
                            "anchor_value": anchor_value,
                            "full_fringe_value": full_value,
                            "candidate_value": candidate_value,
                            "full_gain_over_anchor": full_gain,
                            "candidate_gain_over_anchor": candidate_gain,
                            "fraction_of_full_gain_retained": retention,
                        }
                    )
    return pd.DataFrame(rows)


def validate_comparison(summary: pd.DataFrame) -> None:
    """Reject result mixtures that invalidate paired or runtime comparisons."""
    key_columns = ["model", "method", "candidate_id"]
    duplicate_mask = summary.duplicated(key_columns, keep=False)
    if duplicate_mask.any():
        duplicates = summary.loc[duplicate_mask, key_columns].drop_duplicates()
        raise ValueError(
            "Duplicate completed candidates were found under the result root:\n"
            + duplicates.to_string(index=False)
        )

    if "evaluation_sha256" in summary:
        hashes = summary["evaluation_sha256"].dropna().astype(str).unique()
        if len(hashes) > 1:
            raise ValueError(
                "Candidates use different evaluation splits. Do not aggregate "
                "these runs into one paired comparison."
            )

    if "gpu_name" in summary:
        hardware = summary.dropna(subset=["gpu_name"])
        gpu_names = hardware["gpu_name"].astype(str).unique()
        if len(gpu_names) > 1:
            by_gpu = (
                hardware.groupby("gpu_name")
                .size()
                .sort_values(ascending=False)
                .to_string()
            )
            raise ValueError(
                "Mixed GPU models were found. Measured equal-runtime matching "
                "requires one GPU model across the comparison. Re-run with a "
                "Slurm GPU constraint or node list.\n" + by_gpu
            )


def analyse(
    results_root: Path,
    anchor_method: str,
    anchor_candidate: str,
    budget_multiples: Sequence[float],
    reference_method: str,
    seed: int,
    n_resamples: int,
    models: Sequence[str] | None = None,
) -> None:
    summary = candidate_summary_from_results(results_root)
    if summary.empty:
        raise FileNotFoundError(
            f"No completed candidates found under {results_root}. Run the grid first."
        )
    if models is not None:
        requested_models = {str(model) for model in models}
        summary = summary[summary["model"].isin(requested_models)].copy()
        if summary.empty:
            raise ValueError(
                "No completed candidates remain after filtering to models: "
                + ", ".join(sorted(requested_models))
            )
    validate_comparison(summary)
    summary.to_csv(results_root / "candidate_summary.csv", index=False)
    summary.to_parquet(
        results_root / "candidate_summary.parquet", index=False
    )
    selection = closest_runtime_candidates(
        summary,
        anchor_method=anchor_method,
        anchor_candidate=anchor_candidate,
        multiples=budget_multiples,
    )
    selection = _attach_display_labels(selection, summary)
    selection.to_csv(
        results_root / "equal_budget_selection.csv", index=False
    )

    memory_selection = (
        closest_budget_candidates(
            summary,
            anchor_method=anchor_method,
            anchor_candidate=anchor_candidate,
            multiples=budget_multiples,
            value_column="median_peak_gpu_allocated_bytes",
            budget_name="peak_gpu_allocated_bytes",
        )
        if "median_peak_gpu_allocated_bytes" in summary
        else pd.DataFrame()
    )
    memory_selection = _attach_display_labels(memory_selection, summary)
    memory_selection.to_csv(
        results_root / "equal_memory_selection.csv", index=False
    )
    operation_selection = (
        closest_budget_candidates(
            summary,
            anchor_method=anchor_method,
            anchor_candidate=anchor_candidate,
            multiples=budget_multiples,
            value_column="mean_ops_autograd_equivalent_evaluations",
            budget_name="autograd_equivalent_evaluations",
        )
        if "mean_ops_autograd_equivalent_evaluations" in summary
        else pd.DataFrame()
    )
    operation_selection = _attach_display_labels(
        operation_selection, summary
    )
    operation_selection.to_csv(
        results_root / "equal_autograd_selection.csv", index=False
    )

    selected_quality = _selected_quality(selection, summary)
    selected_quality.to_csv(
        results_root / "equal_budget_quality_summary.csv", index=False
    )
    table_columns = [
        "model",
        "budget_multiple",
        "target_runtime_ms",
        "method",
        "method_label",
        "candidate_id",
        "candidate_label",
        "median_attribution_time_ms",
        "achieved_budget_multiple",
        "runtime_log_distance",
        "median_peak_gpu_allocated_bytes",
        "mean_mas_insertion_blur",
        "mean_mas_deletion_blur",
        "mean_insertion_auc_normalized_blur",
        "mean_deletion_auc_normalized_blur",
        "mean_infidelity",
        "mean_completeness_delta",
        "mean_endpoint_error_kl",
    ]
    representative = selected_quality[
        [column for column in table_columns if column in selected_quality]
    ].copy()
    representative.to_csv(
        results_root / "equal_budget_table.csv", index=False
    )
    (results_root / "equal_budget_table.md").write_text(
        representative.to_markdown(index=False) + "\n", encoding="utf-8"
    )

    available_metrics = [
        metric
        for metric in QUALITY_DIRECTIONS
        if f"mean_{metric}" in summary.columns
    ]
    paired = paired_statistics(
        selection,
        summary,
        reference_method=reference_method,
        metrics=available_metrics,
        seed=seed,
        n_resamples=n_resamples,
    )
    paired.to_csv(
        results_root / "equal_budget_paired_bootstrap.csv", index=False
    )

    full_mask = summary["is_full"].map(
        lambda value: str(value).lower() in {"true", "1", "yes"}
    )
    full = summary[full_mask].copy()
    full.to_csv(results_root / "full_method_points.csv", index=False)
    retention = fringe_gain_retention(
        summary,
        anchor_method=anchor_method,
        anchor_candidate=anchor_candidate,
    )
    retention.to_csv(
        results_root / "fringe_gain_retention.csv", index=False
    )

    for metric, direction in QUALITY_DIRECTIONS.items():
        _plot_pareto(summary, results_root, metric, direction)
        if metric == "mas_insertion_blur":
            _plot_memory_quality(
                summary, results_root, metric, direction
            )

    report = [
        "# Equalized runtime, memory, and quality comparison",
        "",
        f"- Runtime anchor: `{anchor_method}/{anchor_candidate}`",
        f"- Budget multiples: {', '.join(f'{value:g}×' for value in budget_multiples)}",
        "- Matching rule: minimum absolute log-distance from the measured median runtime target.",
        "- Secondary matches: peak allocated CUDA memory and recorded gradient evaluations.",
        f"- Paired reference: `{reference_method}` at its selected candidate for each model and budget.",
        f"- Paired uncertainty: {n_resamples:,} bootstrap resamples over identical evaluation images.",
        "",
        "The CSV files are the authoritative machine-readable outputs. Runtime matching is based on measured attribution-only latency; metric evaluation time is excluded.",
    ]
    (results_root / "EQUAL_BUDGET_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    with (results_root / "analysis_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "anchor_method": anchor_method,
                "anchor_candidate": anchor_candidate,
                "budget_multiples": list(budget_multiples),
                "reference_method": reference_method,
                "bootstrap_seed": seed,
                "bootstrap_resamples": n_resamples,
                "matching_distance": "absolute_log_runtime_ratio",
                "models": (
                    sorted(summary["model"].astype(str).unique().tolist())
                ),
            },
            handle,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select measured equal-runtime candidates and compute paired statistics."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "comparisons" / "EqualBudget",
    )
    parser.add_argument("--anchor-method", default="ig")
    parser.add_argument("--anchor-candidate", default="ig_32")
    parser.add_argument(
        "--budget-multiples", nargs="+", type=float, default=[1, 2, 5, 10]
    )
    parser.add_argument("--reference-method", default="fringe")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional subset of models to analyse.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyse(
        results_root=args.results_root.expanduser().resolve(),
        anchor_method=args.anchor_method,
        anchor_candidate=args.anchor_candidate,
        budget_multiples=args.budget_multiples,
        reference_method=args.reference_method,
        seed=args.seed,
        n_resamples=args.bootstrap_resamples,
        models=args.models,
    )
