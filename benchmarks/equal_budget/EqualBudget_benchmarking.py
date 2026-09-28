from __future__ import annotations

import json
import hashlib
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
from omegaconf import OmegaConf

from benchmarks.equal_budget.core import (
    CandidateSpec,
    QualityEvaluator,
    build_disjoint_split,
    candidate_summary_from_results,
    flatten_candidates,
    json_safe,
    list_image_files,
    load_baseline_pool,
    make_candidate,
    measure_batch_throughput,
    run_candidate,
    system_manifest,
    warmup_candidate,
)
from common.utils import load_model


def _project_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_spec(
    spec: CandidateSpec, model_name: str, model_params: Dict[str, Any]
) -> CandidateSpec:
    if spec.method not in {"fringe", "fringe_binary"}:
        return spec
    if model_name not in model_params:
        raise KeyError(
            f"No FRInGe hyperparameters configured for model {model_name!r}."
        )
    # Candidate values (for example kl_target or cg_max_iters) deliberately
    # override the model's tuned defaults.
    params = {**dict(model_params[model_name]), **spec.params}
    return replace(spec, params=params)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="EqualBudget_config",
)
def main(cfg) -> None:
    device_name = str(cfg.get("device", "auto"))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    dataset_dir = _project_path(cfg.get("dataset_dir", "./data/ImageNet"))
    results_root = _project_path(
        cfg.get("results_root", "./results/comparisons/EqualBudget")
    )
    results_root.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    split_cfg = _plain(cfg.split)
    split = build_disjoint_split(
        list_image_files(dataset_dir),
        baseline_pool_size=int(split_cfg["baseline_pool_size"]),
        warmup_size=int(split_cfg["warmup_size"]),
        evaluation_size=split_cfg.get("evaluation_size"),
        seed=seed,
    )
    split["dataset_directory"] = str(dataset_dir)
    with (results_root / "split_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(json_safe(split), handle, indent=2)

    methods = _plain(cfg.methods)
    specs = flatten_candidates(methods)
    requested = {str(value) for value in cfg.get("only_methods", [])}
    if requested:
        specs = [spec for spec in specs if spec.method in requested]
    if not specs:
        raise ValueError("Method filter removed every candidate.")

    model_params = _plain(cfg.get("fringe_model_params", {}))
    quality_config = _plain(cfg.quality)
    warmup_runs = int(cfg.systems.get("warmup_runs", 10))
    throughput_batch_size = int(
        cfg.systems.get("throughput_batch_size", 4)
    )
    throughput_runs = int(cfg.systems.get("throughput_runs", 3))
    save_attributions = bool(cfg.get("save_attributions", False))
    overwrite = bool(cfg.get("overwrite", False))
    run_manifest = {
        "contract_version": 1,
        "dataset_directory": str(dataset_dir),
        "results_root": str(results_root),
        "seed": seed,
        "target_definition": "original_model_top1_target_logit",
        "timing_scope": "attribution_only_batch_size_1",
        "quality_scope": "outside_attribution_timer",
        "split_manifest": "split_manifest.json",
        "systems": system_manifest(device),
        "models": [str(value) for value in cfg.avail_models],
        "candidate_count": len(specs),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _file_sha256(path)
            for path in [
                PROJECT_ROOT / "benchmarks" / "equal_budget" / "core.py",
                PROJECT_ROOT
                / "benchmarks"
                / "equal_budget"
                / "EqualBudget_benchmarking.py",
                PROJECT_ROOT
                / "methods"
                / "FisherRaoIG"
                / "FisherRaoIntegratedGradientsLogs.py",
                PROJECT_ROOT
                / "methods"
                / "FisherRaoIG"
                / "BinaryFisherRaoIntegratedGradients.py",
                PROJECT_ROOT / "configs" / "EqualBudget_config.yaml",
            ]
        },
    }
    (results_root / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
    with (results_root / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(json_safe(run_manifest), handle, indent=2)

    print(
        f"Equal-budget benchmark | device={device} | "
        f"eval={split['evaluation_size']} | references={split['baseline_pool_size']}"
    )
    print(f"Results: {results_root}")

    for raw_model_name in cfg.avail_models:
        model_name = str(raw_model_name)
        print(f"\n=== {model_name} ===")
        model, transform = load_model(model_name, device=device)
        quality = QualityEvaluator(model, device, quality_config)
        needs_references = any(
            spec.method == "expected_gradients" for spec in specs
        )
        reference_start = time.perf_counter_ns()
        baseline_pool = (
            load_baseline_pool(split["baseline_paths"], transform)
            if needs_references
            else None
        )
        reference_preprocessing_ms = (
            time.perf_counter_ns() - reference_start
        ) / 1e6
        model_output = results_root / model_name
        model_output.mkdir(parents=True, exist_ok=True)
        with (model_output / "model_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                json_safe(
                    {
                        "model": model_name,
                        "reference_pool_loaded": needs_references,
                        "reference_pool_size": (
                            len(split["baseline_paths"])
                            if needs_references
                            else 0
                        ),
                        "reference_preprocessing_time_ms": (
                            reference_preprocessing_ms
                        ),
                    }
                ),
                handle,
                indent=2,
            )

        for raw_spec in specs:
            spec = _model_spec(raw_spec, model_name, model_params)
            output_dir = (
                results_root / model_name / spec.method / spec.candidate_id
            )
            completed = output_dir / "per_image.parquet"
            if completed.exists() and not overwrite:
                print(
                    f"[reuse] {model_name}/{spec.method}/{spec.candidate_id}"
                )
                continue

            print(f"[run] {model_name}/{spec.method}/{spec.candidate_id}")
            initialization_start = time.perf_counter_ns()
            candidate = make_candidate(model, spec, baseline_pool)
            initialization_ms = (
                time.perf_counter_ns() - initialization_start
            ) / 1e6
            warmup_candidate(
                candidate,
                split["warmup_paths"],
                transform,
                device,
                runs=warmup_runs,
                seed=seed - warmup_runs,
            )
            throughput = measure_batch_throughput(
                candidate,
                split["warmup_paths"],
                transform,
                device,
                batch_size=throughput_batch_size,
                runs=throughput_runs,
                seed=seed - 10_000,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "candidate_config.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    json_safe(
                        {
                            "model": model_name,
                            "method": spec.method,
                            "candidate_id": spec.candidate_id,
                            "is_full": spec.is_full,
                            "params": spec.params,
                            "warmup_runs": warmup_runs,
                            "initialization_time_ms": initialization_ms,
                            "throughput": throughput,
                            "evaluation_sha256": split[
                                "evaluation_sha256"
                            ],
                        }
                    ),
                    handle,
                    indent=2,
                )
            with (output_dir / "systems.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    json_safe(
                        {
                            **system_manifest(device),
                            "initialization_time_ms": initialization_ms,
                            "steady_state_batch_size": 1,
                            "warmup_runs": warmup_runs,
                            "batched_throughput": throughput,
                            "graphs_retained_after_attribution": False,
                        }
                    ),
                    handle,
                    indent=2,
                )
            run_candidate(
                candidate=candidate,
                eval_paths=split["evaluation_paths"],
                transform=transform,
                output_dir=output_dir,
                quality=quality,
                device=device,
                seed=seed,
                save_attributions=save_attributions,
            )
        del model, baseline_pool
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Re-index every completed candidate under this shared result root. This
    # makes model/method-partitioned scheduler runs composable.
    summary = candidate_summary_from_results(results_root)
    summary.to_csv(results_root / "candidate_summary.csv", index=False)
    summary.to_parquet(results_root / "candidate_summary.parquet", index=False)
    print(
        "\nCandidate grid complete. Run:\n"
        "  python -m benchmarks.equal_budget.analyse_equal_budget "
        f"--results-root {results_root}"
    )


if __name__ == "__main__":
    main()
