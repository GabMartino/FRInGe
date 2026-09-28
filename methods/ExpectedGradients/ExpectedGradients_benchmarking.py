import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Allow direct execution from ExpectedGradients/ while keeping project imports
# identical to the other benchmark modules.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from tqdm import tqdm

from evaluation.metrics.MetricsWrapper import MetricsWrapper
from common.utils import load_image, load_model


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
DEFAULT_IMAGENET_DIR = "./data/ImageNet"


def _resolve_project_path(path_value: Any) -> str:
    """
    Resolve relative configuration paths against the repository root.

    Hydra's original working directory depends on how the runner is launched.
    Deriving the root from this source file keeps these forms equivalent:

      python -m methods.ExpectedGradients.ExpectedGradients_benchmarking
      python methods/ExpectedGradients/ExpectedGradients_benchmarking.py
      cd methods/ExpectedGradients && python ExpectedGradients_benchmarking.py
    """
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


class ExpectedGradients:
    """
    Expected Gradients implemented as an average of Integrated Gradients
    across references sampled from an empirical baseline distribution.

    For references x' sampled from the baseline pool, this estimates

        E_x' [(x - x') * integral_0^1 grad f(x' + alpha (x - x')) d alpha].

    Captum IntegratedGradients performs the inner numerical integration.
    References are shared across all samples in a call. This makes a fixed
    seed invariant to attribution batch size and gives every image the same
    Monte Carlo baseline set.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        baseline_pool: torch.Tensor,
        num_baselines: int = 8,
        steps_per_baseline: int = 16,
        method: str = "gausslegendre",
        internal_batch_size: Optional[int] = None,
        baseline_batch_size: Optional[int] = None,
        sample_with_replacement: bool = False,
    ):
        if baseline_pool.ndim < 2:
            raise ValueError(
                "baseline_pool must have shape [N, ...] with at least one reference."
            )
        if baseline_pool.shape[0] < 1:
            raise ValueError("baseline_pool must contain at least one reference.")
        if num_baselines < 1:
            raise ValueError("num_baselines must be >= 1.")
        if steps_per_baseline < 1:
            raise ValueError("steps_per_baseline must be >= 1.")
        if not sample_with_replacement and num_baselines > baseline_pool.shape[0]:
            raise ValueError(
                "num_baselines cannot exceed baseline_pool size when "
                "sample_with_replacement=False."
            )

        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.baseline_pool = baseline_pool.detach().to("cpu")
        self.num_baselines = int(num_baselines)
        self.steps_per_baseline = int(steps_per_baseline)
        self.method = str(method)
        self.internal_batch_size = (
            None if internal_batch_size is None else int(internal_batch_size)
        )
        self.baseline_batch_size = (
            self.num_baselines
            if baseline_batch_size is None
            else int(baseline_batch_size)
        )
        if self.baseline_batch_size < 1:
            raise ValueError("baseline_batch_size must be >= 1.")
        self.sample_with_replacement = bool(sample_with_replacement)
        self.ig = IntegratedGradients(self.model)

    def _sample_reference_indices(self, seed: Optional[int]) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(int(seed))

        pool_size = self.baseline_pool.shape[0]
        if self.sample_with_replacement:
            return torch.randint(
                low=0,
                high=pool_size,
                size=(self.num_baselines,),
                generator=generator,
            )

        return torch.randperm(pool_size, generator=generator)[: self.num_baselines]

    def attribute(
        self,
        inputs: torch.Tensor,
        target_idx: torch.Tensor,
        seed: Optional[int] = None,
        baseline_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute Expected Gradients for a batch.

        Args:
            inputs:
                Input tensor [B, ...].
            target_idx:
                Fixed target class per input [B].
            seed:
                Seed used to sample references. With a fixed seed, results are
                invariant to the number of inputs grouped into the batch.
            baseline_indices:
                Optional explicit pool indices. Primarily useful for exact
                reproducibility and tests. Its length must equal num_baselines.

        Returns:
            (attributions, metadata), matching the repository baseline API.
        """
        inputs = inputs.to(self.device)
        target_idx = target_idx.to(self.device, dtype=torch.long)

        if inputs.shape[0] != target_idx.shape[0]:
            raise ValueError(
                f"Batch mismatch: inputs has {inputs.shape[0]} samples but "
                f"target_idx has {target_idx.shape[0]}."
            )
        if tuple(inputs.shape[1:]) != tuple(self.baseline_pool.shape[1:]):
            raise ValueError(
                "Input and baseline shapes differ: "
                f"{tuple(inputs.shape[1:])} vs "
                f"{tuple(self.baseline_pool.shape[1:])}."
            )

        if baseline_indices is None:
            selected_indices = self._sample_reference_indices(seed)
        else:
            selected_indices = torch.as_tensor(
                list(baseline_indices), dtype=torch.long
            )
            if selected_indices.numel() != self.num_baselines:
                raise ValueError(
                    "baseline_indices length must equal num_baselines "
                    f"({self.num_baselines})."
                )
            if (
                selected_indices.min().item() < 0
                or selected_indices.max().item() >= self.baseline_pool.shape[0]
            ):
                raise IndexError("baseline_indices contains an out-of-range index.")

        batch_size = inputs.shape[0]
        attribution_sum = torch.zeros_like(inputs)

        for start in range(0, self.num_baselines, self.baseline_batch_size):
            chunk_indices = selected_indices[
                start : start + self.baseline_batch_size
            ]
            references = self.baseline_pool.index_select(
                0, chunk_indices
            ).to(self.device, dtype=inputs.dtype)
            references_in_chunk = references.shape[0]

            # Pair every input with every reference in this chunk:
            # [B, K, ...] -> [B*K, ...].
            expanded_inputs = (
                inputs.unsqueeze(1)
                .expand(batch_size, references_in_chunk, *inputs.shape[1:])
                .reshape(
                    batch_size * references_in_chunk,
                    *inputs.shape[1:],
                )
            )
            expanded_references = (
                references.unsqueeze(0)
                .expand(batch_size, references_in_chunk, *references.shape[1:])
                .reshape(
                    batch_size * references_in_chunk,
                    *references.shape[1:],
                )
            )
            expanded_targets = (
                target_idx.unsqueeze(1)
                .expand(batch_size, references_in_chunk)
                .reshape(-1)
            )

            chunk_attributions = self.ig.attribute(
                expanded_inputs,
                baselines=expanded_references,
                target=expanded_targets,
                n_steps=self.steps_per_baseline,
                method=self.method,
                internal_batch_size=self.internal_batch_size,
            )
            chunk_attributions = chunk_attributions.reshape(
                batch_size,
                references_in_chunk,
                *inputs.shape[1:],
            )
            attribution_sum += chunk_attributions.sum(dim=1)

        attributions = attribution_sum / float(self.num_baselines)
        metadata = {
            "baseline_indices": selected_indices.tolist(),
            "num_baselines": self.num_baselines,
            "steps_per_baseline": self.steps_per_baseline,
            "total_integration_steps": (
                self.num_baselines * self.steps_per_baseline
            ),
            "sample_with_replacement": self.sample_with_replacement,
        }
        return attributions.detach(), metadata


def _as_list(x: Any, default: List[str]) -> List[str]:
    if x is None:
        return default
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    try:
        from omegaconf import ListConfig

        if isinstance(x, ListConfig):
            return [str(v) for v in list(x)]
    except Exception:
        pass
    return [str(x)]


def _nanmean_stack(xs: List[torch.Tensor]) -> torch.Tensor:
    if len(xs) == 0:
        raise ValueError("No tensors to nanmean.")
    return torch.nanmean(torch.stack(xs, dim=0), dim=0)


def _set_nan_where_invalid(
    vals: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.bool()
    nan = torch.full_like(vals, float("nan"))
    return torch.where(valid_mask, vals, nan)


def _list_image_files(directory: str) -> List[str]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    paths = sorted(
        p.resolve()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No image files found in: {directory}")
    return [str(p) for p in paths]


def _select_baseline_paths(
    all_paths: Sequence[str],
    pool_size: Optional[int],
    seed: int,
) -> List[str]:
    paths = list(all_paths)
    if pool_size is None or int(pool_size) <= 0 or int(pool_size) >= len(paths):
        return paths

    rng = random.Random(int(seed))
    selected = rng.sample(paths, k=int(pool_size))
    return sorted(selected)


def _load_baseline_pool(
    baseline_paths: Sequence[str],
    transform,
) -> torch.Tensor:
    references = [
        load_image(path, transform, device="cpu")
        for path in tqdm(
            baseline_paths,
            desc="Loading Expected Gradients baseline pool",
            leave=False,
        )
    ]
    return torch.cat(references, dim=0)


def _path_set_hash(paths: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(Path(p).resolve()) for p in paths))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_baseline_manifest(
    output_dir: str,
    baselines_dir: str,
    baseline_paths: Sequence[str],
    eval_paths: Sequence[str],
    cfg,
) -> None:
    baseline_set = {str(Path(p).resolve()) for p in baseline_paths}
    eval_set = {str(Path(p).resolve()) for p in eval_paths}
    overlap = sorted(baseline_set & eval_set)

    manifest = {
        "baseline_directory": str(Path(baselines_dir).resolve()),
        "baseline_pool_size": len(baseline_paths),
        "baseline_pool_seed": int(cfg.get("baseline_pool_seed", 42)),
        "baseline_pool_sha256": _path_set_hash(baseline_paths),
        "evaluation_size": len(eval_paths),
        "evaluation_sha256": _path_set_hash(eval_paths),
        "overlap_count": len(overlap),
        "require_disjoint_baselines": bool(
            cfg.get("require_disjoint_baselines", True)
        ),
        "num_baselines_per_attribution": int(cfg.get("num_baselines", 8)),
        "steps_per_baseline": int(cfg.get("steps_per_baseline", 16)),
        "baseline_source": str(
            cfg.get("baseline_source", "image_directory")
        ),
        "target_score": str(cfg.get("target_score", "target_logit")),
        "sampling": "uniform",
        "sample_with_replacement": bool(
            cfg.get("sample_with_replacement", False)
        ),
        "baseline_paths": list(baseline_paths),
    }
    manifest_path = os.path.join(output_dir, "baseline_pool_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="ExpectedGradients_config",
)
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    examples_dir = _resolve_project_path(
        cfg.get("examples_dir", DEFAULT_IMAGENET_DIR)
    )
    baselines_dir = _resolve_project_path(
        cfg.get("baselines_dir", DEFAULT_IMAGENET_DIR)
    )
    results_root = _resolve_project_path(
        cfg.get("results_root", "./results/methods/ExpectedGradients")
    )

    baseline_source = str(
        cfg.get("baseline_source", "image_directory")
    ).lower()
    if baseline_source != "image_directory":
        raise ValueError(
            "Only baseline_source='image_directory' is currently supported."
        )
    target_score = str(cfg.get("target_score", "target_logit")).lower()
    if target_score != "target_logit":
        raise ValueError(
            "ExpectedGradients currently supports target_score='target_logit' "
            "so it remains consistent with the repository benchmark methods."
        )

    eval_paths = _list_image_files(examples_dir)
    max_images = cfg.get("max_images", None)
    if max_images is not None:
        if int(max_images) < 1:
            raise ValueError("max_images must be >= 1 when provided.")
        eval_paths = eval_paths[: int(max_images)]

    all_baseline_paths = _list_image_files(baselines_dir)
    baseline_paths = _select_baseline_paths(
        all_baseline_paths,
        pool_size=cfg.get("baseline_pool_size", None),
        seed=int(cfg.get("baseline_pool_seed", cfg.get("seed", 42))),
    )

    eval_set = {str(Path(p).resolve()) for p in eval_paths}
    baseline_set = {str(Path(p).resolve()) for p in baseline_paths}
    overlap_count = len(eval_set & baseline_set)
    if bool(cfg.get("require_disjoint_baselines", False)) and overlap_count:
        raise RuntimeError(
            "Expected Gradients baseline pool overlaps the evaluation set "
            f"({overlap_count} files). Point baselines_dir to a disjoint "
            "training/reference set or disable require_disjoint_baselines."
        )
    if overlap_count:
        print(
            "WARNING: Expected Gradients baseline pool overlaps the evaluation "
            f"set by {overlap_count} files. This is convenient for local runs "
            "but should be replaced by a disjoint reference set for rebuttal "
            "experiments."
        )

    print(f"device: {device}")
    print(f"examples:  {examples_dir} | n_images={len(eval_paths)}")
    print(
        f"baselines: {baselines_dir} | pool={len(baseline_paths)} | "
        f"overlap={overlap_count}"
    )
    print(f"results:   {results_root}")

    batch_size = int(cfg.get("batch_size", 4))
    seed = int(cfg.get("seed", 42))
    num_baselines = int(cfg.get("num_baselines", 8))
    steps_per_baseline = int(cfg.get("steps_per_baseline", 16))
    resample_per_batch = bool(cfg.get("resample_baselines_per_batch", False))

    save_attributions = bool(cfg.get("save_attributions", True))
    attr_save_dtype = str(cfg.get("attr_save_dtype", "float16")).lower()
    if attr_save_dtype not in {"float16", "float32"}:
        raise ValueError("attr_save_dtype must be 'float16' or 'float32'.")

    causal_baselines = _as_list(
        cfg.get("causal_baselines", None),
        default=["blur", "black", "white"],
    )
    mas_baselines = _as_list(
        cfg.get("mas_baselines", None),
        default=[str(cfg.get("mas_baseline", "blur"))],
    )
    causal_steps = int(cfg.get("causal_steps", 100))
    causal_chunk_size = int(cfg.get("causal_chunk_size", 8))
    causal_normalize = bool(cfg.get("causal_normalize", True))
    mas_step_size = int(cfg.get("mas_step_size", 224))
    mas_klen = int(cfg.get("mas_klen", 15))
    mas_nsig = float(cfg.get("mas_nsig", 3.0))
    mas_batch_size = int(cfg.get("mas_batch_size", 32))
    infid_n = int(cfg.get("infide_n_perturbations", 50))
    infid_noise = float(cfg.get("noise_scale", 0.02))
    infid_chunk = int(cfg.get("infidelity_chunk_size", 10))

    for model_name in cfg.avail_models:
        print(f"\n--- Processing Model (Expected Gradients): {model_name} ---")
        model, transform = load_model(str(model_name), device=device)
        model.eval()

        baseline_pool = _load_baseline_pool(baseline_paths, transform)
        attr_method = ExpectedGradients(
            model=model,
            baseline_pool=baseline_pool,
            num_baselines=num_baselines,
            steps_per_baseline=steps_per_baseline,
            method=str(cfg.get("ig_method", "gausslegendre")),
            internal_batch_size=cfg.get("internal_batch_size", None),
            baseline_batch_size=cfg.get("baseline_batch_size", None),
            sample_with_replacement=bool(
                cfg.get("sample_with_replacement", False)
            ),
        )

        metrics = MetricsWrapper(
            attr_method,
            model,
            device=device,
            causal_steps=causal_steps,
            causal_baseline="blur",
            mas_step_size=mas_step_size,
            mas_baseline=str(cfg.get("mas_baseline", "blur")),
            mas_klen=mas_klen,
            mas_nsig=mas_nsig,
        )

        output_dir = os.path.join(results_root, str(model_name))
        os.makedirs(output_dir, exist_ok=True)
        _write_baseline_manifest(
            output_dir=output_dir,
            baselines_dir=baselines_dir,
            baseline_paths=baseline_paths,
            eval_paths=eval_paths,
            cfg=cfg,
        )

        attrs_dir = os.path.join(output_dir, "attributions")
        if save_attributions:
            os.makedirs(attrs_dir, exist_ok=True)

        all_metrics_data: List[Dict[str, Any]] = []

        for i in tqdm(
            range(0, len(eval_paths), batch_size),
            desc=f"Evaluating {model_name}",
        ):
            batch_paths = eval_paths[i : i + batch_size]
            batch_filenames = [
                os.path.relpath(path, examples_dir) for path in batch_paths
            ]
            img_tensors = torch.cat(
                [
                    load_image(path, transform, device=device)
                    for path in batch_paths
                ],
                dim=0,
            )

            with torch.no_grad():
                targets = model(img_tensors).argmax(dim=1)

            attribution_seed = seed + i if resample_per_batch else seed
            attr_map, attr_metadata = attr_method.attribute(
                img_tensors,
                targets,
                seed=attribution_seed,
            )

            attr_rel_path = None
            if save_attributions:
                save_dtype = (
                    torch.float16
                    if attr_save_dtype == "float16"
                    else torch.float32
                )
                attr_to_save = attr_map.detach().to(
                    "cpu", dtype=save_dtype
                )
                attr_file = f"attrs_batch_{i:06d}.pt"
                attr_path = os.path.join(attrs_dir, attr_file)
                torch.save(
                    {
                        "filenames": batch_filenames,
                        "attr": attr_to_save,
                        "expected_gradients": attr_metadata,
                    },
                    attr_path,
                )
                attr_rel_path = os.path.join("attributions", attr_file)

            ins_auc_by_base = {}
            del_auc_by_base = {}
            ins_auc_raw_by_base = {}
            del_auc_raw_by_base = {}

            for baseline_mode in causal_baselines:
                res = metrics.causal_scorer.score(
                    img_tensors,
                    attr_map,
                    baseline_mode=baseline_mode,
                    chunk_size=causal_chunk_size,
                    normalize=causal_normalize,
                )
                ins_auc_by_base[baseline_mode] = res["insertion_auc"]
                del_auc_by_base[baseline_mode] = res["deletion_auc"]
                ins_auc_raw_by_base[baseline_mode] = res["insertion_auc_raw"]
                del_auc_raw_by_base[baseline_mode] = res["deletion_auc_raw"]

            ins_auc_mean = _nanmean_stack(list(ins_auc_by_base.values()))
            del_auc_mean = _nanmean_stack(list(del_auc_by_base.values()))
            ins_auc_raw_mean = _nanmean_stack(
                list(ins_auc_raw_by_base.values())
            )
            del_auc_raw_mean = _nanmean_stack(
                list(del_auc_raw_by_base.values())
            )

            mas_ins_by_base = {}
            mas_del_by_base = {}
            mas_valid_by_base = {}
            old_mas_baseline = getattr(
                metrics.mas_scorer, "baseline_type", "blur"
            )
            try:
                for mas_baseline in mas_baselines:
                    metrics.mas_scorer.baseline_type = mas_baseline
                    mas_scores = metrics.extract_mas_score(
                        img_tensors,
                        attr_map,
                        inference_batch_size=mas_batch_size,
                    )
                    mas_ins = mas_scores["insertion"]
                    mas_del = mas_scores["deletion"]
                    valid = mas_scores.get(
                        "valid_mask",
                        torch.ones_like(mas_ins, dtype=torch.bool),
                    )
                    mas_ins_by_base[mas_baseline] = _set_nan_where_invalid(
                        mas_ins, valid
                    )
                    mas_del_by_base[mas_baseline] = _set_nan_where_invalid(
                        mas_del, valid
                    )
                    mas_valid_by_base[mas_baseline] = valid
            finally:
                metrics.mas_scorer.baseline_type = old_mas_baseline

            mas_ins_mean = (
                _nanmean_stack(list(mas_ins_by_base.values()))
                if mas_ins_by_base
                else None
            )
            mas_del_mean = (
                _nanmean_stack(list(mas_del_by_base.values()))
                if mas_del_by_base
                else None
            )

            infidelity_scores = metrics.extract_infidelity_score(
                img_tensors,
                attr_map,
                target_indices=targets,
                n_perturbations=infid_n,
                noise_scale=infid_noise,
                seed=seed,
                chunk_size=infid_chunk,
            )
            sparseness_scores = metrics.extract_sparseness_score(attr_map)

            for batch_index in range(img_tensors.shape[0]):
                row = {
                    "image_path": batch_filenames[batch_index],
                    "model": str(model_name),
                    "attr_file": attr_rel_path,
                    "attr_index": int(batch_index),
                    "target_class": int(targets[batch_index].item()),
                    "insertion_auc_mean": float(
                        ins_auc_mean[batch_index].item()
                    ),
                    "deletion_auc_mean": float(
                        del_auc_mean[batch_index].item()
                    ),
                    "insertion_auc_raw_mean": float(
                        ins_auc_raw_mean[batch_index].item()
                    ),
                    "deletion_auc_raw_mean": float(
                        del_auc_raw_mean[batch_index].item()
                    ),
                    "mas_insertion_auc_mean": (
                        float(mas_ins_mean[batch_index].item())
                        if mas_ins_mean is not None
                        else None
                    ),
                    "mas_deletion_auc_mean": (
                        float(mas_del_mean[batch_index].item())
                        if mas_del_mean is not None
                        else None
                    ),
                    "infidelity_score": float(
                        infidelity_scores[batch_index].item()
                    ),
                    "sparseness_score": float(
                        sparseness_scores[batch_index].item()
                    ),
                    "num_baselines": num_baselines,
                    "steps_per_baseline": steps_per_baseline,
                    "total_integration_steps": (
                        num_baselines * steps_per_baseline
                    ),
                    "baseline_pool_size": int(baseline_pool.shape[0]),
                    "baseline_source": baseline_source,
                    "baseline_sampling": "uniform",
                    "baseline_sample_seed": int(attribution_seed),
                    "target_score": target_score,
                }

                for baseline_mode in causal_baselines:
                    row[f"insertion_auc__{baseline_mode}"] = float(
                        ins_auc_by_base[baseline_mode][batch_index].item()
                    )
                    row[f"deletion_auc__{baseline_mode}"] = float(
                        del_auc_by_base[baseline_mode][batch_index].item()
                    )
                    row[f"insertion_auc_raw__{baseline_mode}"] = float(
                        ins_auc_raw_by_base[baseline_mode][batch_index].item()
                    )
                    row[f"deletion_auc_raw__{baseline_mode}"] = float(
                        del_auc_raw_by_base[baseline_mode][batch_index].item()
                    )

                for mas_baseline in mas_baselines:
                    row[f"mas_insertion_auc__{mas_baseline}"] = (
                        float(
                            mas_ins_by_base[mas_baseline][
                                batch_index
                            ].item()
                        )
                        if mas_baseline in mas_ins_by_base
                        else None
                    )
                    row[f"mas_deletion_auc__{mas_baseline}"] = (
                        float(
                            mas_del_by_base[mas_baseline][
                                batch_index
                            ].item()
                        )
                        if mas_baseline in mas_del_by_base
                        else None
                    )
                    if mas_baseline in mas_valid_by_base:
                        row[f"mas_valid__{mas_baseline}"] = bool(
                            mas_valid_by_base[mas_baseline][
                                batch_index
                            ].item()
                        )

                all_metrics_data.append(row)

            model.zero_grad(set_to_none=True)
            del (
                img_tensors,
                attr_map,
                targets,
                infidelity_scores,
                sparseness_scores,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        dataframe = pd.DataFrame(all_metrics_data)
        output_path = os.path.join(output_dir, "results_metrics.parquet")
        dataframe.to_parquet(
            output_path,
            index=False,
            compression="zstd",
        )
        print(f"Saved metrics: {output_path}")

        del model, baseline_pool, attr_method, metrics
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
