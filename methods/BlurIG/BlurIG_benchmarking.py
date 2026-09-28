import os
import math
from typing import Optional, Dict, Any, List, Union

import hydra
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

from evaluation.metrics.MetricsWrapper import MetricsWrapper
from common.utils import load_model, load_image


# ---------------------------
# BlurIG Method (Batched Implementation)
# ---------------------------
class BlurIG:
    """
    Blur Integrated Gradients (Batched).
    Integrates gradients along a path from a fully blurred image to the sharp image.
    """

    def __init__(self, model, max_sigma=10.0, grad_step=0.01):
        self.model = model.eval()
        self.max_sigma = max_sigma
        self.grad_step = grad_step
        self.device = next(model.parameters()).device

    def gaussian_blur(self, inputs, sigma):
        """
        Applies Gaussian Blur to a batch of images using torchvision.
        Sigma can be a single float or a tensor broadcastable to inputs.
        """
        if sigma <= 0.0:
            return inputs

        # Kernel size usually 4*sigma + 1 or similar odd number
        k_size = int(math.ceil(4 * sigma)) + 1
        if k_size % 2 == 0: k_size += 1

        return T.GaussianBlur(kernel_size=k_size, sigma=sigma)(inputs)

    def attribute(self, inputs, target_idx, n_steps=100, chunk_size=32, **kwargs):
        """
        Compute BlurIG attribution.
        Path: from sigma=max_sigma (baseline) to sigma=0 (input).

        Gradients are evaluated in batched chunks across path steps (like
        Captum's internal_batch_size for IG) instead of one sequential
        forward/backward call per step: BatchNorm/etc. run in eval mode with
        no cross-sample dependency, so summing the target score over a chunk
        and taking one autograd.grad call yields the same per-sample gradient
        as evaluating each step individually, at a fraction of the kernel
        launches.
        """
        inputs = inputs.to(self.device)
        target_idx = target_idx.to(self.device)
        B, C, H, W = inputs.shape

        # 1. Define Sigmas (Linear schedule from max_sigma down to 0)
        # You can also use logarithmic schedules for scale-space invariance
        sigmas = torch.linspace(self.max_sigma, 0, n_steps, device=self.device).tolist()

        # BlurIG integral: \int (grad_x(x(sigma)) * dx(sigma)/dsigma) dsigma
        # Approximation: sum( grad_x(x_step) * (x_step+1 - x_step) )
        path_imgs = torch.stack(
            [self.gaussian_blur(inputs, sigma).detach() for sigma in sigmas], dim=0
        )  # [T, B, C, H, W], T = n_steps

        target_idx_expanded = target_idx.view(-1, 1) if target_idx.ndim == 1 else target_idx

        # Gradients are only needed at steps [0, n_steps-2] (the sharp image
        # at the final step is never used as a gradient-evaluation point).
        flat_imgs = path_imgs[:-1].reshape((n_steps - 1) * B, C, H, W)
        flat_targets = target_idx_expanded.repeat(n_steps - 1, 1)

        grads = []
        for start in range(0, flat_imgs.shape[0], chunk_size):
            chunk = flat_imgs[start:start + chunk_size].clone().requires_grad_(True)
            chunk_targets = flat_targets[start:start + chunk_size]
            score = self.model(chunk).gather(1, chunk_targets).sum()
            grad = torch.autograd.grad(score, chunk)[0]
            grads.append(grad.detach())
        grads = torch.cat(grads, dim=0).view(n_steps - 1, B, C, H, W)

        dx = path_imgs[1:] - path_imgs[:-1]  # [T-1, B, C, H, W]
        total_attr = (grads * dx).sum(dim=0)

        return total_attr.detach(), {}


# ---------------------------
# Helpers (Identical)
# ---------------------------
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


def _set_nan_where_invalid(vals: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.bool()
    nan = torch.full_like(vals, float("nan"))
    return torch.where(valid_mask, vals, nan)


# ---------------------------
# Main
# ---------------------------
@hydra.main(version_base=None, config_path="../../configs", config_name="BlurIG_config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_cwd = hydra.utils.get_original_cwd()
    examples_dir = os.path.join(orig_cwd, cfg.examples_dir)
    results_root = os.path.join(orig_cwd, cfg.results_root)

    if not os.path.isdir(examples_dir):
        raise RuntimeError(f"examples dir not found: {examples_dir}")

    images_list = sorted([
        f for f in os.listdir(examples_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if len(images_list) == 0:
        raise RuntimeError(f"No images found in: {examples_dir}")

    print(f"device: {device}")
    print(f"examples: {examples_dir} | n_images={len(images_list)}")
    print(f"results:  {results_root}")

    BATCH_SIZE = int(cfg.get("batch_size", 4))
    SEED = int(cfg.get("seed", 42))

    # Save attributions controls
    save_attributions = bool(cfg.get("save_attributions", True))
    attr_save_dtype = str(cfg.get("attr_save_dtype", "float16")).lower()
    if attr_save_dtype not in ["float16", "float32"]:
        raise ValueError("attr_save_dtype must be 'float16' or 'float32'")

    # Metrics Config
    causal_baselines = _as_list(cfg.get("causal_baselines", None), default=["blur", "black", "white"])
    mas_baselines = _as_list(cfg.get("mas_baselines", None), default=[str(cfg.get("mas_baseline", "blur"))])

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

    # BlurIG Params
    blur_max_sigma = float(cfg.get("max_sigma", 10.0))
    blur_n_steps = int(cfg.get("n_steps", 100))
    blur_internal_batch_size = int(cfg.get("internal_batch_size", 32))

    for model_name in cfg.avail_models:
        print(f"\n--- Processing Model: {model_name} (BlurIG) ---")
        model, transform = load_model(model_name, device=device)
        model.eval()

        # Instantiate BlurIG
        attr_method = BlurIG(
            model,
            max_sigma=blur_max_sigma
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

        output_dir = os.path.join(results_root, model_name)
        os.makedirs(output_dir, exist_ok=True)

        attrs_dir = os.path.join(output_dir, "attributions")
        if save_attributions:
            os.makedirs(attrs_dir, exist_ok=True)

        all_metrics_data: List[Dict[str, Any]] = []

        for i in tqdm(range(0, len(images_list), BATCH_SIZE), desc=f"Evaluating {model_name}"):
            batch_filenames = images_list[i: i + BATCH_SIZE]

            img_tensors = torch.cat([
                load_image(os.path.join(examples_dir, f), transform, device=device)
                for f in batch_filenames
            ], dim=0)

            with torch.no_grad():
                targets = model(img_tensors).argmax(dim=1)

            # A) Attributions
            attr_map, _ = attr_method.attribute(
                img_tensors,
                target_idx=targets,
                n_steps=blur_n_steps,
                chunk_size=blur_internal_batch_size,
            )

            # Save attributions (Batched .pt)
            attr_rel_path = None
            if save_attributions:
                if attr_save_dtype == "float16":
                    attr_to_save = attr_map.detach().to("cpu", dtype=torch.float16)
                else:
                    attr_to_save = attr_map.detach().to("cpu", dtype=torch.float32)

                attr_file = f"attrs_batch_{i:06d}.pt"
                attr_path = os.path.join(attrs_dir, attr_file)

                torch.save(
                    {"filenames": batch_filenames, "attr": attr_to_save},
                    attr_path
                )
                attr_rel_path = os.path.join("attributions", attr_file)

            # B1) Insertion/Deletion multi-baseline
            ins_auc_by_base = {}
            del_auc_by_base = {}
            ins_auc_raw_by_base = {}
            del_auc_raw_by_base = {}

            for bmode in causal_baselines:
                res_norm = metrics.causal_scorer.score(
                    img_tensors, attr_map,
                    baseline_mode=bmode,
                    chunk_size=causal_chunk_size,
                    normalize=causal_normalize,
                )
                res_raw = metrics.causal_scorer.score(
                    img_tensors, attr_map,
                    baseline_mode=bmode,
                    chunk_size=causal_chunk_size,
                    normalize=False,
                )
                ins_auc_by_base[bmode] = res_norm["insertion_auc"]
                del_auc_by_base[bmode] = res_norm["deletion_auc"]
                ins_auc_raw_by_base[bmode] = res_raw["insertion_auc"]
                del_auc_raw_by_base[bmode] = res_raw["deletion_auc"]

            ins_auc_mean = _nanmean_stack(list(ins_auc_by_base.values()))
            del_auc_mean = _nanmean_stack(list(del_auc_by_base.values()))
            ins_auc_raw_mean = _nanmean_stack(list(ins_auc_raw_by_base.values()))
            del_auc_raw_mean = _nanmean_stack(list(del_auc_raw_by_base.values()))

            # B2) MAS multi-baseline
            mas_ins_by_base = {}
            mas_del_by_base = {}
            mas_valid_by_base = {}

            old_mas_baseline = getattr(metrics.mas_scorer, "baseline_type", "blur")
            try:
                for mb in mas_baselines:
                    metrics.mas_scorer.baseline_type = mb
                    mas_scores = metrics.extract_mas_score(
                        img_tensors, attr_map,
                        inference_batch_size=mas_batch_size,
                    )
                    mas_ins = mas_scores["insertion"]
                    mas_del = mas_scores["deletion"]
                    valid = mas_scores.get("valid_mask", torch.ones_like(mas_ins, dtype=torch.bool))

                    mas_ins = _set_nan_where_invalid(mas_ins, valid)
                    mas_del = _set_nan_where_invalid(mas_del, valid)

                    mas_ins_by_base[mb] = mas_ins
                    mas_del_by_base[mb] = mas_del
                    mas_valid_by_base[mb] = valid
            finally:
                metrics.mas_scorer.baseline_type = old_mas_baseline

            mas_ins_mean = _nanmean_stack(list(mas_ins_by_base.values())) if len(mas_ins_by_base) else None
            mas_del_mean = _nanmean_stack(list(mas_del_by_base.values())) if len(mas_del_by_base) else None

            # Infidelity
            infidelity_scores = metrics.extract_infidelity_score(
                img_tensors,
                attr_map,
                target_indices=targets,
                n_perturbations=infid_n,
                noise_scale=infid_noise,
                seed=SEED,
                chunk_size=infid_chunk,
            )

            # Sparseness
            sparseness_scores = metrics.extract_sparseness_score(attr_map)

            # C) Rows
            B = img_tensors.shape[0]
            for b in range(B):
                row = {
                    "image_path": batch_filenames[b],
                    "model": model_name,

                    "attr_file": attr_rel_path,
                    "attr_index": int(b),
                    "target_class": int(targets[b].item()),

                    # baseline-robust aggregates
                    "insertion_auc_mean": float(ins_auc_mean[b].item()),
                    "deletion_auc_mean": float(del_auc_mean[b].item()),
                    "insertion_auc_raw_mean": float(ins_auc_raw_mean[b].item()),
                    "deletion_auc_raw_mean": float(del_auc_raw_mean[b].item()),

                    # MAS aggregates
                    "mas_insertion_auc_mean": float(mas_ins_mean[b].item()) if mas_ins_mean is not None else None,
                    "mas_deletion_auc_mean": float(mas_del_mean[b].item()) if mas_del_mean is not None else None,

                    # others
                    "infidelity_score": float(infidelity_scores[b].item()),
                    "sparseness_score": float(sparseness_scores[b].item()),

                    # BlurIG Params
                    "max_sigma": blur_max_sigma,
                    "n_steps": blur_n_steps
                }

                for bmode in causal_baselines:
                    row[f"insertion_auc__{bmode}"] = float(ins_auc_by_base[bmode][b].item())
                    row[f"deletion_auc__{bmode}"] = float(del_auc_by_base[bmode][b].item())
                    row[f"insertion_auc_raw__{bmode}"] = float(ins_auc_raw_by_base[bmode][b].item())
                    row[f"deletion_auc_raw__{bmode}"] = float(del_auc_raw_by_base[bmode][b].item())

                for mb in mas_baselines:
                    row[f"mas_insertion_auc__{mb}"] = float(
                        mas_ins_by_base[mb][b].item()) if mb in mas_ins_by_base else None
                    row[f"mas_deletion_auc__{mb}"] = float(
                        mas_del_by_base[mb][b].item()) if mb in mas_del_by_base else None
                    if mb in mas_valid_by_base:
                        row[f"mas_valid__{mb}"] = bool(mas_valid_by_base[mb][b].item())

                all_metrics_data.append(row)

            del img_tensors, attr_map, targets, infidelity_scores, sparseness_scores
            torch.cuda.empty_cache()

        # Save metrics parquet
        df = pd.DataFrame(all_metrics_data)
        out_path = os.path.join(output_dir, "results_metrics.parquet")
        df.to_parquet(out_path, index=False, compression="zstd")
        print(f"Saved metrics: {out_path}")

        del model, attr_method, metrics
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()