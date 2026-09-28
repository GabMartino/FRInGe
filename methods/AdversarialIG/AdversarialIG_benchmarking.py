import os
import math
from typing import Optional, Dict, Any, List, Union

import hydra
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from evaluation.metrics.MetricsWrapper import MetricsWrapper
from common.utils import load_model, load_image


# ---------------------------
# AGI Method (Batched & optimized)
# ---------------------------
class AGI:
    """
    Adversarial Gradient Integration (AGI) - Batched.
    Paper: Explainable AI via Adversarial Gradient Integration (Pan et al., 2021)
    """

    def __init__(self, model, epsilon=0.05, max_iter=20, topk=5):
        self.model = model.eval()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.topk = topk
        self.device = next(model.parameters()).device

    def _pgd_path_accumulate(self, inputs, init_preds, target_class_idx):
        """
        Generates an adversarial path to 'target_class_idx' and integrates gradients along it.
        """
        batch_size = inputs.size(0)

        # We accumulate attribution here
        integrated_grads = torch.zeros_like(inputs)

        # Current perturbed image state
        x_curr = inputs.clone().detach()

        # --- FIX: target_class_idx is already a tensor [B,], so we just use it directly ---
        target_tensor = target_class_idx.to(device=self.device, dtype=torch.long)

        # Active mask: Image is active if it hasn't reached the target class decision boundary yet
        # If target == init_pred, it's already "there" (or invalid), so inactive.
        active_mask = (target_tensor != init_preds)

        for _ in range(self.max_iter):
            if not active_mask.any():
                break

            x_curr.requires_grad = True
            output = self.model(x_curr)

            # 1. Check if attack succeeded (crossed boundary)
            current_preds = output.argmax(dim=1)
            has_crossed = (current_preds == target_tensor)

            # Update active mask: turn off those that just crossed
            active_mask = active_mask & (~has_crossed)

            if not active_mask.any():
                break

            # 2. Gradient for ATTACK (Ascent on Target Class)
            # We want to maximize P(target_class)
            target_score = output.gather(1, target_tensor.view(-1, 1)).sum()

            self.model.zero_grad()
            if x_curr.grad is not None: x_curr.grad.zero_()
            target_score.backward(retain_graph=True)
            grad_adv = x_curr.grad.data.detach().clone()

            # 3. Gradient for EXPLANATION (Gradient of Original Class)
            # We integrate the gradient of f_original(x)
            # Note: AGI paper integrates gradient of the class we are explaining (init_preds)
            init_score = output.gather(1, init_preds.view(-1, 1)).sum()

            self.model.zero_grad()
            if x_curr.grad is not None: x_curr.grad.zero_()
            init_score.backward()
            grad_explain = x_curr.grad.data.detach().clone()

            # 4. Take Step (PGD / FGSM style)
            # Move towards target class
            step = self.epsilon * grad_adv.sign()

            x_next = x_curr + step
            # Optional: Clamp to valid image range if needed, usually [0,1] or normalized range
            # Here assuming normalized, so we just take the step.

            # 5. Integrate: contribution = grad_explain * -dx
            # The path is from Input -> Boundary.
            # Integration definition: \int (grad) * dx.
            # dx here is (x_next - x_curr).
            dx = x_next - x_curr

            # Accumulate only for active images
            mask_b = active_mask.view(batch_size, 1, 1, 1).float()

            # AGI formula usually implies summing the gradients along the path.
            # Since we move Input -> Boundary, we are removing features.
            # contribution += grad(x) * dx
            integrated_grads += (grad_explain * -dx) * mask_b

            x_curr = x_next.detach()

        return integrated_grads

    def attribute(self, inputs, target_idx=None, **kwargs):
        """
        Attributes the predictions for the given inputs.
        Note: AGI integrates paths to *multiple* decision boundaries to approximate the shape.
        """
        inputs = inputs.to(self.device)
        if target_idx is None:
            # If not provided, assume we explain the max predicted class
            with torch.no_grad():
                target_idx = self.model(inputs).argmax(dim=1)
        else:
            target_idx = target_idx.to(self.device)

        # 1. Identify Top-K Targets to attack (excluding the true class)
        with torch.no_grad():
            logits = self.model(inputs)
            # Set true class logits to -inf so we don't pick them as targets
            logits_scatter = logits.clone()
            logits_scatter.scatter_(1, target_idx.view(-1, 1), -float('inf'))

            # Pick top-k nearest classes
            _, topk_targets = torch.topk(logits_scatter, k=self.topk, dim=1)  # (B, topk)

        total_attribution = torch.zeros_like(inputs)

        # 2. Integrate path to each target
        # We loop over K, but process the batch in parallel for each K
        for k in range(self.topk):
            current_targets = topk_targets[:, k]  # (B,)
            path_attr = self._pgd_path_accumulate(inputs, target_idx, current_targets)
            total_attribution += path_attr

        # Average over the paths
        avg_attribution = total_attribution / self.topk

        return avg_attribution.detach(), {}


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
@hydra.main(version_base=None, config_path="../../configs", config_name="AdversarialIG_config")
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

    # AGI Specific
    agi_epsilon = float(cfg.get("epsilon", 0.05))
    agi_max_iter = int(cfg.get("max_iter", 20))
    agi_topk = int(cfg.get("topk", 5))

    for model_name in cfg.avail_models:
        print(f"\n--- Processing Model: {model_name} (AGI) ---")
        model, transform = load_model(model_name, device=device)
        model.eval()

        attr_method = AGI(
            model,
            epsilon=agi_epsilon,
            max_iter=agi_max_iter,
            topk=agi_topk
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
            attr_map, _ = attr_method.attribute(img_tensors, targets)

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

                    # AGI Params
                    "epsilon": agi_epsilon,
                    "max_iter": agi_max_iter,
                    "topk": agi_topk
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