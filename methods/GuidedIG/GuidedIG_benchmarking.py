import os
from typing import Optional, Dict, Any, List, Union

import hydra
import pandas as pd
import torch
from tqdm import tqdm

from evaluation.metrics.MetricsWrapper import MetricsWrapper
from common.utils import load_model, load_image


# ---------------------------
# Guided Integrated Gradients (Method)
# ---------------------------
class GuidedIG:
    """
    Batched Implementation of Guided Integrated Gradients.
    Paper: Guided Integrated Gradients: an Adaptive Path Method for Removing Noise
    (arXiv:2106.09788v1)
    """

    def __init__(self, model, n_steps=50, fraction=0.25, max_dist=0.02):
        self.model = model.eval()
        self.n_steps = n_steps
        self.fraction = fraction
        self.max_dist = max_dist
        self.device = next(model.parameters()).device

    def attribute(self, inputs, target_idx, baselines=None, **kwargs):
        """
        Batched Guided IG Attribution.
        inputs: (B, C, H, W)
        target_idx: (B,)
        """
        inputs = inputs.detach().to(self.device)
        target_idx = target_idx.to(self.device)
        batch_size = inputs.shape[0]

        # Default baseline is black (zeros)
        if baselines is None:
            baselines = torch.zeros_like(inputs)
        elif isinstance(baselines, (int, float)):
            baselines = torch.full_like(inputs, baselines)
        else:
            baselines = baselines.to(self.device)

        # x starts at baseline
        x = baselines.clone().detach().requires_grad_(True)
        attr = torch.zeros_like(inputs)

        # Pre-compute total distance for the straight line path
        total_diff = inputs - baselines
        l1_total = torch.abs(total_diff).flatten(1).sum(1).view(batch_size, 1, 1, 1)
        # Avoid div by zero
        l1_total[l1_total == 0] = 1.0

        # Expand targets for gather
        target_idx_expanded = target_idx.view(-1, 1)

        for step in range(self.n_steps):
            # Capture state at start of step
            x_old = x.clone().detach()

            # 1. Gradient Calculation
            if x.grad is not None:
                x.grad.zero_()

            output = self.model(x)
            target_score = output.gather(1, target_idx_expanded).sum()

            # Compute gradients
            grads = torch.autograd.grad(target_score, x, create_graph=False)[0]

            # 2. Update Step (No Grad) - Determine where to move x next
            with torch.no_grad():
                alpha_step = (step + 1.0) / self.n_steps
                alpha_min = max(alpha_step - self.max_dist, 0.0)
                alpha_max = min(alpha_step + self.max_dist, 1.0)

                target_l1_remaining = l1_total * (1.0 - alpha_step)
                current_l1 = torch.abs(inputs - x).flatten(1).sum(1).view(batch_size, 1, 1, 1)

                # Budget: Amount of L1 distance to remove
                l1_budget = torch.clamp(current_l1 - target_l1_remaining, min=0)

                # -- Constraint 1: Fix lagging features --
                rel_diff = inputs - baselines
                rel_mask = rel_diff.abs() > 1e-9

                curr_alpha = torch.zeros_like(x)
                curr_alpha[rel_mask] = (x[rel_mask] - baselines[rel_mask]) / rel_diff[rel_mask]

                # Force features lagging behind alpha_min to catch up
                mask_lagging = (curr_alpha < alpha_min) & rel_mask
                x_min_bound = baselines + alpha_min * total_diff
                x[mask_lagging] = x_min_bound[mask_lagging]

                # Re-calculate budget after mandatory moves
                current_l1 = torch.abs(inputs - x).flatten(1).sum(1).view(batch_size, 1, 1, 1)
                l1_budget = torch.clamp(current_l1 - target_l1_remaining, min=0)

                # -- Constraint 2: Greedy Selection based on Gradients --
                x_max_bound = baselines + alpha_max * total_diff

                # Distance available to move for each pixel before hitting alpha_max
                dist_to_bound = torch.abs(x_max_bound - x)
                dist_to_bound = torch.where(rel_mask, dist_to_bound, torch.zeros_like(x))

                # Flatten for sorting
                B, C, H, W = x.shape
                flat_dist = dist_to_bound.view(B, -1)
                flat_grads = torch.abs(grads).view(B, -1)

                # Sort features by Gradient Magnitude (Ascending)
                # We move features with smallest gradients first (least impact on score)
                sorted_grads, indices = torch.sort(flat_grads, dim=1)

                # Reorder avail distances
                sorted_dist_avail = torch.gather(flat_dist, 1, indices)
                cumsum_dist = torch.cumsum(sorted_dist_avail, dim=1)

                # Find cut-off
                budget_met_mask = cumsum_dist >= l1_budget.view(B, 1)
                # First index where budget is met
                cut_off_indices = torch.argmax(budget_met_mask.int(), dim=1)

                # Handle case where budget > total available (move everything)
                all_false = ~budget_met_mask.any(dim=1)
                cut_off_indices[all_false] = flat_dist.shape[1] - 1

                # Create Full Move Mask
                range_mat = torch.arange(flat_dist.shape[1], device=self.device).expand(B, -1)
                move_fully_mask_sorted = range_mat < cut_off_indices.unsqueeze(1)

                # Map back to original indices
                move_fully_mask_flat = torch.zeros_like(move_fully_mask_sorted, dtype=torch.bool)
                move_fully_mask_flat.scatter_(1, indices, move_fully_mask_sorted)
                move_fully_mask = move_fully_mask_flat.view(B, C, H, W)

                # Apply Full Moves (Set to max bound)
                x = torch.where(move_fully_mask, x_max_bound, x)

                # Apply Partial Move to the specific cut-off pixel
                # Note: indices for gather must be strictly checked
                valid_cutoff = torch.clamp(cut_off_indices, 0, flat_dist.shape[1] - 1)

                # How much budget was used by full moves?
                # cumsum at index (cutoff - 1)
                used_budget = torch.gather(cumsum_dist, 1, (valid_cutoff.unsqueeze(1) - 1).clamp(min=0))
                used_budget[valid_cutoff == 0] = 0  # If we picked 0th element as cutoff, 0 used before

                remaining = torch.clamp(l1_budget.view(B) - used_budget.view(B), min=0)

                partial_idx = torch.gather(indices, 1, valid_cutoff.unsqueeze(1))

                # Move partial pixel
                x_flat = x.view(B, -1)
                current_vals = torch.gather(x_flat, 1, partial_idx)

                diff_flat = total_diff.view(B, -1)
                signs = torch.gather(torch.sign(diff_flat), 1, partial_idx)

                # Move 'remaining' distance in the correct direction
                new_vals = current_vals + signs * remaining.unsqueeze(1)
                x_flat.scatter_(1, partial_idx, new_vals)
                x = x_flat.view(B, C, H, W)

            # Restore gradients for next iteration
            x = x.detach().requires_grad_(True)

            # 3. Accumulate Attribution (Riemann Sum)
            # dx = (x_new - x_old)
            attr += grads * (x - x_old)

        return attr.detach(), {}


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
@hydra.main(version_base=None, config_path="../../configs", config_name="GuidedIG_config")
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

    # Guided IG specific
    gig_n_steps = int(cfg.get("n_steps", 100))
    gig_fraction = float(cfg.get("fraction", 0.25))
    gig_max_dist = float(cfg.get("max_dist", 0.02))

    for model_name in cfg.avail_models:
        print(f"\n--- Processing Model: {model_name} (Guided IG) ---")
        model, transform = load_model(model_name, device=device)
        model.eval()

        attr_method = GuidedIG(
            model,
            n_steps=gig_n_steps,
            fraction=gig_fraction,
            max_dist=gig_max_dist
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

                    # Guided IG params
                    "n_steps": gig_n_steps,
                    "fraction": gig_fraction,
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