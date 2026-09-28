import os
import random
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
# IG2 Method (Provided)
# ---------------------------
class IG2:
    """
    IG2: Integrated Gradients on Iterative Gradient path.
    """

    def __init__(self, model, step_size=0.05, n_steps=50, layer_name=None):
        self.model = model.eval()
        self.step_size = step_size
        self.n_steps = n_steps
        self.device = next(model.parameters()).device
        self.representation = None

        # Hook for representation layer (Feature Extraction)
        if layer_name is None:
            layer = self._auto_find_layer(model)
        else:
            layer = dict([*model.named_modules()])[layer_name]

        self.hook_handle = layer.register_forward_hook(self._hook_fn)

    def _auto_find_layer(self, model):
        """Attempts to find the penultimate layer for representation distance."""
        if hasattr(model, 'avgpool'): return model.avgpool
        if hasattr(model, 'classifier'): return model.classifier[-2]
        # Fallback: Last Conv2d layer
        for name, module in list(model.named_modules())[::-1]:
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise ValueError("Could not auto-detect representation layer. Pass `layer_name`.")

    def _hook_fn(self, module, input, output):
        self.representation = output.flatten(1)

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()

    def _normalize_grad(self, grad, p=2, epsilon=1e-12):
        """Normalize gradients for the iterative path search (L2 norm)."""
        flat_grad = grad.view(grad.size(0), -1)
        norm = torch.norm(flat_grad, p=p, dim=1).view(-1, 1, 1, 1)
        return grad / (norm + epsilon)

    def get_grad_path(self, inputs, references):
        """
        Iteratively search for the path from input to reference.
        Minimizes Euclidean distance in Representation Space.
        """
        B = inputs.size(0)
        N = references.size(0)

        # Expand inputs: [Img1, Img1, Img2, Img2] (interleave)
        current_x = inputs.repeat_interleave(N, dim=0).clone().detach()
        current_x.requires_grad = True

        # Expand references: [Ref1, Ref2, Ref1, Ref2] (repeat)
        refs_expanded = references.repeat(B, 1, 1, 1)

        # 1. Get Reference Representation
        with torch.no_grad():
            _ = self.model(refs_expanded)
            ref_rep = self.representation.detach().clone()

        path = [current_x.detach().clone()]

        # 2. Iterative Update
        for _ in range(self.n_steps):
            _ = self.model(current_x)
            curr_rep = self.representation

            loss = F.mse_loss(curr_rep, ref_rep, reduction='sum')
            grad = torch.autograd.grad(loss, current_x)[0]
            norm_grad = self._normalize_grad(grad)

            current_x = current_x - (norm_grad * self.step_size)
            path.append(current_x.detach().clone())

            # Re-attach for next step gradient
            current_x = current_x.detach().requires_grad_(True)

        return path

    def attribute(self, inputs, target_idx, references=None, **kwargs):
        inputs = inputs.to(self.device)
        target_idx = target_idx.to(self.device)

        B = inputs.size(0)

        # Default Random References if None
        if references is None:
            references = torch.randn(5, inputs.size(1), inputs.size(2), inputs.size(3), device=self.device) * 0.1
        else:
            references = references.to(self.device)

        N = references.size(0)

        # 1. Generate Path
        path_steps = self.get_grad_path(inputs, references)

        # 2. Integrate Gradients along Path (Reverse: GradCF -> Explicand)
        path_steps = path_steps[::-1]

        total_gradients = 0

        # Expand targets
        if isinstance(target_idx, torch.Tensor):
            targets_expanded = target_idx.repeat_interleave(N, dim=0)

        # Riemann Sum
        for i in range(len(path_steps) - 1):
            x_step = path_steps[i].detach().clone().requires_grad_(True)
            output = self.model(x_step)

            if isinstance(target_idx, torch.Tensor):
                score = output.gather(1, targets_expanded.view(-1, 1)).squeeze()
            else:
                score = output[:, target_idx]

            if score.ndim == 0: score = score.view(1)

            grad = torch.autograd.grad(torch.unbind(score), x_step)[0]
            dx = path_steps[i + 1] - path_steps[i]
            total_gradients += grad * dx

        # 3. Average over references
        total_gradients = total_gradients.view(B, N, inputs.size(1), inputs.size(2), inputs.size(3))
        avg_attribution = total_gradients.mean(dim=1)

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
@hydra.main(version_base=None, config_path="../../configs", config_name="IG2_config")
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
    rng = random.Random(SEED)  # Random generator for reference sampling

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

    # IG2 Specific Config
    ig2_n_references = int(cfg.get("n_references", 5))
    ig2_step_size = float(cfg.get("step_size", 0.05))
    ig2_n_steps = int(cfg.get("n_steps", 50))

    for model_name in cfg.avail_models:
        print(f"\n--- Processing Model: {model_name} (IG2) ---")
        model, transform = load_model(model_name, device=device)
        model.eval()

        # Instantiate IG2
        attr_method = IG2(
            model,
            step_size=ig2_step_size,
            n_steps=ig2_n_steps,
            layer_name=cfg.get("layer_name", None)
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

        # Iterate over images
        for i in tqdm(range(0, len(images_list), BATCH_SIZE), desc=f"Evaluating {model_name}"):
            batch_filenames = images_list[i: i + BATCH_SIZE]

            # 1. Load Batch
            img_tensors = torch.cat([
                load_image(os.path.join(examples_dir, f), transform, device=device)
                for f in batch_filenames
            ], dim=0)

            with torch.no_grad():
                targets = model(img_tensors).argmax(dim=1)

            # 2. Load References (IG2 Specific)
            # Sample random other images from the dataset to act as references
            ref_filenames = rng.sample(images_list, ig2_n_references)
            ref_tensors = torch.cat([
                load_image(os.path.join(examples_dir, f), transform, device=device)
                for f in ref_filenames
            ], dim=0)

            # 3. Attributions (Passing references)
            attr_map, _ = attr_method.attribute(img_tensors, targets, references=ref_tensors)

            # 4. Save Attributions (Batched .pt)
            attr_rel_path = None
            if save_attributions:
                if attr_save_dtype == "float16":
                    attr_to_save = attr_map.detach().to("cpu", dtype=torch.float16)
                else:
                    attr_to_save = attr_map.detach().to("cpu", dtype=torch.float32)

                attr_file = f"attrs_batch_{i:06d}.pt"
                attr_path = os.path.join(attrs_dir, attr_file)

                torch.save(
                    {
                        "filenames": batch_filenames,
                        "attr": attr_to_save,
                        # Optional: store which refs were used for reproducibility
                        "ref_filenames": ref_filenames
                    },
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

                    # IG2 Params log
                    "n_references": ig2_n_references,
                    "step_size": ig2_step_size,
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

        # Cleanup hooks
        try:
            attr_method.remove_hook()
        except:
            pass
        del model, attr_method, metrics
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()