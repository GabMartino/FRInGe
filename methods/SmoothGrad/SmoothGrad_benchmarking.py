import os
from typing import Optional, Dict, Any, List, Union

import hydra
import pandas as pd
import torch
from tqdm import tqdm

from evaluation.metrics.MetricsWrapper import MetricsWrapper
from common.utils import load_model, load_image


# ---------------------------
# SmoothGrad (vectorized)
# ---------------------------
class SmoothGrad:
    def __init__(self, model, stdev_spread=0.15, n_samples=25, magnitude=False):
        self.model = model.eval()
        self.stdev_spread = float(stdev_spread)
        self.n_samples = int(n_samples)
        self.magnitude = bool(magnitude)
        self.device = next(model.parameters()).device

    @torch.enable_grad()
    def attribute(self, inputs: torch.Tensor, target_idx: torch.Tensor, seed: Optional[int] = None):
        inputs = inputs.to(self.device)
        target_idx = target_idx.to(self.device)

        B, C, H, W = inputs.shape
        N = self.n_samples

        x_max = inputs.amax(dim=(1, 2, 3), keepdim=True)
        x_min = inputs.amin(dim=(1, 2, 3), keepdim=True)
        noise_std = self.stdev_spread * (x_max - x_min).clamp_min(1e-6)

        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(int(seed))

        x_rep = inputs.repeat_interleave(N, dim=0).detach().requires_grad_(True)  # (B*N,C,H,W)
        t_rep = target_idx.repeat_interleave(N, dim=0)                            # (B*N,)

        noise_std_rep = noise_std.repeat_interleave(N, dim=0)  # (B*N,1,1,1)
        noise = torch.randn(x_rep.shape, generator=gen, device=self.device) * noise_std_rep
        noisy = x_rep + noise

        logits = self.model(noisy)  # (B*N, num_classes)
        scores = logits.gather(1, t_rep.view(-1, 1)).squeeze(1)  # (B*N,)

        grads = torch.autograd.grad(scores.sum(), x_rep, retain_graph=False, create_graph=False)[0]  # (B*N,C,H,W)
        grads = grads.view(B, N, C, H, W)

        if self.magnitude:
            attr = torch.sqrt((grads ** 2).mean(dim=1) + 1e-12)  # (B,C,H,W)
        else:
            attr = grads.mean(dim=1)  # (B,C,H,W)

        return attr.detach(), {}


# ---------------------------
# Helpers
# ---------------------------
def _as_list(x: Any, default: List[str]) -> List[str]:
    if x is None:
        return default

    # Native Python containers
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]

    # OmegaConf containers (Hydra)
    try:
        from omegaconf import ListConfig
        if isinstance(x, ListConfig):
            return [str(v) for v in list(x)]
    except Exception:
        pass

    # Single value (string or other)
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
@hydra.main(version_base=None, config_path="../../configs", config_name="smoothgrad_config")
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
    # "float16" is strongly recommended for disk
    attr_save_dtype = str(cfg.get("attr_save_dtype", "float16")).lower()  # {"float16","float32"}
    if attr_save_dtype not in ["float16", "float32"]:
        raise ValueError("attr_save_dtype must be 'float16' or 'float32'")

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

    for model_name in cfg.avail_models:
        print(f"\n--- Processing Model: {model_name} ---")
        model, transform = load_model(model_name, device=device)
        model.eval()

        smooth_grad_method = SmoothGrad(
            model,
            stdev_spread=float(cfg.get("std", 0.15)),
            n_samples=int(cfg.get("n_samples", 25)),
            magnitude=bool(cfg.get("magnitude", False)),
        )

        metrics = MetricsWrapper(
            smooth_grad_method,
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

        # where we store attribution tensor chunks
        attrs_dir = os.path.join(output_dir, "attributions")
        if save_attributions:
            os.makedirs(attrs_dir, exist_ok=True)

        all_metrics_data: List[Dict[str, Any]] = []

        # ✅ FIX: iterate over ALL images
        for i in tqdm(range(0, len(images_list), BATCH_SIZE), desc=f"Evaluating {model_name}"):
            batch_filenames = images_list[i: i + BATCH_SIZE]

            img_tensors = torch.cat([
                load_image(os.path.join(examples_dir, f), transform, device=device)
                for f in batch_filenames
            ], dim=0)  # (B,C,H,W)

            with torch.no_grad():
                targets = model(img_tensors).argmax(dim=1)  # (B,)

            # A) attributions
            attr_sg, _ = smooth_grad_method.attribute(img_tensors, targets, seed=SEED)

            # ✅ Save attributions per-batch (recommended)
            attr_rel_path = None
            if save_attributions:
                # cast for disk
                if attr_save_dtype == "float16":
                    attr_to_save = attr_sg.detach().to("cpu", dtype=torch.float16)
                else:
                    attr_to_save = attr_sg.detach().to("cpu", dtype=torch.float32)

                # stable chunk filename
                attr_file = f"attrs_batch_{i:06d}.pt"
                attr_path = os.path.join(attrs_dir, attr_file)

                # store filenames so file is self-describing
                torch.save(
                    {"filenames": batch_filenames, "attr": attr_to_save},
                    attr_path
                )
                # store relative path in parquet
                attr_rel_path = os.path.join("attributions", attr_file)

            # B1) Insertion/Deletion multi-baseline (raw+norm)
            ins_auc_by_base = {}
            del_auc_by_base = {}
            ins_auc_raw_by_base = {}
            del_auc_raw_by_base = {}

            for bmode in causal_baselines:
                res_norm = metrics.causal_scorer.score(
                    img_tensors, attr_sg,
                    baseline_mode=bmode,
                    chunk_size=causal_chunk_size,
                    normalize=causal_normalize,
                )
                res_raw = metrics.causal_scorer.score(
                    img_tensors, attr_sg,
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
                        img_tensors, attr_sg,
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
                attr_sg,
                target_indices=targets,
                n_perturbations=infid_n,
                noise_scale=infid_noise,
                seed=SEED,
                chunk_size=infid_chunk,
            )

            # Sparseness
            sparseness_scores = metrics.extract_sparseness_score(attr_sg)

            # C) rows
            B = img_tensors.shape[0]
            for b in range(B):
                row = {
                    "image_path": batch_filenames[b],
                    "model": model_name,

                    # attribution pointer
                    "attr_file": attr_rel_path,          # e.g. "attributions/attrs_batch_000128.pt"
                    "attr_index": int(b),                # index inside that tensor
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
                }

                for bmode in causal_baselines:
                    row[f"insertion_auc__{bmode}"] = float(ins_auc_by_base[bmode][b].item())
                    row[f"deletion_auc__{bmode}"] = float(del_auc_by_base[bmode][b].item())
                    row[f"insertion_auc_raw__{bmode}"] = float(ins_auc_raw_by_base[bmode][b].item())
                    row[f"deletion_auc_raw__{bmode}"] = float(del_auc_raw_by_base[bmode][b].item())

                for mb in mas_baselines:
                    row[f"mas_insertion_auc__{mb}"] = float(mas_ins_by_base[mb][b].item()) if mb in mas_ins_by_base else None
                    row[f"mas_deletion_auc__{mb}"] = float(mas_del_by_base[mb][b].item()) if mb in mas_del_by_base else None
                    if mb in mas_valid_by_base:
                        row[f"mas_valid__{mb}"] = bool(mas_valid_by_base[mb][b].item())

                all_metrics_data.append(row)

            del img_tensors, attr_sg, targets, infidelity_scores, sparseness_scores
            torch.cuda.empty_cache()

        # Save metrics parquet
        df = pd.DataFrame(all_metrics_data)
        out_path = os.path.join(output_dir, "results_metrics.parquet")
        df.to_parquet(out_path, index=False, compression="zstd")
        print(f"Saved metrics: {out_path}")

        del model, smooth_grad_method, metrics
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
