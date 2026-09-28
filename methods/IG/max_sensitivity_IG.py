import os
from typing import Dict, Any, List, Optional

import hydra
import pandas as pd
import torch
from tqdm import tqdm
from captum.attr import IntegratedGradients

from common.utils import load_model, load_image


# ---------------------------
# Integrated Gradients Wrapper
# ---------------------------
class IGWrapper:
    def __init__(
        self,
        model: torch.nn.Module,
        n_steps: int = 300,
        method: str = "gausslegendre",
        internal_batch_size: Optional[int] = 1,
    ):
        self.model = model.eval()
        self.n_steps = int(n_steps)
        self.method = str(method)
        self.internal_batch_size = internal_batch_size
        self.device = next(model.parameters()).device
        self.ig = IntegratedGradients(self.model)

    def attribute(self, inputs: torch.Tensor, target_idx: torch.Tensor, seed: Optional[int] = None):
        """
        Standardized interface:
            returns (attributions, metadata_dict)
        """
        del seed  # kept only for API compatibility with other methods

        inputs = inputs.to(self.device)
        target_idx = target_idx.to(self.device)

        baselines = torch.zeros_like(inputs)

        attr = self.ig.attribute(
            inputs=inputs,
            baselines=baselines,
            target=target_idx,
            n_steps=self.n_steps,
            method=self.method,
            internal_batch_size=self.internal_batch_size,
            return_convergence_delta=False,
        )

        return attr.detach(), {}


# ---------------------------
# Max-Sensitivity Scorer
# ---------------------------
class MaxSensitivityScorer:
    def __init__(self, attribution_method):
        self.attribution_method = attribution_method

    def score(
        self,
        image: torch.Tensor,
        targets: torch.Tensor,
        original_attr: torch.Tensor,
        radius: float = 0.02,
        n_perturbations: int = 10,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        device = image.device
        B = image.size(0)

        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
        else:
            gen = None

        attr_orig_flat = original_attr.detach().to(device).reshape(B, -1)
        attr_orig_norm = attr_orig_flat / (torch.norm(attr_orig_flat, dim=1, keepdim=True) + 1e-9)

        max_sensitivities = torch.zeros(B, device=device)

        # Sequential perturbation loop for memory safety
        for k in range(n_perturbations):
            if gen is None:
                noise = (torch.rand_like(image) * 2.0 - 1.0) * radius
            else:
                noise = (torch.rand(image.shape, generator=gen, device=device, dtype=image.dtype) * 2.0 - 1.0) * radius

            inputs_noisy = image + noise

            attr_noisy, _ = self.attribution_method.attribute(inputs_noisy, targets, seed=seed)

            attr_noisy_flat = attr_noisy.reshape(B, -1).to(device)
            attr_noisy_norm = attr_noisy_flat / (torch.norm(attr_noisy_flat, dim=1, keepdim=True) + 1e-9)

            diff_norm = torch.norm(attr_orig_norm - attr_noisy_norm, dim=1)
            noise_norm = torch.norm(noise.reshape(B, -1), dim=1)

            sensitivity = diff_norm / (noise_norm + 1e-9)
            max_sensitivities = torch.maximum(max_sensitivities, sensitivity)

        return max_sensitivities


# ---------------------------
# Main Benchmark Loop
# ---------------------------
@hydra.main(version_base=None, config_path="../../configs", config_name="IG_config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_cwd = hydra.utils.get_original_cwd()
    examples_dir = os.path.join(orig_cwd, cfg.examples_dir)
    results_root = os.path.join(orig_cwd, cfg.results_root)

    if not os.path.isdir(examples_dir):
        raise RuntimeError(f"examples dir not found: {examples_dir}")

    images_list = sorted([
        f for f in os.listdir(examples_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".jpeg", ".bmp", ".webp"))
    ])[:64]

    if len(images_list) == 0:
        raise RuntimeError(f"No images found in: {examples_dir}")

    batch_size = int(cfg.get("batch_size", 4))
    seed = int(cfg.get("seed", 42))

    # Max-sensitivity params
    n_perturbations = int(cfg.get("infide_n_perturbations", 10))
    noise_scale = float(cfg.get("noise_scale", 0.02))

    # IG params
    ig_n_steps = int(cfg.get("n_steps", 300))
    ig_method = str(cfg.get("ig_method", "gausslegendre"))
    ig_internal_batch_size = cfg.get("internal_batch_size", 1)

    print(f"device:   {device}")
    print(f"examples: {examples_dir} | n_images={len(images_list)}")
    print(f"results:  {results_root}")
    print(f"IG steps: {ig_n_steps} | method: {ig_method} | internal_batch_size: {ig_internal_batch_size}")

    for model_name in cfg.avail_models:
        print(f"\n--- Processing Model: {model_name} (Captum IG Max-Sensitivity) ---")

        model, transform = load_model(model_name, device=device)
        model.eval()

        attr_method = IGWrapper(
            model=model,
            n_steps=ig_n_steps,
            method=ig_method,
            internal_batch_size=ig_internal_batch_size,
        )
        scorer = MaxSensitivityScorer(attr_method)

        output_dir = os.path.join(results_root, model_name)
        os.makedirs(output_dir, exist_ok=True)

        all_metrics_data: List[Dict[str, Any]] = []

        for i in tqdm(range(0, len(images_list), batch_size), desc=f"Evaluating {model_name}"):
            batch_filenames = images_list[i: i + batch_size]

            img_tensors = torch.cat([
                load_image(os.path.join(examples_dir, f), transform, device=device)
                for f in batch_filenames
            ], dim=0)

            with torch.no_grad():
                targets = model(img_tensors).argmax(dim=1)

            # 1) Base attribution
            attr_map, _ = attr_method.attribute(img_tensors, targets, seed=seed)

            # 2) Max-sensitivity score
            sensitivities = scorer.score(
                image=img_tensors,
                targets=targets,
                original_attr=attr_map,
                radius=noise_scale,
                n_perturbations=n_perturbations,
                seed=seed,
            )

            # 3) Construct rows
            B = img_tensors.shape[0]
            for b in range(B):
                row = {
                    "image_path": batch_filenames[b],
                    "model": model_name,
                    "target_class": int(targets[b].item()),
                    "max_sensitivity_score": float(sensitivities[b].item()),
                    "n_steps": ig_n_steps,
                    "ig_method": ig_method,
                    "internal_batch_size": ig_internal_batch_size,
                    "n_perturbations": n_perturbations,
                    "noise_scale": noise_scale,
                }
                all_metrics_data.append(row)

            model.zero_grad(set_to_none=True)
            del img_tensors, attr_map, targets, sensitivities
            if device.type == "cuda":
                torch.cuda.empty_cache()

        df = pd.DataFrame(all_metrics_data)
        out_path = os.path.join(output_dir, "results_max_sensitivity.parquet")
        df.to_parquet(out_path, index=False, compression="zstd")
        print(f"Saved metrics: {out_path}")

        del model, attr_method, scorer
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()