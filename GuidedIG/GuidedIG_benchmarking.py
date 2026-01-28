import os
import math
import hydra
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from metrics.MetricsWrapper import MetricsWrapper
from utils import load_model, load_image


class GuidedIG:
    """
    PyTorch implementation of Guided Integrated Gradients based on the
    logic provided in the paper's reference implementation.
    """

    def __init__(self, forward_func):
        self.forward_func = forward_func

    def attribute(self, inputs, baselines=None, target=None, n_steps=200,
                  fraction=0.25, max_dist=0.02, **kwargs):
        """
        Args:
            inputs: Input tensor (B, C, H, W) - Currently supports B=1 strictly.
            baselines: Baseline tensor.
            target: Target class index.
            n_steps: Number of Riemann steps.
            fraction: Fraction of features to update per step.
            max_dist: Maximum allowed deviation from the straight line (0-1).
        """
        # Ensure inputs are on the correct device
        # We don't necessarily need gradients on 'inputs' (the target destination),
        # but we do need them on 'x' (the current point).
        inputs = inputs.clone().detach()
        device = inputs.device

        if baselines is None:
            baselines = torch.zeros_like(inputs)

        # Guided IG Logic currently assumes single sample processing
        if inputs.shape[0] != 1:
            raise NotImplementedError("Guided IG implementation currently supports batch_size=1 only.")

        x_input = inputs
        x_baseline = baselines

        # --- FIX 1: Initialize x with requires_grad=True ---
        # We detach to ensure it's a leaf variable, then enable gradients.
        x = x_baseline.clone().detach().requires_grad_(True)

        attr = torch.zeros_like(x_input)

        # Pre-compute total L1 distance
        total_diff = x_input - x_baseline
        l1_total = torch.abs(total_diff).sum()

        if l1_total == 0:
            return attr

        # Helper to translate x to alpha (relative position on straight line)
        def translate_x_to_alpha(x_curr, x_in, x_base):
            diff = x_in - x_base
            # Avoid division by zero
            alpha_map = torch.where(diff != 0, (x_curr - x_base) / diff, torch.tensor(float('nan'), device=device))
            return alpha_map

        # Helper to translate alpha to x
        def translate_alpha_to_x(alpha_val, x_in, x_base):
            return x_base + (x_in - x_base) * alpha_val

        # Integration Loop
        for step in range(n_steps):
            # 1. Calculate Gradients
            # Zero out previous gradients if they exist
            if x.grad is not None:
                x.grad.zero_()

            # Forward pass
            output = self.forward_func(x)

            # Compute gradients w.r.t x
            # This is where it failed previously because x didn't require grad
            grad = torch.autograd.grad(outputs=output, inputs=x)[0]
            grad_actual = grad.clone()

            # --- FIX 2: Use torch.no_grad() for updates ---
            # We update x manually. We don't want these operations recorded in the graph
            # because 'x' acts as a fresh input for the next iteration's gradient.
            with torch.no_grad():
                # 2. Determine Alpha bounds for this step
                alpha_step = (step + 1.0) / n_steps
                alpha_min = max(alpha_step - max_dist, 0.0)
                alpha_max = min(alpha_step + max_dist, 1.0)

                x_min = translate_alpha_to_x(alpha_min, x_input, x_baseline)
                x_max = translate_alpha_to_x(alpha_max, x_input, x_baseline)

                # Target L1 distance for this step
                l1_target = l1_total * (1 - (step + 1) / n_steps)

                # 3. Inner Loop: Select features to move to satisfy L1 constraint
                gamma = float('inf')

                loop_limit = 100
                loop_count = 0

                while gamma > 1.0 and loop_count < loop_limit:
                    loop_count += 1
                    x_old = x.clone()

                    x_alpha = translate_x_to_alpha(x, x_input, x_baseline)
                    x_alpha[torch.isnan(x_alpha)] = alpha_max

                    # Enforce alpha_min constraint
                    mask_behind = x_alpha < alpha_min
                    x[mask_behind] = x_min[mask_behind]

                    l1_current = torch.abs(x - x_input).sum()

                    # Check convergence
                    if torch.isclose(l1_current, l1_target, rtol=1e-9, atol=1e-9):
                        attr += (x - x_old) * grad_actual
                        break

                    # Selection logic
                    grad_for_selection = torch.abs(grad_actual).clone()
                    mask_at_max = x == x_max
                    grad_for_selection[mask_at_max] = float('inf')

                    flat_grads = grad_for_selection.view(-1)

                    if torch.isinf(flat_grads).all():
                        threshold = float('inf')
                    else:
                        valid_grads = flat_grads[flat_grads != float('inf')]
                        if len(valid_grads) == 0:
                            threshold = float('inf')
                        else:
                            threshold = torch.quantile(valid_grads, fraction, interpolation='lower')

                    s_mask = (grad_for_selection <= threshold) & (grad_for_selection != float('inf'))

                    l1_s = torch.abs(x - x_max)[s_mask].sum()

                    if l1_s > 0:
                        gamma = (l1_current - l1_target) / l1_s
                    else:
                        gamma = float('inf')

                    if gamma > 1.0:
                        x[s_mask] = x_max[s_mask]
                    else:
                        target_val = translate_alpha_to_x(gamma, x_max, x)
                        x[s_mask] = target_val[s_mask]

                    # Update attribution
                    attr += (x - x_old) * grad_actual

            # Crucial: After the no_grad block, x still requires_grad=True
            # because we modified it in-place (or if we replaced it, we'd need to re-enable).
            # In-place modification of a leaf tensor that requires grad is tricky,
            # but since we are inside no_grad, it updates the data without tracking history.
            # However, we must ensure it remains a leaf node for the NEXT autograd call.
            # PyTorch's in-place update inside no_grad usually preserves the requires_grad flag.

        return attr, torch.tensor(0.0)


# --- Main Pipeline ---

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    images_path = os.listdir("./examples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avail_models = cfg.avail_models

    # Assumes cfg structure has IG params, we might want to ensure fraction/max_dist are there
    # For safety, I'll default them if not in cfg.IG
    ig_cfg = cfg.IG
    fraction = getattr(ig_cfg, 'fraction', 0.25)  # Default from Guided IG paper code
    max_dist = getattr(ig_cfg, 'max_dist', 0.02)  # Default from Guided IG paper code

    for model_name in avail_models:
        print("Loading model: ", model_name)
        all_results_metrics = []
        all_attributions = []

        model, transform = load_model(model_name, device=device)
        model.eval()

        # Guided IG is computationally heavier than standard IG due to the inner search loop.
        # tqdm helps track progress.
        for image_path in tqdm(images_path):
            results_metrics = {}
            results_attributions = {}
            img = load_image(os.path.join("./examples", image_path), transform, device=device)

            with torch.no_grad():
                # Note: argmax gives the class index.
                # Be aware this explains the PREDICTED class, not necessarily the True class.
                target_idx = model(img).argmax(dim=1)

            # Wrapper for the model to return the specific logit
            def model_forward_batch(x):
                logits = model(x)
                return logits[:, target_idx]

            # --- CHANGED: Use GuidedIG instead of IntegratedGradients ---
            guided_ig = GuidedIG(model_forward_batch)

            # The GuidedIG.attribute method mimics the Captum signature roughly
            # to be compatible with manual calls.
            attr_ig, delta = guided_ig.attribute(inputs=img,
                                                 baselines=torch.zeros_like(img),
                                                 n_steps=ig_cfg.n_steps,
                                                 fraction=fraction,
                                                 max_dist=max_dist)

            # --- WARNING: MetricsWrapper Compatibility ---
            # If MetricsWrapper uses `ig_classic.attribute` internally (e.g. for infidelity),
            # passing `guided_ig` works because we implemented `.attribute`.
            metrics = MetricsWrapper(guided_ig, model)

            ############## INSERTION/DELETION AUC
            ins_del_scores = metrics.extract_insertion_deletion_auc(img, attr_ig)
            results_metrics["insertion_auc"] = ins_del_scores["insertion_auc"]
            results_metrics["deletion_auc"] = ins_del_scores["deletion_auc"]

            ############# MAS INSERTION/DELETION
            mas_scores = metrics.extract_mas_score(img, attr_ig)
            results_metrics["mas_insertion_auc"] = mas_scores["insertion"]
            results_metrics["mas_deletion_auc"] = mas_scores["deletion"]

            ############# MAX Sensitivity
            # Note: Max Sensitivity usually requires running attribution multiple times.
            # Guided IG is slow. This step might take a while.
            max_sensitivity_score = metrics.extract_max_sensitivity_score(img, attr_ig,
                                                                          baselines=torch.zeros_like(img),
                                                                          n_steps=ig_cfg.n_steps,
                                                                          # method="riemann_trapezoid", # Guided IG doesn't support methods
                                                                          return_convergence_delta=True)
            results_metrics["max_sensitivity_score"] = max_sensitivity_score.item()

            ################### INFIDELITY
            infidelity_score = metrics.extract_infidelity_score(img, attr_ig, target_idx=target_idx,
                                                                n_perturbations=50,
                                                                noise_scale=0.02)
            results_metrics["infidelity_score"] = infidelity_score.item()

            ####################### SPARSENESS
            sparseness_score = metrics.extract_sparseness_score(attr_ig)
            results_metrics["sparseness_score"] = sparseness_score.item()

            ################################################
            results_metrics["image_path"] = image_path
            # Guided IG implicitly minimizes delta, but our implementation returns dummy 0
            # unless we explicitly track the integration error vs total diff.
            results_metrics["delta"] = delta.item()

            results_attributions["image_path"] = image_path
            # Convert to list for parquet storage
            results_attributions["attr"] = attr_ig.to("cpu").detach().numpy().tolist()

            all_results_metrics.append(results_metrics)
            all_attributions.append(results_attributions)
            del img

        df = pd.DataFrame(all_results_metrics)
        os.makedirs(f"./results/GuidedIG/{model_name}", exist_ok=True)
        df.to_parquet(f"./results/GuidedIG/{model_name}/results_metrics.parquet", index=False, compression="zstd")
        del df
        df = pd.DataFrame(all_attributions)
        df.to_parquet(f"./results/GuidedIG/{model_name}/results_attributions.parquet", index=False, compression="zstd")

        del model


if __name__ == "__main__":
    main()