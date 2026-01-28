import os
import hydra
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

# Import your metrics and utils
from metrics.Infidelity import InfidelityScorer
from metrics.InsDelAUC import CausalMetricScorer
from metrics.MaxSensitivity import MaxSensitivityScorer
from metrics.MetricsWrapper import MetricsWrapper
from metrics.Sparseness import SparsenessScorer
from utils import load_model, load_image


class IG2:
    """
    IG2: Integrated Gradients on Iterative Gradient path.
    Paper: IG2: Integrated Gradient on Iterative Gradient (2024)
    """

    def __init__(self, model, step_size=0.05, n_steps=50, layer_name=None):
        self.model = model
        self.step_size = step_size
        self.n_steps = n_steps
        self.device = next(model.parameters()).device
        self.representation = None

        # Hook for representation layer (Feature Extraction)
        # Uses penultimate layer activation as representation [cite: 203]
        if layer_name is None:
            layer = self._auto_find_layer(model)
        else:
            layer = dict([*model.named_modules()])[layer_name]

        layer.register_forward_hook(self._hook_fn)

    def _auto_find_layer(self, model):
        """Attempts to find the penultimate layer for representation distance."""
        if hasattr(model, 'avgpool'): return model.avgpool
        if hasattr(model, 'classifier'): return model.classifier[-2]
        for name, module in list(model.named_modules())[::-1]:
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise ValueError("Could not auto-detect representation layer. Pass `layer_name`.")

    def _hook_fn(self, module, input, output):
        self.representation = output.flatten(1)

    def _normalize_grad(self, grad, p=2, epsilon=1e-12):
        """
        Normalize gradients for the iterative path search (L2 norm)[cite: 1093].
        Corresponds to Eq 20 in the paper.
        """
        flat_grad = grad.view(grad.size(0), -1)
        norm = torch.norm(flat_grad, p=p, dim=1).view(-1, 1, 1, 1)
        return grad / (norm + epsilon)

    def get_grad_path(self, inputs, references):
        """
        Iteratively search for the path from input to reference.
        Minimizes Euclidean distance in Representation Space.
        Generates GradPath and GradCF (Baseline)[cite: 164].
        """
        batch_size = references.size(0)
        # current_x starts at Explicand
        current_x = inputs.repeat(batch_size, 1, 1, 1).clone().detach()
        current_x.requires_grad = True

        # 1. Get Reference Representation
        with torch.no_grad():
            _ = self.model(references)
            ref_rep = self.representation.detach().clone()

        path = [current_x.detach().clone()]

        # 2. Iterative Update (Algorithm 1) [cite: 164]
        for _ in range(self.n_steps):
            _ = self.model(current_x)
            curr_rep = self.representation

            # Optimization objective: minimize representation distance
            # Note: Using MSE here is fine as gradients are normalized.
            loss = F.mse_loss(curr_rep, ref_rep, reduction='sum')

            grad = torch.autograd.grad(loss, current_x)[0]
            norm_grad = self._normalize_grad(grad)

            # Update: Descent direction minimizing distance to reference [cite: 193]
            current_x = current_x - (norm_grad * self.step_size)

            path.append(current_x.detach().clone())
            current_x = current_x.detach().requires_grad_(True)

        return path

    def attribute(self, inputs, target_idx, references=None, **kwargs):
        """
        Compute IG2 attributions.
        inputs: (1, C, H, W)
        references: (B, C, H, W) - Counterfactual images
        """
        if references is None:
            # Fallback (Paper suggests samples from different categories [cite: 755])
            references = torch.randn_like(inputs).repeat(5, 1, 1, 1) * 0.1

        references = references.to(self.device)

        # 1. Generate Path (GradPath)
        # path_steps goes: [Explicand, ..., GradCF]
        path_steps = self.get_grad_path(inputs, references)

        # 2. Integrate Gradients along Path
        # The integration must be from Baseline (GradCF) -> Explicand[cite: 245].
        # We reverse the path to iterate 0 (Baseline) -> 1 (Explicand).
        path_steps = path_steps[::-1]

        total_gradients = 0

        # Loop n_steps times
        for i in tqdm(range(len(path_steps) - 1)):
            x_step = path_steps[i].detach().clone().requires_grad_(True)

            output = self.model(x_step)

            if isinstance(target_idx, torch.Tensor):
                score = output.gather(1, target_idx.view(-1, 1)).squeeze()
            else:
                score = output[:, target_idx]

            if score.ndim == 0: score = score.view(1)

            # Calculate Gradient of explicand's prediction [cite: 242]
            grad = torch.autograd.grad(torch.unbind(score), x_step)[0]

            # Riemann Sum: grad * dx
            # dx points from current step towards Explicand
            dx = path_steps[i + 1] - path_steps[i]
            total_gradients += grad * dx

        # 3. Average over the batch of references (Expected IG2) [cite: 793]
        # "To eliminate the influence of the reference choice... we use the expectation" [cite: 762]
        avg_attribution = total_gradients.mean(dim=0, keepdim=True)

        return avg_attribution.detach(), {}


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    print(os.getcwd())
    images_path = os.listdir("../examples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avail_models = cfg.avail_models

    for model_name in avail_models:
        print(f"Loading model: {model_name} | Method: IG2")
        model, transform = load_model(model_name, device=device)
        model.eval()

        # Instantiate IG2
        ig2_method = IG2(model, step_size=0.05, n_steps=100)

        all_results_metrics = []
        all_attributions = []

        # Loop through images to explain
        for image_path in tqdm(images_path[:1]):
            results_metrics = {}
            results_attributions = {}

            # 1. Load Explicand (Input Image)
            img = load_image(os.path.join("../examples", image_path), transform, device=device)

            # 2. Determine Explicand Class
            with torch.no_grad():
                # We need the class to ensure references are counterfactual (different class)
                target_idx = model(img).argmax(dim=1)

            # [cite_start]3. Dynamic Counterfactual Reference Sampling [cite: 755, 804]
            # Instead of a static list, we resample for every image to ensure class difference.
            ref_images = []

            # Create a pool of potential candidates (all images except the current one)
            candidate_paths = [p for p in images_path if p != image_path]
            # Shuffle to sample randomly
            np.random.shuffle(candidate_paths)

            target_ref_count = 5

            for cand_path in candidate_paths:
                if len(ref_images) >= target_ref_count:
                    break

                try:
                    # Load candidate image
                    r_img = load_image(os.path.join("../examples", cand_path), transform, device=device)

                    # Check candidate class
                    with torch.no_grad():
                        r_class = model(r_img).argmax(dim=1)

                    # Only accept if it belongs to a different class (Counterfactual)
                    if r_class != target_idx:
                        ref_images.append(r_img)

                except Exception as e:
                    # Skip corrupt images or load errors
                    continue

            # Create batch or fallback to noise if selection failed
            if len(ref_images) > 0:
                ref_batch = torch.cat(ref_images, dim=0)
            else:
                print(f"Warning: No valid counterfactuals found for {image_path}. Using Gaussian noise.")
                ref_batch = torch.randn_like(img).repeat(target_ref_count, 1, 1, 1) * 0.1

            # --- Attribute with IG2 ---
            # Pass the dynamically sampled ref_batch
            attr_ig2, _ = ig2_method.attribute(inputs=img,
                                               target_idx=target_idx,
                                               references=ref_batch)

            # --- Metrics Calculation (Unchanged) ---
            metrics = MetricsWrapper(ig2_method, model)

            # INSERTION/DELETION AUC
            ins_del_scores = metrics.extract_insertion_deletion_auc(img, attr_ig2)
            results_metrics["insertion_auc"] = ins_del_scores["insertion_auc"]
            results_metrics["deletion_auc"] = ins_del_scores["deletion_auc"]

            # MAS INSERTION/DELETION
            mas_scores = metrics.extract_mas_score(img, attr_ig2)
            results_metrics["mas_insertion_auc"] = mas_scores["insertion"]
            results_metrics["mas_deletion_auc"] = mas_scores["deletion"]

            # MAX Sensitivity
            max_sensitivity_score = metrics.extract_max_sensitivity_score(
                img, attr_ig2, target_idx=target_idx, references=ref_batch
            )
            results_metrics["max_sensitivity_score"] = max_sensitivity_score.item()

            # INFIDELITY
            infidelity_score = metrics.extract_infidelity_score(
                img, attr_ig2, target_idx=target_idx, n_perturbations=50, noise_scale=0.02
            )
            results_metrics["infidelity_score"] = infidelity_score.item()

            # SPARSENESS
            sparseness_score = metrics.extract_sparseness_score(attr_ig2)
            results_metrics["sparseness_score"] = sparseness_score.item()

            # Save Results
            results_metrics["image_path"] = image_path
            results_metrics["delta"] = None  # Delta calculation omitted in user snippet

            results_attributions["image_path"] = image_path
            results_attributions["attr"] = attr_ig2.to("cpu").detach().numpy().tolist()

            all_results_metrics.append(results_metrics)
            all_attributions.append(results_attributions)
            del img

        # Save to Parquet
        df = pd.DataFrame(all_results_metrics)
        os.makedirs(f"./results/IG2/{model_name}", exist_ok=True)
        df.to_parquet(f"./results/IG2/{model_name}/results_metrics.parquet", index=False, compression="zstd")

        df = pd.DataFrame(all_attributions)
        df.to_parquet(f"./results/IG2/{model_name}/results_attributions.parquet", index=False, compression="zstd")

        del model


if __name__ == "__main__":
    main()

