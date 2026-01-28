import os
import hydra
import pandas as pd
import torch
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


# --- 1. Include SmoothGrad Class Definition ---
class SmoothGrad:
    """
    SmoothGrad: Averages gradients over Gaussian noise.
    Paper: SmoothGrad: removing noise by adding noise (2017)
    """

    def __init__(self, model, stdev_spread=0.15, n_samples=25, magnitude=False):
        self.model = model
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude
        self.device = next(model.parameters()).device

    def attribute(self, inputs, target_idx, **kwargs):
        """
        inputs: (B, C, H, W)
        target_idx: int or Tensor
        kwargs: Absorbs 'n_steps' or other params passed by generic scorers.
        """
        x = inputs.to(self.device).requires_grad_(True)

        # Calculate standard deviation for noise based on spread and input range
        # Assuming input is approx [0, 1] or normalized.
        std_tensor = torch.ones_like(x) * self.stdev_spread * (x.max() - x.min())

        total_gradients = torch.zeros_like(x)

        for i in range(self.n_samples):
            # 1. Add Noise
            noise = torch.normal(mean=torch.zeros_like(x), std=std_tensor)
            noisy_input = x + noise

            # 2. Forward Pass
            output = self.model(noisy_input)

            # 3. Score Selection (Robust handling)
            if isinstance(target_idx, torch.Tensor):
                score = output.gather(1, target_idx.view(-1, 1)).squeeze()
            else:
                score = output[:, target_idx]

            # FIX: Handle scalar degeneracy
            if score.ndim == 0:
                score = score.view(1)

            # 4. Backward
            # Explicitly grab gradient w.r.t noisy_input
            grad = torch.autograd.grad(torch.unbind(score), noisy_input, retain_graph=False)[0]

            if self.magnitude:
                total_gradients += (grad * grad)
            else:
                total_gradients += grad

        # 5. Average
        avg_gradient = total_gradients / self.n_samples

        if self.magnitude:
            avg_gradient = torch.sqrt(avg_gradient)

        return avg_gradient.detach(), {}


# --- 2. Main Evaluation Loop ---

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    print(os.getcwd())
    images_path = os.listdir("../examples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avail_models = cfg.avail_models

    # Use a default if 'SmoothGrad' config section is missing
    sg_config = getattr(cfg, "SmoothGrad", {"n_samples": 25, "stdev_spread": 0.15})
    n_samples = sg_config.get("n_samples", 25)

    for model_name in avail_models:
        print(f"Loading model: {model_name} | Method: SmoothGrad")
        all_results_metrics = []
        all_attributions = []

        model, transform = load_model(model_name, device=device)
        model.eval()

        # Instantiate SmoothGrad
        smooth_grad_method = SmoothGrad(model, n_samples=n_samples)

        for image_path in tqdm(images_path):
            results_metrics = {}
            results_attributions = {}
            img = load_image(os.path.join("../examples", image_path), transform, device=device)

            with torch.no_grad():
                target_idx = model(img).argmax(dim=1)

            # --- Attribute ---
            attr_sg, _ = smooth_grad_method.attribute(inputs=img, target_idx=target_idx)

            metrics = MetricsWrapper(smooth_grad_method, model)

            ############## INSERTION/DELETION AUC
            ins_del_scores = metrics.extract_insertion_deletion_auc(img, attr_sg)
            results_metrics["insertion_auc"] = ins_del_scores["insertion_auc"]
            results_metrics["deletion_auc"] = ins_del_scores["deletion_auc"]

            ############# MAS INSERTION/DELETION
            mas_scores = metrics.extract_mas_score(img, attr_sg)
            results_metrics["mas_insertion_auc"] = mas_scores["insertion"]
            results_metrics["mas_deletion_auc"] = mas_scores["deletion"]

            ############# MAX Sensitivity

            # max_sensitivity_score = metrics.extract_max_sensitivity_score(img, attr_sg,
            #                                                               target_idx=target_idx)
            # results_metrics["max_sensitivity_score"] = max_sensitivity_score.item()
            ################### INFIDELITY

            infidelity_score = metrics.extract_infidelity_score(img, attr_sg, target_idx=target_idx,
                                                                n_perturbations=50,
                                                                noise_scale=0.02)
            results_metrics["infidelity_score"] = infidelity_score.item()

            ####################### SPARSENESS

            sparseness_score = metrics.extract_sparseness_score(attr_sg)
            results_metrics["sparseness_score"] = sparseness_score.item()

            ################################################
            results_metrics["image_path"] = image_path
            results_metrics["delta"] = None

            results_attributions["image_path"] = image_path
            results_attributions["attr"] = attr_sg.to("cpu").detach().numpy().tolist()

            all_results_metrics.append(results_metrics)
            all_attributions.append(results_attributions)
            del img

        df = pd.DataFrame(all_results_metrics)
        os.makedirs(f"./results/SmoothGrad/{model_name}", exist_ok=True)
        df.to_parquet(f"./results/SmoothGrad/{model_name}/results_metrics.parquet", index=False, compression="zstd")

        df = pd.DataFrame(all_attributions)
        df.to_parquet(f"./results/SmoothGrad/{model_name}/results_attributions.parquet", index=False,
                      compression="zstd")

        del model


if __name__ == "__main__":
    main()