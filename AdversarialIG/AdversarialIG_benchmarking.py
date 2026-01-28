import os
import hydra
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from metrics.MetricsWrapper import MetricsWrapper
from utils import load_model, load_image


class AGI:
    """
    Adversarial Gradient Integration (AGI).
    Interprets predictions by integrating gradients along the steepest ascent path
    towards various adversarial examples (discriminating true class from false classes).
    """

    def __init__(self, model, epsilon=0.05, max_iter=20, topk=5):
        self.model = model
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.topk = topk
        self.device = next(model.parameters()).device

    def fgsm_step(self, image, epsilon, data_grad_adv, data_grad_lab):
        """
        One step of ascent.
        Returns:
            perturbed_rect: The new image state (clamped).
            delta: The contribution to the AGI integral for this step.
                   (Gradient * dx) -> approximated as -Grad_True * (x_new - x_old)
        """
        # Ascend towards adversarial class
        delta_img = epsilon * data_grad_adv.sign()

        perturbed_image = image + delta_img
        perturbed_rect = torch.clamp(perturbed_image, min=0, max=1)

        # Calculate actual step taken (dx)
        step_change = perturbed_rect - image

        # Contribution to AGI: -1 * Gradient_True_Class * dx
        # (Eq 8 in paper: We integrate -Grad_True * direction)
        delta_contribution = -data_grad_lab * step_change

        return perturbed_rect, delta_contribution

    def pgd_step(self, image, init_pred, targeted_class):
        """
        Projected Gradient Descent to find path to specific adversarial class.
        """
        perturbed_image = image.clone().detach()
        c_delta = torch.zeros_like(image)  # Cumulative attribution for this path

        for i in range(self.max_iter):
            perturbed_image.requires_grad = True
            output = self.model(perturbed_image)
            output_probs = F.softmax(output, dim=1)

            # 1. Check if we reached the adversarial class (Attack Success)
            pred = output.max(1, keepdim=True)[1]
            if pred.item() == targeted_class.item():
                break

            # 2. Gradient w.r.t. Adversarial Class (Ascent Direction)
            loss_adv = output_probs[0, targeted_class.item()]
            self.model.zero_grad()
            if perturbed_image.grad is not None: perturbed_image.grad.zero_()
            loss_adv.backward(retain_graph=True)
            data_grad_adv = perturbed_image.grad.data.detach().clone()

            # 3. Gradient w.r.t. True Class (Feature Importance)
            loss_lab = output_probs[0, init_pred.item()]
            self.model.zero_grad()
            if perturbed_image.grad is not None: perturbed_image.grad.zero_()
            loss_lab.backward()
            data_grad_lab = perturbed_image.grad.data.detach().clone()

            # 4. Take Step & Accumulate
            perturbed_image, delta = self.fgsm_step(perturbed_image, self.epsilon, data_grad_adv, data_grad_lab)
            c_delta += delta

            # Detach for next iteration
            perturbed_image = perturbed_image.detach()

        return c_delta

    def attribute(self, inputs, target_idx, **kwargs):
        """
        inputs: (B, C, H, W)
        target_idx: int or Tensor
        """
        # FIX: inputs from MaxSensitivity are results of arithmetic (img + noise),
        # so they are not leaves. We detach() to make them a fresh leaf node,
        # then enable gradient tracking.
        inputs = inputs.clone().detach().to(self.device)
        inputs.requires_grad = True

        # Ensure single batch processing for loop logic
        if inputs.size(0) > 1:
            raise NotImplementedError("AGI Batch processing > 1 not optimized in this script.")

        # 1. Get Initial Prediction
        output = self.model(inputs)
        init_pred = output.max(1, keepdim=True)[1]

        # 2. Select Adversarial Classes to Discriminate Against
        num_classes = output.shape[1]
        step = max(1, int(num_classes / self.topk))
        selected_ids = range(0, num_classes, step)

        total_agi = torch.zeros_like(inputs)

        # 3. Iterate over selected adversarial target classes
        for class_id in selected_ids:
            targeted = torch.tensor([class_id]).to(self.device)

            # Skip if target is the true class
            if targeted.item() == init_pred.item():
                continue

            # Compute AGI for this specific discrimination task (True vs Target)
            path_contribution = self.pgd_step(inputs, init_pred, targeted)
            total_agi += path_contribution

        return total_agi.detach(), {}


# --- Main Evaluation Loop ---

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    print(os.getcwd())
    images_path = os.listdir("../examples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avail_models = cfg.avail_models

    for model_name in avail_models:
        print(f"Loading model: {model_name} | Method: AGI")
        model, transform = load_model(model_name, device=device)
        model.eval()

        # Instantiate AGI
        # Tunable parameters: epsilon (step size) and iter (path length)
        # topk determines how many "false classes" we sum over.
        agi_method = AGI(model, epsilon=0.05, max_iter=20, topk=20)

        all_results_metrics = []
        all_attributions = []

        for image_path in tqdm(images_path):
            results_metrics = {}
            results_attributions = {}
            img = load_image(os.path.join("../examples", image_path), transform, device=device)

            with torch.no_grad():
                target_idx = model(img).argmax(dim=1)

            # --- Attribute ---
            attr_agi, _ = agi_method.attribute(inputs=img, target_idx=target_idx)

            # --- Metrics Calculation (Unchanged) ---
            metrics = MetricsWrapper(agi_method, model)

            # INSERTION/DELETION AUC
            ins_del_scores = metrics.extract_insertion_deletion_auc(img, attr_agi)
            results_metrics["insertion_auc"] = ins_del_scores["insertion_auc"]
            results_metrics["deletion_auc"] = ins_del_scores["deletion_auc"]

            # MAS INSERTION/DELETION
            mas_scores = metrics.extract_mas_score(img, attr_agi)
            results_metrics["mas_insertion_auc"] = mas_scores["insertion"]
            results_metrics["mas_deletion_auc"] = mas_scores["deletion"]

            # # MAX Sensitivity
            # max_sensitivity_score = metrics.extract_max_sensitivity_score(
            #     img, attr_agi, target_idx=target_idx)
            # results_metrics["max_sensitivity_score"] = max_sensitivity_score.item()

            # INFIDELITY
            infidelity_score = metrics.extract_infidelity_score(
                img, attr_agi, target_idx=target_idx, n_perturbations=50, noise_scale=0.02
            )
            results_metrics["infidelity_score"] = infidelity_score.item()

            # # SPARSENESS
            # sparseness_score = metrics.extract_sparseness_score(attr_agi)
            # results_metrics["sparseness_score"] = sparseness_score.item()

            ################################################
            results_metrics["image_path"] = image_path
            results_metrics["delta"] = None

            results_attributions["image_path"] = image_path
            results_attributions["attr"] = attr_agi.to("cpu").detach().numpy().tolist()

            all_results_metrics.append(results_metrics)
            all_attributions.append(results_attributions)
            del img

        df = pd.DataFrame(all_results_metrics)
        os.makedirs(f"./results/AGI/{model_name}", exist_ok=True)
        df.to_parquet(f"./results/AGI/{model_name}/results_metrics_top_20.parquet", index=False, compression="zstd")

        df = pd.DataFrame(all_attributions)
        df.to_parquet(f"./results/AGI/{model_name}/results_attributions_top_20.parquet", index=False, compression="zstd")

        del model


if __name__ == "__main__":
    main()