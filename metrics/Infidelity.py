import torch
import numpy as np
from tqdm import tqdm
from typing import Optional


class InfidelityScorer:
    """
    Computes Infidelity: The expected mean-squared error between the
    attribution-predicted change and the actual model output change.

    Formula: E [ ( I^T * Attr - (F(x) - F(x-I)) )^2 ]
    where I is a random perturbation.

    Goal: LOWER is better.
    """

    def __init__(self, model):
        self.model = model

    def score(
            self,
            image: torch.Tensor,
            attribution: torch.Tensor,
            target_idx: int,
            n_perturbations: int = 50,
            noise_scale: float = 0.02
    ) -> float:
        """
        Args:
            image: (1, C, H, W) input tensor.
            attribution: (1, C, H, W) attribution tensor.
            target_idx: The class index to monitor.
            n_perturbations: Number of noise samples.
            noise_scale: Magnitude (std dev) of Gaussian noise.
        """
        device = image.device

        # Flatten attribution for dot product
        attr_flat = attribution.detach().reshape(1, -1).to(device)

        # 1. Get original score F(x)
        with torch.no_grad():
            orig_score = self.model(image)
            orig_score = torch.gather(orig_score, dim=1, index=target_idx.unsqueeze(-1))
        infidelity_sum = 0.0

        print(f"Running Infidelity Check (N={n_perturbations})...")
        for _ in tqdm(range(n_perturbations), desc="Perturbations"):
            # 2. Generate Perturbation I (Gaussian)
            noise = torch.randn_like(image).to(device) * noise_scale
            noise_flat = noise.reshape(1, -1)
            # 3. Compute Function Change: F(x) - F(x - I)
            # (We subtract noise to match the "deletion" intuition, but I can be positive/negative)
            with torch.no_grad():
                perturbed_score = self.model(image - noise)

                perturbed_score = torch.gather(perturbed_score, dim=1, index=target_idx.unsqueeze(-1))


            func_diff = orig_score - perturbed_score
            # 4. Compute Predicted Change: I dot Attr
            est_diff = (noise_flat * attr_flat).sum()

            # 5. Squared Error
            infidelity_sum += (est_diff - func_diff) ** 2

        # Return Expected Value (Mean)
        return (infidelity_sum / n_perturbations)