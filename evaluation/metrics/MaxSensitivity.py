import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Callable


class MaxSensitivityScorer:
    """
    Computes the Max-Sensitivity metric: measures the maximum change in the
    attribution map for a small change in the input.

    Formula: S(x) = max ||Attr(x) - Attr(x+noise)|| / ||noise||

    Goal: LOWER is better (implies stability).
    """

    def __init__(self, attribution_method):
        """
        Args:
            attribution_method: An instance of your attribution class (e.g. MaxEntropyFR).
                                Must have an .attribute(input_tensor, ...) method.
        """
        self.attribution_method = attribution_method

    def score(
            self,
            image: torch.Tensor,
            original_attr: Optional[torch.Tensor] = None,
            radius: float = 0.02,
            n_perturbations: int = 10,
            **attr_kwargs
    ) -> float:
        """
        Calculates the score.

        Args:
            image: (1, C, H, W) input tensor.
            original_attr: (Optional) Pre-computed attribution for the clean image.
                           If None, it will be computed.
            radius: The L-infinity radius of the noise ball (magnitude of perturbation).
            n_perturbations: Number of random noise samples to test.
            **attr_kwargs: Arguments to pass to .attribute() (e.g. n_steps, c, etc.)

        Returns:
            float: The Max-Sensitivity score (Lower = More Stable).
        """
        device = image.device
        B = image.size(0)

        # 1. Compute/Get Baseline Attribution
        if original_attr is None:
            # Note: We detach/cpu immediately to save VRAM during the loop
            original_attr, _ = self.attribution_method.attribute(image, **attr_kwargs)
        original_attr = original_attr.to(device)
        # Ensure flattened comparison for norms
        attr_orig_flat = original_attr.detach().reshape(B, -1)
        # Normalize attribution to focus on shape/structure, not just magnitude
        # (Standard practice: make it unit norm)
        attr_orig_norm = attr_orig_flat / (torch.norm(attr_orig_flat, dim=-1, keepdim=True) + 1e-9)

        max_sensitivity = torch.zeros(B).to(device)

        print(f"Running Max-Sensitivity (N={n_perturbations}, Radius={radius})...")
        for _ in tqdm(range(n_perturbations)):
            # 2. Generate Perturbation
            # We use uniform noise in the L-infinity ball of size 'radius'
            noise = (torch.rand_like(image) * 2 - 1) * radius
            img_noisy = image + noise.to(device)

            # 3. Compute Perturbed Attribution
            res = self.attribution_method.attribute(img_noisy, **attr_kwargs)
            if len(res) == 1:
                attr_noisy = res
            else:
                attr_noisy, _ = res
            attr_noisy = attr_noisy.to(device)
            # 4. Compare
            attr_noisy_flat = attr_noisy.detach().reshape(B, -1)
            attr_noisy_norm = attr_noisy_flat / (torch.norm(attr_noisy_flat, dim=1, keepdim=True) + 1e-9)

            # Sensitivity = ||Attr_orig - Attr_new|| / ||Noise||
            # We use L2 norm for the attributions
            diff_norm = torch.norm(attr_orig_norm - attr_noisy_norm, dim=1)
            noise_norm = torch.norm(noise.reshape(B, -1), dim=1)
            sensitivity = (diff_norm / (noise_norm + 1e-9))

            (mask,)  = torch.where(sensitivity > max_sensitivity)
            max_sensitivity[mask] = sensitivity[mask]

        return max_sensitivity