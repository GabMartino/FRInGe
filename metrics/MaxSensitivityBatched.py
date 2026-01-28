import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Tuple


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
            attribution_method: An instance of your attribution class.
                                Must have an .attribute(input_tensor, ...) method.
        """
        self.attribution_method = attribution_method

    def score(
            self,
            image: torch.Tensor,
            original_attr: Optional[torch.Tensor] = None,
            radius: float = 0.02,
            n_perturbations: int = 10,
            batch_size: int = 8,  # New param to manage VRAM
            **attr_kwargs
    ) -> torch.Tensor:
        """
        Calculates the score using parallel batch processing.

        Args:
            image: (B, C, H, W) input tensor.
            original_attr: (Optional) Pre-computed attribution for the clean image.
            radius: The L-infinity radius of the noise ball.
            n_perturbations: Number of random noise samples to test.
            batch_size: How many perturbations to process simultaneously on the GPU.
            **attr_kwargs: Arguments to pass to .attribute()

        Returns:
            torch.Tensor: The Max-Sensitivity score per image in batch (shape: [B]).
        """
        device = image.device
        B = image.size(0)

        # 1. Compute/Get Baseline Attribution
        if original_attr is None:
            # Note: We assume attribution returns either Tensor or (Tensor, Info)
            res = self.attribution_method.attribute(image, **attr_kwargs)
            if isinstance(res, tuple):
                original_attr = res[0]
            else:
                original_attr = res

        # Normalize original attribution (Pre-compute to avoid doing it inside loop)
        # Flatten: (B, -1)
        attr_orig_flat = original_attr.detach().to(device).reshape(B, -1)
        # Unit norm: ||Attr||_2 = 1
        attr_orig_norm = attr_orig_flat / (torch.norm(attr_orig_flat, dim=1, keepdim=True) + 1e-9)

        # We need to compare every perturbed version to its specific original image
        # We will store the max sensitivity found so far for each image in the batch
        max_sensitivities = torch.zeros(B, device=device)

        # 2. Prepare Inputs for Batch Processing
        # Total operations needed: B * n_perturbations
        # We process 'batch_size' PERTURBATIONS at a time.
        # Note: If image batch B=4 and batch_size=10, we process 40 images effective.

        total_steps = (n_perturbations + batch_size - 1) // batch_size

        # We repeat the original images to match the chunk size later
        # But to save memory, we generate noise in chunks inside the loop

        pbar = tqdm(range(total_steps), desc="Max-Sensitivity (Parallel)")

        for _ in pbar:
            # Determine current chunk size (might be smaller at the end)
            current_chunk_size = min(batch_size, n_perturbations)
            n_perturbations -= current_chunk_size  # decrement counter
            if current_chunk_size <= 0: break

            # Expand image: (B, C, H, W) -> (B * chunk, C, H, W)
            # interleave ensures: [Img1, Img1, Img2, Img2...]
            inputs_expanded = image.repeat_interleave(current_chunk_size, dim=0)

            # Generate Noise
            noise = (torch.rand_like(inputs_expanded) * 2 - 1) * radius
            inputs_noisy = inputs_expanded + noise

            # 3. Compute Perturbed Attribution (Batch inference)
            res = self.attribution_method.attribute(inputs_noisy, **attr_kwargs)

            if isinstance(res, tuple):
                attr_noisy = res[0]
            else:
                attr_noisy = res

            # 4. Compute Sensitivity Scores Vectorized
            # Reshape to (B, Chunk, -1) to separate batch items from their perturbations
            attr_noisy_flat = attr_noisy.reshape(B, current_chunk_size, -1).to(device)

            # Normalize perturbed attributions
            attr_noisy_norm = attr_noisy_flat / (torch.norm(attr_noisy_flat, dim=2, keepdim=True) + 1e-9)

            # Reshape Original to (B, 1, -1) for broadcasting
            attr_orig_broad = attr_orig_norm.unsqueeze(1)  # (B, 1, Features)

            # Diff Norm: ||Attr_orig - Attr_noisy||
            # Result shape: (B, Chunk)
            diff_norm = torch.norm(attr_orig_broad - attr_noisy_norm, dim=2)

            # Noise Norm: ||Noise||
            noise_flat = noise.reshape(B, current_chunk_size, -1)
            noise_norm = torch.norm(noise_flat, dim=2)

            # Sensitivity for this chunk
            sensitivity_chunk = diff_norm / (noise_norm + 1e-9)

            # 5. Update Max (Reduce over the chunk dimension)
            # max_sens_chunk shape: (B,)
            max_sens_chunk = sensitivity_chunk.max(dim=1).values

            max_sensitivities = torch.max(max_sensitivities, max_sens_chunk)

        return max_sensitivities