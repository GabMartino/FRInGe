import torch


class SparsenessScorer:
    """
    Computes the Gini Index for a batch of attribution maps.

    Range: [0, 1]
    - 0: Perfectly uniform (High entropy/Noisy)
    - 1: Perfectly sparse (Concentrated on a single pixel)
    """

    @torch.no_grad()
    def score(self, attributions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attributions: (B, C, H, W) or (B, H, W) tensor.
        Returns:
            torch.Tensor: Gini indices of shape (B,).
        """
        # 1. Process batch dimensions
        B = attributions.shape[0]
        device = attributions.device

        # Flatten and take absolute values: [B, D] where D = C*H*W
        attr = torch.abs(attributions).reshape(B, -1)

        # 2. Sort values per image (Ascending)
        # Gini index calculation requires sorted data
        attr, _ = torch.sort(attr, dim=1)

        n = attr.shape[1]

        # 3. Compute Gini Index in a vectorized manner
        # Formula: G = (2 * sum(i * x_i) / (n * sum(x_i))) - (n + 1) / n

        # Indices i from 1 to n: [1, 2, ..., n]
        indices = torch.arange(1, n + 1, device=device).float()

        # sum(i * x_i) for each image in batch
        weighted_sum = torch.sum(indices * attr, dim=1)

        # total_sum (sum of x_i) for each image
        total_sum = torch.sum(attr, dim=1).clamp(min=1e-9)

        # Final calculation
        gini = (2 * weighted_sum) / (n * total_sum) - (n + 1) / n

        # Due to floating point precision or near-zero maps, clip to [0, 1]
        return torch.clamp(gini, 0.0, 1.0)