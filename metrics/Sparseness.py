import numpy as np
import torch


class SparsenessScorer:
    """
    Computes the Gini Index of the attribution map.

    Range: [0, 1]
    - 0: Perfectly uniform (Attribution spread everywhere -> Bad/Noisy)
    - 1: Perfectly sparse (Attribution on 1 pixel -> Very Concise)

    Goal: HIGHER is usually better (indicates a sharp explanation).
    """

    def score(self, attribution: torch.Tensor) -> float:
        """
        Args:
            attribution: (1, C, H, W) tensor.
        """
        # 1. Flatten and take absolute values
        attr = torch.abs(attribution).detach().cpu().numpy().flatten()

        # 2. Sort values (required for Gini calculation)
        attr = np.sort(attr)

        # 3. Compute Gini
        # G = (2 * sum(i * x_i) / (n * sum(x_i))) - (n + 1) / n
        n = len(attr)
        if n == 0: return 0.0

        # Add small epsilon to avoid div by zero if map is all zeros
        total_sum = attr.sum() + 1e-9

        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * attr)) / (n * total_sum) - (n + 1) / n

        return gini