import torch
from typing import Optional


class InfidelityScorer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def score(
            self,
            images: torch.Tensor,  # [B, C, H, W]
            attributions: torch.Tensor,  # [B, C, H, W]
            target_indices: torch.Tensor,  # [B]
            n_perturbations: int = 50,
            noise_scale: float = 0.02,
            seed: int = 42,
            max_examples_per_pass: int = 32  # STRICT memory ceiling
    ) -> torch.Tensor:
        B, C, H, W = images.shape
        images = images.to(self.device)
        # Flatten attributions to [B, D] where D = C*H*W
        attr_flat = attributions.detach().reshape(B, -1).to(self.device)
        target_indices = target_indices.to(self.device)

        # 1. Baseline F(x)
        orig_logits = self.model(images)
        orig_scores = torch.gather(orig_logits, dim=1, index=target_indices.unsqueeze(-1))  # [B, 1]

        # 2. Setup Accumulators
        total_sq_error = torch.zeros(B, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # 3. Double-Loop Chunking: Iterating through images in the batch first
        # This prevents creating (Perturbations * Batch) sized tensors.
        for b_idx in range(B):
            img = images[b_idx:b_idx + 1]  # [1, C, H, W]
            attr = attr_flat[b_idx:b_idx + 1]  # [1, D]
            target = target_indices[b_idx:b_idx + 1]  # [1]
            f_x = orig_scores[b_idx]  # Scalar

            img_sq_error = 0.0

            # Process perturbations for THIS specific image in chunks
            for p_start in range(0, n_perturbations, max_examples_per_pass):
                p_count = min(max_examples_per_pass, n_perturbations - p_start)

                # Generate noise on the fly for this chunk
                # Noise: [p_count, C, H, W]
                noise = torch.randn(p_count, C, H, W, generator=generator, device=self.device) * noise_scale
                noise_flat = noise.reshape(p_count, -1)

                # Perturbed inputs: [p_count, C, H, W]
                perturbed_inputs = img - noise

                # Model Inference
                p_logits = self.model(perturbed_inputs)
                # Get scores for the target class: [p_count, 1]
                p_scores = torch.gather(p_logits, dim=1, index=target.expand(p_count, 1))

                # Math: (I) - (II)
                # (I)  LHS: F(x) - F(x-noise)
                func_diff = f_x - p_scores.squeeze()  # [p_count]

                # (II) RHS: dot_product(noise, attribution)
                # noise_flat: [p_count, D], attr: [1, D]
                est_diff = torch.matmul(noise_flat, attr.t()).squeeze()  # [p_count]

                # Accumulate Squared Error
                img_sq_error += torch.sum((func_diff - est_diff) ** 2)

            total_sq_error[b_idx] = img_sq_error / n_perturbations

        return total_sq_error