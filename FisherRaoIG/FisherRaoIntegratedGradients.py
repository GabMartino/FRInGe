from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms.functional as TF
# Ensure these imports point to your actual file locations
from FisherRaoIG.FR_utils import compute_waypoints, spherical_loss, fisher_mat_vect_product, compute_gradient, \
    preconditionedCG_solve, biharmonic_matvec, estimate_grad_sq_precond, biharmonic_matvec, estimate_grad_sq_precond
from FisherRaoIG.BinaryFisherRaoIntegratedGradients import (
    BinaryFisherRaoIntegratedGradients,
)
from metrics.MetricsWrapper import MetricsWrapper
from utils import load_model, load_image, visualize_attributions


class FisherRaoIntegratedGradients:

    def __init__(self, model: torch.nn.Module,
                 model_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 target_idx: torch.Tensor,
                 cg_max_iters=100,
                 cg_rtol=1e-3,
                 cg_atol=1e-5):
        self.model = model
        self.model_forward = model_forward
        self.target_idx = target_idx
        self.cg_max_iters = cg_max_iters
        self.cg_rtol = cg_rtol
        self.cg_atol = cg_atol
        self.model.eval()

    def compute_softmax(self, x):
        return F.softmax(self.model(x), dim=-1)

    def _gradient_target_logit(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model_forward(x, self.target_idx)
        (grads,) = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))
        return grads.detach()

    def _attribute_binary(
            self,
            x: torch.Tensor,
            kl_target: float,
            delta_euc: float,
            eta_max: float,
            use_sobolev_preconditioner: bool,
            lambda_ratio: float,
            smoothing: bool,
            gamma_step: float,
            gamma_prior: float):
        """Run the same FRInGe setting with target-vs-rest Fisher geometry."""
        binary_method = BinaryFisherRaoIntegratedGradients(
            model=self.model,
            model_forward=self.model_forward,
            target_idx=self.target_idx,
        )
        attributions, stats = binary_method.attribute(
            x=x,
            kl_target=kl_target,
            delta_euc=delta_euc,
            eta_max=eta_max,
            lam=lambda_ratio,
            smoothing=smoothing,
            use_sobolev_preconditioner=use_sobolev_preconditioner,
            blur_kernel_size=5,
            blur_sigma=2.0,
            gamma_step=gamma_step,
            gamma_prior=gamma_prior,
            Ainv_iters=self.cg_max_iters,
            Ainv_rtol=self.cg_rtol,
            Ainv_atol=self.cg_atol,
            safety_buffer=0.1,
            show_progress=True,
        )
        self.last_binary_stats = stats
        mean_delta = float(
            torch.tensor(stats["completeness_delta"], dtype=torch.float64)
            .mean()
            .item()
        )
        print(f"Completeness Delta: {mean_delta:.4f}")
        return attributions, mean_delta

    def attribute(self,
                  x: torch.Tensor,
                  kl_target: float = None,
                  T_desired: int = None,
                  p_target: torch.Tensor = None,
                  fisher=True,
                  binary: bool = False,
                  ### FISHER PARAMETERS
                  delta_euc: float = 10,
                  eta_max: float = 10,
                  use_sobolev_preconditioner: bool = True,
                  lambda_ratio: float = 0.01,
                  ### SMOOTHING PARAMETERS
                  smoothing=False,
                  gamma_step: float = 0.01,  # Controls Solver Stability (LHS)
                  gamma_prior: float = 0.001):  # Controls Final Map Denoising (RHS)

        if kl_target is None and T_desired is None:
            raise ValueError("At least one of the kl_target and T_desired should be set.")

        if binary:
            if not fisher:
                raise ValueError("binary=True requires fisher=True.")
            if kl_target is None:
                raise ValueError(
                    "binary=True currently requires kl_target to define the "
                    "shared waypoint and trust-region budget."
                )
            if p_target is not None:
                raise ValueError(
                    "binary=True uses the collapsed uniform reference and "
                    "does not accept a custom p_target."
                )
            return self._attribute_binary(
                x=x,
                kl_target=kl_target,
                delta_euc=delta_euc,
                eta_max=eta_max,
                use_sobolev_preconditioner=use_sobolev_preconditioner,
                lambda_ratio=lambda_ratio,
                smoothing=smoothing,
                gamma_step=gamma_step,
                gamma_prior=gamma_prior,
            )

        eta_max_tensor = torch.tensor(eta_max, device=x.device, dtype=x.dtype)
        x_prev = x.clone().requires_grad_(True)

        attributions = torch.zeros_like(x)

        with torch.no_grad():
            logit_start = self.model_forward(x, self.target_idx)
            p_start = self.compute_softmax(x_prev)

        C = p_start.shape[-1]
        B = p_start.shape[0]
        s_alphas, mask, n_steps = compute_waypoints(p_start=p_start,
                                                    kl_target=kl_target,
                                                    num_classes=C,
                                                    p_target=p_target,
                                                    T_desired=T_desired)

        max_T = int(n_steps.max().item())
        v = None

        # >>> NEW: compute grad_prev ONCE, then roll it forward
        grad_prev = self._gradient_target_logit(x_prev)

        # Waypoint zero is the starting prediction.  State x_k must target the
        # next waypoint, so iterate over indices 1, ..., max_T - 1.
        for i in tqdm(range(1, max_T), desc="FisherRao"):
            s_alpha = s_alphas[:, i, :]
            active_mask = mask[:, i]

            # If all samples in batch are done, stop early
            if not active_mask.any():
                break

            current_p = self.compute_softmax(x_prev)

            # >>> FIX: Prevent sqrt(0) NaNs
            current_p_safe = current_p.clamp(min=1e-10)
            current_p_sqrt = current_p_safe.sqrt()

            # 1. Compute Loss and Gradient
            spher_loss = spherical_loss(current_p_sqrt, s_alpha)
            grad_spher_loss = compute_gradient(spher_loss, x_prev, retain_graph=False)

            if fisher:
                with torch.no_grad():
                    _, vjp_fn = torch.func.vjp(lambda inp: self.model(inp), x_prev)

                def G_mv(v_in):
                    return fisher_mat_vect_product(x=x_prev,
                                                   p=current_p_safe.detach(),
                                                   v=v_in,
                                                   f=self.model,
                                                   vjp_fn=vjp_fn)

                # Define the Linear System LHS (A * v)
                def A_fn(v_in):
                    res = G_mv(v_in) + (lambda_ratio * v_in)
                    if smoothing:
                        res += gamma_step * biharmonic_matvec(v_in)
                    return res

                # >>> FIX: Sobolev-Fisher Preconditioner
                if smoothing and use_sobolev_preconditioner:
                    def M_inv(v_in):
                        # Gaussian blur acts as the inverse Laplacian/Biharmonic operator
                        v_blurred = TF.gaussian_blur(v_in, kernel_size=[5, 5], sigma=[2.0, 2.0])
                        return v_blurred / lambda_ratio
                else:
                    def M_inv(v_in):
                        return v_in / lambda_ratio

                # Define the Linear System RHS (b)
                if smoothing:
                    total_loss = grad_spher_loss + gamma_prior * biharmonic_matvec(x_prev)
                else:
                    total_loss = grad_spher_loss

                extra_dims = [1] * (total_loss.ndim - 1)
                total_loss = total_loss * active_mask.view(B, *extra_dims).float()

                # Solve (G + R)v = g
                v, _, _, _ = preconditionedCG_solve(
                    A_fn=A_fn,
                    b=total_loss,
                    M_op=M_inv,
                    x0=None if v is None else v,
                    max_iter=self.cg_max_iters,
                    rtol=self.cg_rtol,
                    atol=self.cg_atol,
                )

                # Trust Region Step Size Calculation
                with torch.no_grad():
                    fisher_norm_squared = (v * G_mv(v)).reshape(B, -1).sum(-1).clamp_min(1e-12)
                    eta_kl = torch.sqrt(2.0 * kl_target / fisher_norm_squared)
                    eta_euclid = delta_euc / v.reshape(B, -1).norm(dim=-1).clamp_min(1e-12)
                    eta = torch.minimum(eta_max_tensor.expand_as(eta_euclid),
                                        torch.minimum(eta_kl, eta_euclid))
            else:
                v = grad_spher_loss
                eta = eta_max_tensor.expand(B)

            # Update Step
            extra_dims = [1] * (x_prev.ndim - 1)
            mask_b = active_mask.view(B, *extra_dims).float()
            eta_b = eta.view(B, *extra_dims)

            with torch.no_grad():
                step = mask_b * (eta_b * v)
                x_next = x_prev - step

            # >>> CHANGED: trapezoid rule using rolled grad_prev (no recompute at x_prev)
            x_next_req = x_next.detach().requires_grad_(True)
            grad_next = self._gradient_target_logit(x_next_req)
            attributions += 0.5 * (grad_next + grad_prev) * (-step)

            # roll forward
            grad_prev = grad_next
            x_prev = x_next_req

        # Final Sanity Check
        with torch.no_grad():
            logit_end = self.model_forward(x_prev, self.target_idx)

        delta = (attributions.reshape(B, -1).sum(-1) - (logit_end - logit_start)).abs()
        print(f"Completeness Delta: {delta.mean().item():.4f}")

        return -attributions, delta.mean().item()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transformation = load_model("vit_b_16", device=device)

    # Update paths to your actual images
    img_path1 = "../Datasets/ImageNet/n01580077_jay.JPEG"
    img_path2 = "../Datasets/ImageNet/n01443537_goldfish.JPEG"
    img1 = load_image(img_path1, transformation, device=device)
    img2 = load_image(img_path2, transformation, device=device)
    batch = torch.cat([img1, img2], dim=0)

    with torch.no_grad():
        logits = model(batch)
        target_idx = logits.argmax(dim=1)
    print(f"Targets: {target_idx}")


    def model_forward(x, target):
        return model(x).gather(1, target[:, None]).squeeze(1)


    fr_ig = FisherRaoIntegratedGradients(model=model,
                                         model_forward=model_forward,
                                         target_idx=target_idx,
                                         cg_max_iters=20)  # Truncated CG is fine

    print("Running Attribution...")
    attributions = fr_ig.attribute(x=batch,
                                   kl_target=0.1,
                                   fisher=True,
                                   delta_euc=10,
                                   eta_max=2,
                                   use_sobolev_preconditioner=True,

                                   lambda_ratio=0.1,

                                   smoothing=True,
                                   gamma_step=0.1,
                                   gamma_prior=0.01
                                   )

    print("Generating visualizations...")
    visualize_attributions(batch, attributions)

    # Initialize Metrics with BLACK baseline for valid Deletion scores
    metrics = MetricsWrapper(fr_ig, model)

    for i in range(attributions.shape[0]):
        print(f"--- Image {i} ---")
        attr = attributions[i].detach().unsqueeze(0)
        image = batch[i].detach().unsqueeze(0)
        target = target_idx[i].detach().unsqueeze(0)

        # Standard Metrics
        scores = metrics.extract_insertion_deletion_auc(image, attr)
        print(scores)
        print(f"Insertion AUC: {scores['insertion_auc'].cpu().item():.4f}")
        print(f"Deletion AUC:  {scores['deletion_auc'].cpu().item():.4f}")

        # MAS Metrics (Make sure you updated the MASMetric class code!)
        mas_scores = metrics.extract_mas_score(image, attr)
        print(f"MAS Ins AUC:   {mas_scores['insertion'].cpu().item():.4f}")
        print(f"MAS Del AUC:   {mas_scores['deletion'].cpu().item():.4f}")

        # Infidelity
        infidelity_score = metrics.extract_infidelity_score(
            image, attr, target_indices=target
        )
        print(f"Infidelity:    {infidelity_score.item():.6f}")
