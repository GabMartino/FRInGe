import random
import os
from functools import partial
from typing import Tuple, Optional, Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sympy.physics.quantum.density import entropy
from tqdm import tqdm

from FisherRaoIG.plot_utils import visualize_geometry, set_icml_matplotlib_style
from metrics.Infidelity import InfidelityScorer
from metrics.MaxSensitivity import MaxSensitivityScorer
from metrics.Sparseness import SparsenessScorer


# Re-use your helper if available, otherwise simple placeholder
# from FisherRaoIG.utils_2 import show_attribution_highlight
# from metrics.InsDelAUC import CausalMetricScorer

def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IntegratedGradients:
    """
    Classical Integrated Gradients (Linear Path) with Full Geometric Analysis.

    Path: Euclidean Linear Interpolation (Baseline -> Input)
    Attribution: Path Integral of Gradient
    Logging: Computes Fisher-Rao metrics passively to compare with Geodesic methods.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            model_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            target_idx: torch.Tensor,
            baseline: Optional[torch.Tensor] = None,
            seed: int = 42,
            use_exact_jvp: bool = False
    ):
        self.model = model
        self.model_forward = model_forward
        self.target_idx = target_idx
        self.baseline = baseline
        self.seed = seed
        self.use_exact_jvp = use_exact_jvp
        self.device = next(model.parameters()).device
        self.model.eval()
        self._reset_rng()

    def _reset_rng(self):
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.seed)

    # =========================================================================
    # Geometry Helper (Copied for Logging Purposes Only)
    # =========================================================================

    def _compute_jvp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Passive JVP computation for logging geometry."""
        if self.use_exact_jvp:
            _, Jv = torch.func.jvp(lambda inp: self.model(inp), (x,), (v,))
            return Jv

        with torch.no_grad():
            B = x.shape[0]
            view_shape = (B,) + (1,) * (x.ndim - 1)

            v_flat = v.reshape(B, -1)
            v_norm = v_flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            v_normalized = v / v_norm.view(*view_shape)

            x_norm = x.reshape(B, -1).norm(dim=1, keepdim=True)
            eps = (1e-3 * x_norm).clamp(1e-5, 1e-1)
            eps_view = eps.reshape(*view_shape)

            f_pos = self.model(x + eps_view * v_normalized)
            f_neg = self.model(x - eps_view * v_normalized)

            # (B, C) / (B, 1)
            J_u = (f_pos - f_neg) / (2.0 * eps.reshape(B, 1))
            return J_u * v_norm

    def _fisher_matvec(self, x: torch.Tensor, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes G(x) @ v for logging."""
        Jv = self._compute_jvp(x, v)
        pJv = p * Jv
        S_Jv = pJv - p * pJv.sum(dim=1, keepdim=True)
        with torch.enable_grad():
            logits = self.model(x)
            (JT_S_Jv,) = torch.autograd.grad(logits, x, grad_outputs=S_Jv, retain_graph=True)
        return torch.nan_to_num(JT_S_Jv)

    def _estimate_fisher_scale(self, x: torch.Tensor, p: torch.Tensor, n_probes: int = 4) -> torch.Tensor:
        """Estimate average eigenvalue (Curvature scale)."""
        B = x.shape[0]
        v_shape = (n_probes, *x.shape)
        v = torch.randn(v_shape, generator=self.rng, device=x.device)

        # FIX: Explicitly detach and require grad because this function is often called
        # inside a torch.no_grad() block (in the analysis loop).
        # We need x_rep to be a leaf node that allows gradient computation locally.
        x_rep = x.repeat(n_probes, 1, 1, 1).detach().requires_grad_(True)

        p_rep = p.repeat(n_probes, 1)
        v_flat = v.reshape(-1, *x.shape[1:])

        # Now this call will succeed because x_rep requires grad
        Gv_flat = self._fisher_matvec(x_rep, p_rep, v_flat)

        v = v.view(n_probes, B, -1)
        Gv = Gv_flat.view(n_probes, B, -1)

        num = (v * Gv).sum(dim=2)
        den = (v * v).sum(dim=2).clamp_min(1e-30)
        return (num / den).mean(dim=0)

    # =========================================================================
    # Main Logic
    # =========================================================================

    def attribute(
            self,
            x: torch.Tensor,
            n_steps: int = 100,
            n_fisher_probes: int = 4,  # Used only for analysis
    ) -> Tuple[torch.Tensor, Dict[str, List[np.ndarray]]]:

        self._reset_rng()
        B = x.shape[0]

        # 1. Define Baseline (Black Image if None)
        baseline = self.baseline if self.baseline is not None else torch.zeros_like(x)

        # 2. Setup Logging
        info = {
            "eta_step": [],  # (Pseudo) Step size
            "fisher_norm_sq": [],  # How much probability changes per Euclidean step
            "lambda_bar": [],  # Local curvature
            "spherical_loss": [],  # Deviation from target distribution (optional)
            "delta_riemann": [],  # The "Cost" of the linear step in manifold terms
            "delta_euclidean": [], # The Euclidean step size (Constant for IG)
            "logit": [],
            "norm_entropy": []
        }

        # 3. Pre-calculate Linear Steps
        # IG Step: dx = (x - x_baseline) / n_steps
        total_diff = x - baseline
        step_vector = total_diff / n_steps

        # We start at baseline, end at x
        x_current = baseline.clone().requires_grad_(True)
        attributions = torch.zeros_like(x)

        # Precompute target direction (optional, for consistency with FR metric)
        # We can check how far the current P(x) is from uniform U
        # just to compare "Spherical Loss" evolution

        for t in tqdm(range(n_steps), desc="Integrated Gradients"):
            # A. Current State
            logits = self.model(x_current)
            p = F.softmax(logits, dim=1)
            # --- GEOMETRIC ANALYSIS (PASSIVE) ---
            with torch.no_grad():
                # 1. Euclidean Step Size (Constant for Linear Path)
                euclidean_dist = step_vector.reshape(B, -1).norm(dim=1)

                info["logit"].append(torch.gather(input=logits, dim=1, index= self.target_idx[:, None]).cpu().squeeze().numpy().item())

                # 2. Fisher Norm Squared along the path direction
                # ||v||_G^2 = v^T G v
                Gv = self._fisher_matvec(x_current, p, step_vector)
                fisher_norm_sq = (step_vector * Gv).reshape(B, -1).sum(dim=1)

                # 3. Riemannian Distance of this step
                # d_R approx sqrt( v^T G v )
                riemann_dist = torch.sqrt(fisher_norm_sq.clamp(min=1e-12))

                # 4. Curvature Scale
                lambda_bar = self._estimate_fisher_scale(x_current, p, n_probes=n_fisher_probes)

                # 5. Spherical Loss (Distance to Uniform - purely for comparison)
                u = torch.ones_like(p) / p.shape[-1]
                p_sqrt = p.sqrt()
                u_sqrt = u.sqrt()
                # Just measure dot product as proxy for spherical position
                spherical_pos = 1.0 - torch.einsum("bc,bc->b", p_sqrt, u_sqrt)
                entropy = -torch.sum(p * torch.log(p + 1e-9), dim=1).cpu().numpy() / (np.log(logits.shape[-1]) + 1e-9)
                # Log Everything
                info["eta_step"].append(torch.ones(B).numpy())  # Placeholder 1.0
                info["fisher_norm_sq"].append(fisher_norm_sq.cpu().numpy())
                info["lambda_bar"].append(lambda_bar.cpu().numpy())
                info["spherical_loss"].append(spherical_pos.cpu().numpy())
                info["delta_riemann"].append(riemann_dist.cpu().numpy())
                info["norm_entropy"].append(entropy)
                info["delta_euclidean"].append(euclidean_dist.cpu().numpy())

            # B. Attribution Gradient
            # Standard IG: grad(x_i)
            grad = self._gradient_target_logit(x_current)

            # Accumulate: grad * step
            attributions += grad * step_vector

            # C. Advance
            # In-place update for next iteration
            x_current = x_current + step_vector
            # Detach to free graph, then require grad for next autograd
            x_current = x_current.detach().requires_grad_(True)

        # Post-process info
        for k in info:
            info[k] = np.stack(info[k], axis=0)

        return attributions.detach().cpu(), info

    def _gradient_target_logit(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model_forward(x, self.target_idx)
        (grads,) = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))
        return grads.detach()


# =============================================================================
# Main: Head-to-Head Comparison
# =============================================================================

def main():
    from utils import load_model, load_image  # Assuming these exist in your env

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transformation = load_model("resnet18", device=device)

    # Load image
    img = load_image("../examples/n01580077_jay.JPEG",transformation,  device=device)
    # Batch of 1 for clarity in plotting
    batch = img

    with torch.no_grad():
        target_idx = model(batch).argmax(dim=1)

    def model_forward_batch(x, target):
        return model(x).gather(1, target[:, None]).squeeze(1)

    print(f"Target Class: {target_idx.item()}")

    # ---------------------------------------------------------
    # 1. Run Classical Integrated Gradients
    # ---------------------------------------------------------
    print("\n--- Running Classical Integrated Gradients ---")
    ig_classic = IntegratedGradients(model, model_forward_batch, target_idx)
    print(batch.shape)
    attr_ig, info_ig = ig_classic.attribute(batch, n_steps=100)
    print(info_ig["logit"])

    # # 2. Instantiate Scorers
    # sens_scorer = MaxSensitivityScorer(ig_classic)
    # infid_scorer = InfidelityScorer(model)
    # gini_scorer = SparsenessScorer()
    #
    # # 3. Run Evaluations
    # print("\n--- Evaluation Report ---")
    #
    # # # A. Robustness
    # # s_score = sens_scorer.score(batch, original_attr=attr_ig,
    # #                             n_steps=100)
    # # print(f"Max-Sensitivity: {s_score:.4f} (Lower is stable)")
    #
    # # B. Faithfulness
    # i_score = infid_scorer.score(batch, attr_ig, target_idx.item())
    # print(f"Infidelity:      {i_score:.6f} (Lower is faithful)")
    #
    # # C. Conciseness
    # g_score = gini_scorer.score(attr_ig)
    # print(f"Gini Index:      {g_score:.4f} (Higher is sparse/sharp)")
    #
    # print("-" * 25)


    # ---------------------------------------------------------
    # 2. Run Fisher-Rao (Previous Code - simulated import here)
    # ---------------------------------------------------------
    # Assuming MaxEntropyFR is imported from previous cell/file
    # ig_fr = MaxEntropyFR(model, model_forward_batch, target_idx)
    # attr_fr, info_fr = ig_fr.attribute(batch, n_steps=100, c=1.0)

    # For this script to be standalone runnable, I'll just plot the IG results
    # and explain what you will see compared to FR.

    # ---------------------------------------------------------
    # 3. Plotting IG Geometry
    # ---------------------------------------------------------
    riemann = info_ig['delta_riemann'][:, 0]
    euclidean = info_ig['delta_euclidean'][:, 0]
    steps = np.arange(len(euclidean)) / len(euclidean)

    ##reverse
    riemann = riemann[::-1]
    euclidean = euclidean[::-1]
    steps = steps[::-1]
    normalized_entropy = info_ig['norm_entropy'][::-1]
    set_icml_matplotlib_style(base_fontsize=13, tick_fontsize=11, legend_fontsize=8, title_fontsize=14, line_width=2.0)
    visualize_geometry(riemann, euclidean, steps, entropy=normalized_entropy, title="IG Geometry \n Riemannian distance vs Euclidean distance increment per step")


if __name__ == "__main__":
    main()