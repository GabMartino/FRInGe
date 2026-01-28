from __future__ import annotations
import os
import pickle

from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple


import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from FisherRaoIG.plot_utils import plot_all_diagnostics, visualize_attr_triplet, visualize_geometry, \
    set_icml_matplotlib_style
from utils import load_model, load_image, denormalize_image


@dataclass
class LogConfig:
    enabled: bool = True

    # If True, tensors are detached->cpu->numpy at log time (lower GPU memory usage).
    to_cpu_numpy: bool = True

    # Visualization / intermediate artifacts
    store_viz: bool = True          # store intermediate_x / attr maps
    plot_viz: bool = False          # actually plt.show()
    viz_every: int = 10             # visualize/log viz every N global steps


class LogBuffer:
    """
    Centralizes logging and ensures consistent finalization:
    - Scalars -> np.array([..])
    - Arrays  -> np.stack([...], axis=0)
    - Empty   -> np.array([])
    """
    def __init__(self, cfg: LogConfig):
        self.cfg = cfg
        self.data: Dict[str, List[Any]] = defaultdict(list)

        # viz state (not returned unless you want it)
        self._viz_state: Dict[str, Any] = {}
        self._prev_attr_raw: Optional[np.ndarray] = None

    def add(self, key: str, value: Any):
        if not self.cfg.enabled:
            return

        if isinstance(value, torch.Tensor):
            v = value.detach()
            if self.cfg.to_cpu_numpy:
                v = v.cpu()
            value = v.numpy()

        self.data[key].append(value)

    def add_scalar(self, key: str, value: Any):
        if not self.cfg.enabled:
            return
        self.data[key].append(float(value))

    def finalize(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k, vlist in self.data.items():
            if len(vlist) == 0:
                out[k] = np.array([])
                continue

            v0 = vlist[0]
            if isinstance(v0, (float, int, np.floating, np.integer)):
                out[k] = np.asarray(vlist, dtype=np.float32)
            else:
                # For arrays/images/vectors: stack along time
                out[k] = np.stack(vlist, axis=0)
        return out

    # ---- Visualization logging ----

    def log_viz_step(
        self,
        *,
        x_curr: torch.Tensor,
        attributions: torch.Tensor,
        current_entropy: np.ndarray,
        current_logit: np.ndarray,
        step_idx: int,
        denormalize_image: Callable[[torch.Tensor], torch.Tensor],
    ):
        """
        Stores:
          - intermediate_x: (H,W,3) float image
          - intermediate_entropy: (B,) or scalar array
          - intermediate_attr_cum: cumulative attribution map A_t (H,W)
          - intermediate_attr: delta attribution map d_t (H,W) in global-scaled [-1,1]
          - current_logit: (B,) or scalar array

        Also (optionally) plots a 1x3 panel.
        """
        if not (self.cfg.enabled and self.cfg.store_viz):
            return

        # only do work when needed
        if self.cfg.viz_every > 1 and (step_idx % self.cfg.viz_every != 0):
            return

        img_vis = denormalize_image(x_curr[0].detach()).permute(1, 2, 0).cpu().numpy().clip(0, 1)

        # cumulative attribution at this step (your original definition)
        A_t = attributions[0].detach().cpu().abs().sum(dim=0).numpy()

        # delta vs previous
        if self._prev_attr_raw is None:
            d_t = np.zeros_like(A_t)
        else:
            d_t = A_t - self._prev_attr_raw
        self._prev_attr_raw = A_t.copy()

        # global scaling (stable across steps)
        # cumulative scale
        p99 = float(np.percentile(A_t, 99))
        self._viz_state["attr_scale_p99"] = max(self._viz_state.get("attr_scale_p99", 0.0), p99)
        A_vis = np.clip(A_t / (self._viz_state["attr_scale_p99"] + 1e-9), 0, 1)

        # delta scale
        p99d = float(np.percentile(np.abs(d_t), 99))
        self._viz_state["delta_scale_p99"] = max(self._viz_state.get("delta_scale_p99", 0.0), p99d)
        d_vis = np.clip(d_t / (self._viz_state["delta_scale_p99"] + 1e-9), -1, 1)

        # store
        self.add("intermediate_x", img_vis)
        self.add("intermediate_entropy", current_entropy)
        self.add("intermediate_attr_cum", A_t)
        self.add("intermediate_attr", d_vis)
        self.add("current_logit", current_logit)

        # optional quick plot
        if self.cfg.plot_viz:
            plt.figure(figsize=(15, 3))
            plt.subplot(1, 3, 1)
            plt.imshow(img_vis)
            plt.title(f"Step {step_idx} - Ent: {current_entropy}")
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.imshow(A_vis, cmap="inferno", vmin=0, vmax=1)
            plt.title("Attribution (global-scaled)")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.imshow(d_vis, cmap="coolwarm", vmin=-1, vmax=1)
            plt.title("Δ vs prev (global-scaled)")
            plt.axis("off")
            plt.show()

_LAPLACIAN_KERNEL = torch.tensor([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]], dtype=torch.float32)


def laplacian_matvec(v: torch.Tensor, pad_mode: str = "reflect") -> torch.Tensor:
    # Ensure kernel is on the correct device and type only when needed
    global _LAPLACIAN_KERNEL
    if _LAPLACIAN_KERNEL.device != v.device or _LAPLACIAN_KERNEL.dtype != v.dtype:
        _LAPLACIAN_KERNEL = _LAPLACIAN_KERNEL.to(device=v.device, dtype=v.dtype)

    kernel = _LAPLACIAN_KERNEL.view(1, 1, 3, 3)
    C = v.shape[1]
    # Use expand instead of repeat to save memory (view only)
    weight = kernel.expand(C, -1, -1, -1)

    v_pad = F.pad(v, (1, 1, 1, 1), mode=pad_mode)
    return F.conv2d(v_pad, weight, padding=0, groups=C)

def biharmonic_matvec(v: torch.Tensor, pad_mode: str = "reflect") -> torch.Tensor:
    return laplacian_matvec(laplacian_matvec(v, pad_mode=pad_mode), pad_mode=pad_mode)

class FisherRaoIntegratedGradients:
    def __init__(
            self,
            model: torch.nn.Module,
            model_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            target_idx: torch.Tensor,
            seed: int = 42,
            use_exact_jvp: bool = True
    ):
        self.model = model
        self.model_forward = model_forward
        self.target_idx = target_idx
        self.seed = seed
        self.use_exact_jvp = use_exact_jvp
        self.device = next(model.parameters()).device
        self.model.eval()
        self._reset_rng()

    def _reset_rng(self):
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.seed)

    def _compute_geodesic_waypoints(self, x: torch.Tensor, n_steps: int,
                                    baseline_p: torch.Tensor = None) -> torch.Tensor:
        logits = self.model(x)
        p_x = F.softmax(logits, dim=1)

        if baseline_p is not None:
            u = baseline_p
        else:
            u = torch.ones_like(p_x) / p_x.shape[-1]

        # Clamp BEFORE sqrt to prevent NaN gradients
        p_sqrt = torch.sqrt(p_x.clamp(min=1e-12))
        u_sqrt = torch.sqrt(u.clamp(min=1e-12))

        dot = torch.einsum("bc,bc->b", p_sqrt, u_sqrt)
        theta = torch.acos(dot.clamp(-1 + 1e-7, 1 - 1e-7))
        sin_theta = torch.sin(theta)

        small_angle = sin_theta.abs() < 1e-6
        ts = torch.linspace(0.0, 1.0, n_steps, device=self.device, dtype=x.dtype)

        phi_list = []
        for t in ts:
            w1 = torch.where(small_angle, 1 - t, torch.sin((1 - t) * theta) / sin_theta)
            w2 = torch.where(small_angle, t, torch.sin(t * theta) / sin_theta)
            phi_t = p_sqrt * w1.unsqueeze(-1) + u_sqrt * w2.unsqueeze(-1)
            phi_list.append(phi_t)

        return torch.stack(phi_list, dim=1)

    def _compute_jvp(self, x: torch.Tensor, v: torch.Tensor, f=None) -> torch.Tensor:
        f = f if f is not None else self.model

        if self.use_exact_jvp:
            _, Jv = torch.func.jvp(lambda inp: f(inp), (x,), (v,))
            return Jv

        with torch.no_grad():
            # Target RMS displacement in x-space (normalized units)
            delta_rms = 1e-12

            # Compute per-sample RMS of v
            v_flat = v.reshape(v.shape[0], -1)
            v_rms = torch.sqrt((v_flat * v_flat).mean(dim=1)).clamp_min(1e-12)

            # eps per sample so that ||eps * v||_rms ~= delta_rms
            eps = (delta_rms / v_rms).clamp(min=1e-9, max=1e-2)  # reasonable bounds

            # Broadcast eps to v shape
            eps_view = eps.view((v.shape[0],) + (1,) * (v.ndim - 1))

            f_pos = f(x + eps_view * v)
            f_neg = f(x - eps_view * v)
            return (f_pos - f_neg) / (2.0 * eps.view(-1, 1))  # see note below

    def _fisher_matvec(
            self,
            x: torch.Tensor,
            p: torch.Tensor,
            v: torch.Tensor,
            *,
            vjp_fn=None,  # callable: cotangent -> (J^T cotangent,)
            f=None  # optional, passed to JVP
    ) -> torch.Tensor:
        f = f if f is not None else self.model

        # Jv
        Jv = self._compute_jvp(x, v, f=f)

        # S(Jv) where S = diag(p) - p p^T
        pJv = p * Jv
        S_Jv = pJv - p * pJv.sum(dim=1, keepdim=True)

        # J^T S_Jv
        if vjp_fn is None:
            # Fallback: original behavior (will re-run forward each time)
            with torch.enable_grad():
                logits = f(x)
                (JT_S_Jv,) = torch.autograd.grad(
                    logits, x, grad_outputs=S_Jv, retain_graph=False
                )
        else:
            # Closure path: no forward here
            with torch.enable_grad():
                (JT_S_Jv,) = vjp_fn(S_Jv)

        return torch.nan_to_num(JT_S_Jv)

    def _estimate_fisher_diagonal(self, grad: torch.Tensor,
                                  gauss_blur_kernel_diag: int = 5,
                                  gauss_blur_sigma_diag: float = 2.0,
                                  clamp_min_diag: float = 1e-1,
                                  clamp_max_diag: float = 1e1 ) -> torch.Tensor:
        """
        Estimates a Spatially Adaptive Damping Matrix (Diagonal) using grad**2.
        Includes smoothing and clamping to prevent 'shattering' optimization.
        """
        # 1. Raw Empirical Fisher
        g_sq = grad ** 2

        # 2. Spatial Smoothing (Crucial for Diagonal Stability)
        # Prevents single-pixel spikes from freezing the solver
        # Kernel 5, Sigma 1.0 = Gentle blur
        g_sq_smooth = gaussian_blur(g_sq, kernel_size=gauss_blur_kernel_diag, sigma=gauss_blur_sigma_diag)

        # 3. Normalize
        # We want the 'shape' of the curvature, not the absolute magnitude.
        # Magnitude is controlled by 'alpha' in the main loop.
        mean_scale = g_sq_smooth.reshape(grad.shape[0], -1).mean(dim=1, keepdim=True)
        mean_scale = mean_scale.view((grad.shape[0],) + (1,) * (grad.ndim - 1))

        normalized_diag = g_sq_smooth / (mean_scale + 1e-12)

        # 4. Dynamic Range Clip
        # Prevent condition number explosion. If contrast > 1e4, CG fails.
        normalized_diag = normalized_diag.clamp(min=clamp_min_diag, max=clamp_max_diag)

        return normalized_diag

    def _pcg_solve(
            self,
            A_fn,
            b: torch.Tensor,
            M_inv: torch.Tensor,  # elementwise inverse preconditioner, same shape as b
            x0: Optional[torch.Tensor] = None,
            max_iter: int = 20,
            atol: float = 1e-5,
            rtol: float = 1e-5,
    ):
        """
        Preconditioned Conjugate Gradient (PCG) with a diagonal (Jacobi) preconditioner.
        Uses M_inv ~ 1 / diag(D) where D is your damping_diag.
        """

        b = torch.nan_to_num(b, nan=0.0, posinf=1e5, neginf=-1e5)
        B = b.shape[0]

        x = torch.zeros_like(b) if x0 is None else x0.detach().clone()

        # r = b - A x
        r = b - A_fn(x) if x0 is not None else b.clone()

        # termination threshold based on ||r||2
        b_norm = b.reshape(B, -1).norm(dim=1)
        tol_sq = (b_norm * rtol + atol) ** 2

        # z = M^{-1} r  (Jacobi preconditioning)
        z = M_inv * r

        # p = z
        p = z.clone()

        # Scalars for CG
        rs_old = (r * r).reshape(B, -1).sum(dim=1)  # ||r||^2 for stopping
        rz_old = (r * z).reshape(B, -1).sum(dim=1)  # r^T z for alpha/beta

        n_steps = 0
        for k in range(max_iter):
            n_steps += 1

            Ap = A_fn(p)
            pAp = (p * Ap).reshape(B, -1).sum(dim=1)

            # Curvature / definiteness check
            bad_curvature = (pAp <= 1e-20) | torch.isnan(pAp) | torch.isinf(pAp)

            # Avoid division by 0/NaN
            pAp_safe = pAp.masked_fill(bad_curvature, 1.0)
            rz_safe = rz_old.masked_fill(bad_curvature, 0.0)

            alpha = rz_safe / pAp_safe

            # Active batches: not converged and not bad curvature
            mask_active = (~bad_curvature) & (rs_old > tol_sq)

            if not mask_active.any():
                break

            alpha_v = alpha.masked_fill(~mask_active, 0.0).reshape((B,) + (1,) * (b.ndim - 1))

            x = x + alpha_v * p
            r = r - alpha_v * Ap

            rs_new = (r * r).reshape(B, -1).sum(dim=1)

            # Preconditioned residual
            z = M_inv * r
            rz_new = (r * z).reshape(B, -1).sum(dim=1)

            # beta = (r_new^T z_new) / (r_old^T z_old)
            denom = rz_old.clamp_min(1e-30)
            beta = (rz_new / denom).masked_fill(~mask_active, 0.0)

            beta_v = beta.reshape((B,) + (1,) * (b.ndim - 1))
            p = z + beta_v * p

            rs_old = rs_new
            rz_old = rz_new

        x = torch.nan_to_num(x, nan=0.0)

        return x, n_steps

    def _gradient_target_logit(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model_forward(x, self.target_idx)
        (grads,) = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))
        return grads.detach()
    def compute_optimal_steps(self, p_start: torch.Tensor, kl_target: float, num_classes: int) -> int:
        """Calculates synchronization steps T based on Fisher-Rao geometry."""
        # 1. Target is uniform distribution
        u_sqrt = torch.full_like(p_start, 1.0 / np.sqrt(num_classes))
        p_sqrt = p_start.sqrt()

        # 2. Total Geodesic Distance
        rho = (p_sqrt * u_sqrt).sum(dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
        total_arc_length = 2 * torch.acos(rho).max().item()

        # 3. Step Size from KL
        step_size = np.sqrt(2 * kl_target)

        # 4. Steps needed (+10% safety buffer for curvature)
        return int(np.ceil((total_arc_length / step_size) * 1.1))
    def attribute(
        self,
        x: torch.Tensor,
        ### Iteration
        max_correction_steps: int = 10,
        tolerance: float = 0.05,
        ## step
        kl_target: float = 0.05,
        eta_base: float = 3.0,
        delta_euc: float = 3.0,
        ## Fisher Rao Approx
        cg_max_iter: int = 20,
        baseline_p: torch.Tensor = None,
        alpha_start: float = 100.0,
        alpha_end: float = 10.0,
        gauss_blur_kernel_diag: int = 3,
        gauss_blur_sigma_diag: float = 1.0,
        gamma: float = 0.0001,
        clamp_min_diag: float = 1e-1,
        clamp_max_diag: float = 1e1,
        diag_floor: float = 1e-3,
        fisher: bool = True,
        verbose: bool = False,

        # NEW: single place to control all logging/viz behavior
        log_cfg: Optional[LogConfig] = None,

        # needed for viz logging (same as your existing helper)
        denormalize_image: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:

        self._reset_rng()
        B = x.shape[0]
        eta_base = torch.tensor(eta_base, device=x.device, dtype=torch.float32)
        cfg = log_cfg if log_cfg is not None else LogConfig(
            enabled=True,
            to_cpu_numpy=True,
            store_viz=True,
            plot_viz=bool(verbose),   # tie plotting to verbose by default
            viz_every=10,
        )
        log = LogBuffer(cfg)


        x_prev = x.clone().requires_grad_(True)
        attributions = torch.zeros_like(x)
        direction = None

        with torch.no_grad():
            logits = self.model(x)
            p = F.softmax(logits, dim=1)
            f_start = torch.gather(logits, -1, self.target_idx[:, None])

        n_waypoints = self.compute_optimal_steps(p_start=p, kl_target=kl_target, num_classes=logits.shape[-1])
        waypoints = self._compute_geodesic_waypoints(x, n_waypoints, baseline_p)
        global_step = 0
        torch.cuda.empty_cache()

        for t in tqdm(range(1, n_waypoints), desc="Fisher Descent"):
            progress = t / n_waypoints
            curr_alpha = alpha_start * (1 - progress) + alpha_end * progress
            s_t = waypoints[:, t, :]

            for sub_step in range(max_correction_steps):
                logits = self.model(x_prev)
                p = F.softmax(logits, dim=1)

                curr_ent = -torch.sum(p * torch.log(p + 1e-9), dim=1)

                # Clamp BEFORE sqrt to prevent NaN gradients
                p_safe = p.clamp(min=1e-12)
                p_sqrt = p_safe.sqrt()
                grad_prev = self._gradient_target_logit(x_prev)
                spherical_loss = 1.0 - torch.einsum("bc, bc -> b", p_sqrt, s_t)

                if spherical_loss.max().item() < tolerance and sub_step > 0:
                    break

                (grad_spherical_loss,) = torch.autograd.grad(
                    spherical_loss,
                    x_prev,
                    grad_outputs=torch.ones_like(spherical_loss),
                    retain_graph=True,
                )
                grad_spherical_loss = torch.nan_to_num(grad_spherical_loss)

                total_loss = grad_spherical_loss + gamma * biharmonic_matvec(x_prev)

                # ----------------------- Solve direction -----------------------
                if fisher:
                    diag_shape = self._estimate_fisher_diagonal(
                        grad_spherical_loss,
                        gauss_blur_kernel_diag=gauss_blur_kernel_diag,
                        gauss_blur_sigma_diag=gauss_blur_sigma_diag,
                        clamp_min_diag=clamp_min_diag,
                        clamp_max_diag=clamp_max_diag,
                    )
                    damping_diag = (curr_alpha * diag_shape) + diag_floor

                    def vjp_fn(cotangent: torch.Tensor):
                        (gx,) = torch.autograd.grad(
                            outputs=logits,
                            inputs=x_prev,
                            grad_outputs=cotangent,
                            retain_graph=True,
                            create_graph=False,
                            allow_unused=False,
                        )
                        return (gx,)

                    def G_mv(v):
                        return self._fisher_matvec(x_prev, p, v, vjp_fn=vjp_fn, f=self.model)

                    def A_fn(v):
                        return G_mv(v) + damping_diag * v + gamma * biharmonic_matvec(v)

                    approx_laplacian_diag = 20.0
                    precond_diag = damping_diag + (gamma * approx_laplacian_diag)
                    M_inv = 1.0 / precond_diag.clamp_min(1e-8)

                    direction, n_cg_steps = self._pcg_solve(
                        A_fn,
                        total_loss,
                        M_inv=M_inv,
                        x0=direction,
                        max_iter=cg_max_iter,
                        rtol=5e-3,
                        atol=1e-8,
                    )
                    Av = G_mv(direction)
                    fisher_norm_sq = (direction * Av).reshape(B, -1).sum(dim=1).clamp_min(1e-12)
                    eta_kl = torch.sqrt(2.0 * kl_target / fisher_norm_sq)
                    #eta = torch.minimum(eta_base, eta_kl)
                    with torch.no_grad():
                        v_flat = direction.reshape(B, -1)
                    eta_euclid = delta_euc / v_flat.norm(dim=1)
                    eta = torch.minimum(eta_base, torch.minimum(eta_kl, eta_euclid))




                    # residual (for solver quality)
                    with torch.no_grad():
                        res = total_loss - A_fn(direction)
                        rel_res = res.reshape(B, -1).norm(dim=1) / (
                            total_loss.reshape(B, -1).norm(dim=1) + 1e-12
                        )
                        log.add("relative_residual", rel_res)
                        log.add("alpha_t", torch.full((B,), float(curr_alpha), device=rel_res.device))
                        log.add("damped_fisher_step_size", 0.5 * (eta ** 2) * fisher_norm_sq)
                        log.add("n_cg_steps", torch.full((B,), float(n_cg_steps), device=rel_res.device))

                        # energies
                        Gv = G_mv(direction)
                        Dv = damping_diag * direction + gamma * biharmonic_matvec(direction)
                        G_energy = (direction * Gv).reshape(B, -1).sum(dim=1)
                        D_energy = (direction * Dv).reshape(B, -1).sum(dim=1)
                        frac_D = D_energy / (G_energy + D_energy + 1e-12)
                        log.add("fraction_D_vs_G", frac_D)
                        log.add("pure_fisher_step_size", 0.5 * (eta ** 2) * G_energy)

                        # norms
                        v_flat = direction.reshape(B, -1)
                        v_norm_sq = (v_flat.norm(dim=1) ** 2)
                        log.add("euclidean_norm_sq", eta * v_norm_sq)

                        # diagnostic scalars (global across tensor)
                        d = diag_shape.detach()
                        D = damping_diag.detach()
                        log.add_scalar("diag_min", d.amin().item())
                        log.add_scalar("diag_med", d.median().item())
                        log.add_scalar("diag_max", d.amax().item())
                        log.add_scalar("D_min", D.amin().item())
                        log.add_scalar("D_med", D.median().item())
                        log.add_scalar("D_max", D.amax().item())
                        log.add_scalar("D_cond_proxy", (D.amax() / D.amin().clamp_min(1e-30)).item())



                else:
                    # keep shape consistent: (B,)
                    eta = torch.full((B,), float(eta_base), device=x.device, dtype=x.dtype)
                    direction = grad_spherical_loss

                    # still log placeholders so downstream plotting doesn’t break
                    log.add("relative_residual", torch.full((B,), np.nan, device=x.device, dtype=x.dtype))
                    log.add("alpha_t", torch.full((B,), float(curr_alpha), device=x.device, dtype=x.dtype))
                    log.add("damped_fisher_step_size", torch.full((B,), np.nan, device=x.device, dtype=x.dtype))
                    log.add("pure_fisher_step_size", torch.full((B,), np.nan, device=x.device, dtype=x.dtype))
                    log.add("euclidean_norm_sq", torch.full((B,), np.nan, device=x.device, dtype=x.dtype))
                    log.add("fraction_D_vs_G", torch.full((B,), np.nan, device=x.device, dtype=x.dtype))
                    log.add("n_cg_steps", torch.full((B,), 0.0, device=x.device, dtype=x.dtype))

                # ----------------------- Step & IG accumulation -----------------------
                eta_view = eta.reshape((B,) + (1,) * (x.ndim - 1))
                x_next = x_prev - eta_view * direction
                x_next = x_next.detach().requires_grad_(True)

                step_vec = (x_next - x_prev).detach()


                grad_next = self._gradient_target_logit(x_next)
                attributions += 0.5 * (grad_prev + grad_next) * step_vec

                with torch.no_grad():
                    f_curr = self.model_forward(x_next.detach(), self.target_idx)
                    completeness_err = (attributions.reshape(B, -1).sum(-1) - (f_curr - f_start)).abs()

                    grad_mid = 0.5 * (grad_prev + grad_next)
                    total_grad_norm = grad_mid.reshape(B, -1).norm(dim=1)
                    directional_deriv = (grad_mid * step_vec).reshape(B, -1).sum(dim=1)

                    log.add("total_grad_norm", total_grad_norm)
                    log.add("directional_derivative", directional_deriv)
                    # core per-step logs
                    log.add("logit", f_curr)
                    log.add("spherical_loss", spherical_loss)
                    log.add("eta_step", eta)
                    log.add("delta", completeness_err)
                    log.add("entropy", curr_ent)

                    # geometry / step diagnostics
                    s = step_vec.reshape(B, -1)
                    s_norm = s.norm(dim=1).mean().item()
                    log.add_scalar("step_l2",s_norm )
                    log.add_scalar("step_linf", s.abs().max(dim=1).values.mean().item())
                    dot = (p_sqrt * s_t).sum(dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
                    theta_err = torch.acos(dot)
                    log.add_scalar("theta_err", theta_err.mean().item())
                    log.add_scalar("delta_euclidean", s.norm(dim=1))
                    # 2) Fisher pullback step (pure G)
                    G_step = G_mv(step_vec)  # uses current p/logits closure
                    fisher_norm_sq_step = (step_vec * G_step).reshape(B, -1).sum(dim=1).clamp_min(1e-12)
                    delta_riemann = torch.sqrt(fisher_norm_sq_step)
                    A_step = A_fn(step_vec)  # uses current p/logits closure
                    delta_riemann_reg = torch.sqrt((step_vec * A_step).reshape(B, -1).sum(dim=1).clamp_min(1e-12))

                    log.add("delta_riemann", delta_riemann)
                    log.add("delta_riemann_regularized", delta_riemann_reg )

                # viz logging (optional)
                if cfg.enabled and cfg.store_viz:
                    if denormalize_image is None:
                        # If you forget to pass it, we simply skip viz to avoid crashing.
                        pass
                    else:
                        log.log_viz_step(
                            x_curr=x_prev,
                            attributions=-attributions,
                            current_entropy=curr_ent.detach().cpu().numpy(),
                            current_logit=f_curr.detach().cpu().numpy(),
                            step_idx=global_step,
                            denormalize_image=denormalize_image,
                        )

                x_prev = x_next
                global_step += 1

        info = log.finalize()
        return -attributions.detach().cpu(), info



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- Load model + image ----
    model, transformation = load_model("resnet18", device=device)
    img_path = "../examples/n01580077_jay.JPEG"
    if not os.path.exists(img_path):
        print(f"Missing image: {img_path}")
        return

    img = load_image(img_path, transformation, device=device)
    batch = img  # already (1,C,H,W)

    with torch.no_grad():
        logits0 = model(batch)
        target_idx = logits0.argmax(dim=1)

    def model_forward_batch(x, target):
        return model(x).gather(1, target[:, None]).squeeze(1)

    # ---- Run FR IG ----
    ig = FisherRaoIntegratedGradients(
        model=model,
        model_forward=model_forward_batch,
        target_idx=target_idx,
        use_exact_jvp=True,
    )

    cfg = LogConfig(
        enabled=True,
        to_cpu_numpy=True,
        store_viz=True,
        plot_viz=True,   # set False if you only want the final geometry plot
        viz_every=3,
    )

    attributions, info = ig.attribute(
        batch,
        max_correction_steps=1,
        cg_max_iter=20,
        eta_base=10,
        delta_euc=10,
        kl_target=0.001,
        gamma=0.01,
        alpha_start=0.01,
        alpha_end=0.01,
        diag_floor=0.01,
        clamp_max_diag=10,
        clamp_min_diag=0.01,
        gauss_blur_sigma_diag=2.0,
        gauss_blur_kernel_diag=5,
        log_cfg=cfg,
        fisher=True,
        denormalize_image=denormalize_image,
        verbose=True,
    )

    # ---- Plot geometry diagnostics ----
    # Note: delta_euclidean was logged as a vector; don't call add_scalar for it.
    delta_euclidean = info["delta_euclidean"].squeeze()
    delta_riemann = info["delta_riemann"].squeeze()
    entropy = info["entropy"].squeeze()

    # Normalize entropy by log(C) using *final* logits shape is fine; C is constant anyway.
    num_classes = logits0.shape[-1]
    entropy = entropy / np.log(num_classes)

    T = len(delta_euclidean)
    steps = np.arange(T, dtype=float) / max(T, 1)

    set_icml_matplotlib_style(
        base_fontsize=13,
        tick_fontsize=11,
        legend_fontsize=8,
        title_fontsize=14,
        line_width=2.0,
    )

    visualize_geometry(
        delta_riemann,
        delta_euclidean,
        steps,
        entropy=entropy,
        title="FRInGe Geometry (Euclidean TR)\nRiemannian distance vs Euclidean distance increment per step",
    )


if __name__ == "__main__":
    main()