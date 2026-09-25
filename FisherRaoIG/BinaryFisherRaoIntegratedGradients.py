"""Selectable binary optimization for Fisher--Rao Integrated Gradients.

This module implements the target-vs-rest optimization used by the canonical
``FisherRaoIG`` implementation when ``binary=True``:

* collapse the C-class predictive distribution to ``[p_t, 1 - p_t]``;
* track the Fisher--Rao Slerp from that belief to ``[1/C, 1 - 1/C]``;
* use the closed-form damped natural-gradient direction induced by the
  rank-one binary Fisher pullback;
* integrate the target-logit gradient along the realised input-space path.

The optional ``smoothing`` branch mirrors the spatial regularisation used by
the full categorical implementation.  It is deliberately not part of the
closed-form method claimed in the paper: it applies Sherman--Morrison around
an approximately inverted regularisation operator.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

from FisherRaoIG.FR_utils import biharmonic_matvec, preconditionedCG_solve


class FisherRaoIntegratedGradients2Class:
    """FRInGe-2 using the target-vs-rest Fisher geometry."""

    def __init__(
        self,
        model: nn.Module,
        model_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        target_idx: torch.Tensor,
    ) -> None:
        self.model = model.eval()
        self.model_forward = model_forward
        self.target_idx = target_idx

    @staticmethod
    def _safe_probability(p: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        return p.clamp(min=eps, max=1.0 - eps)

    @staticmethod
    def _logodds_one_vs_rest_from_logits(
        logits: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return F_t - logsumexp(F_j, j != t) without forming probabilities."""
        if logits.ndim != 2:
            raise ValueError(f"Expected logits with shape (B, C), got {tuple(logits.shape)}")
        if logits.shape[1] < 2:
            raise ValueError("The target-vs-rest collapse requires at least two classes.")

        z_t = logits.gather(1, target_idx[:, None]).squeeze(1)
        target_mask = F.one_hot(target_idx, num_classes=logits.shape[1]).bool()
        z_rest = logits.masked_fill(target_mask, -torch.inf)
        return z_t - torch.logsumexp(z_rest, dim=1)

    @classmethod
    def _binary_slerp_waypoints(
        cls,
        p_t0: torch.Tensor,
        num_classes: int,
        kl_target: float,
        safety_buffer: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build padded Slerp waypoints in binary square-root coordinates.

        The returned waypoint at index zero is the starting belief.  Therefore
        state ``x_k`` must target waypoint ``k + 1``.
        """
        if kl_target <= 0:
            raise ValueError("kl_target must be positive.")
        if not 0.0 <= safety_buffer < 1.0:
            raise ValueError("safety_buffer must lie in [0, 1).")
        if num_classes < 2:
            raise ValueError("num_classes must be at least two.")

        p_t0 = cls._safe_probability(p_t0)
        q_t = torch.full_like(p_t0, 1.0 / float(num_classes))

        psi0 = torch.stack((p_t0.sqrt(), (1.0 - p_t0).sqrt()), dim=-1)
        psi_star = torch.stack((q_t.sqrt(), (1.0 - q_t).sqrt()), dim=-1)

        inner = (psi0 * psi_star).sum(dim=-1).clamp(-1.0, 1.0)
        theta = torch.acos(inner)
        total_fr_distance = 2.0 * theta
        step_length = math.sqrt(2.0 * float(kl_target))
        intervals = torch.ceil(
            (total_fr_distance / step_length) * (1.0 + safety_buffer)
        ).long()
        n_steps = (intervals + 1).clamp(min=2)

        batch_size = p_t0.shape[0]
        max_steps = int(n_steps.max().item())
        indices = torch.arange(max_steps, device=p_t0.device)
        mask = indices[None, :] < n_steps[:, None]
        alpha = indices[None, :].to(p_t0.dtype) / (n_steps - 1)[:, None].to(p_t0.dtype)
        alpha = alpha.masked_fill(~mask, 0.0)

        theta_b = theta[:, None, None]
        alpha_b = alpha[:, :, None]
        sin_theta = torch.sin(theta_b)
        small = theta_b.abs() < 1e-7
        safe_sin = torch.where(small, torch.ones_like(sin_theta), sin_theta)

        w0 = torch.sin((1.0 - alpha_b) * theta_b) / safe_sin
        w1 = torch.sin(alpha_b * theta_b) / safe_sin
        w0 = torch.where(small, 1.0 - alpha_b, w0)
        w1 = torch.where(small, alpha_b, w1)

        waypoints = w0 * psi0[:, None, :] + w1 * psi_star[:, None, :]
        waypoints = waypoints.masked_fill(~mask[:, :, None], 0.0)
        return waypoints, mask, n_steps

    def _gradient_target_logit(self, x: torch.Tensor) -> torch.Tensor:
        target_score = self.model_forward(x, self.target_idx)
        if target_score.ndim != 1:
            raise ValueError(
                "model_forward must return one scalar target score per batch element."
            )
        (gradient,) = torch.autograd.grad(
            target_score,
            x,
            grad_outputs=torch.ones_like(target_score),
            retain_graph=False,
            create_graph=False,
        )
        return gradient.detach()

    @staticmethod
    def _batch_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left * right).reshape(left.shape[0], -1).sum(dim=1)

    @staticmethod
    def _view_batch(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return values.view(values.shape[0], *([1] * (reference.ndim - 1)))

    @classmethod
    def _closed_form_direction(
        cls,
        u: torch.Tensor,
        fisher_scalar: torch.Tensor,
        beta: torch.Tensor,
        lam: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve ``(a uu^T + lam I)v = beta u`` exactly."""
        u_sq = cls._batch_dot(u, u).clamp_min(0.0)
        denominator = (lam + fisher_scalar * u_sq).clamp_min(1e-12)
        scale = beta / denominator
        return cls._view_batch(scale, u) * u, denominator

    @staticmethod
    def _constraint_labels(
        eta_kl: torch.Tensor,
        eta_euclid: torch.Tensor,
        eta_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, list[str]]:
        candidates = torch.stack((eta_kl, eta_euclid, eta_max), dim=1)
        indices = candidates.argmin(dim=1)
        names = ["kl", "euclid", "max"]
        return indices, [names[index] for index in indices.cpu().tolist()]

    def _apply_regularizer_inverse(
        self,
        rhs: torch.Tensor,
        lam: float,
        gamma_step: float,
        use_sobolev_preconditioner: bool,
        blur_kernel_size: int,
        blur_sigma: float,
        max_iters: int,
        rtol: float,
        atol: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Approximately apply (lam I + gamma_step Delta^2)^-1."""
        if gamma_step == 0.0:
            zeros = torch.zeros(rhs.shape[0], device=rhs.device, dtype=torch.long)
            return rhs / lam, zeros

        def operator(value: torch.Tensor) -> torch.Tensor:
            return lam * value + gamma_step * biharmonic_matvec(value)

        if use_sobolev_preconditioner:
            if blur_kernel_size <= 0 or blur_kernel_size % 2 == 0:
                raise ValueError("blur_kernel_size must be a positive odd integer.")
            if blur_sigma <= 0:
                raise ValueError("blur_sigma must be positive.")

            def preconditioner(value: torch.Tensor) -> torch.Tensor:
                return TF.gaussian_blur(
                    value,
                    kernel_size=[blur_kernel_size, blur_kernel_size],
                    sigma=[blur_sigma, blur_sigma],
                ) / lam
        else:
            def preconditioner(value: torch.Tensor) -> torch.Tensor:
                return value / lam

        solution, iterations, _, _ = preconditionedCG_solve(
            A_fn=operator,
            b=rhs,
            M_op=preconditioner,
            max_iter=max_iters,
            rtol=rtol,
            atol=atol,
        )
        return solution, iterations

    def attribute(
        self,
        x: torch.Tensor,
        kl_target: float,
        delta_euc: float = 10.0,
        eta_max: float = 10.0,
        lam: float = 0.01,
        *,
        smoothing: bool = False,
        use_sobolev_preconditioner: bool = True,
        blur_kernel_size: int = 5,
        blur_sigma: float = 2.0,
        gamma_step: float = 0.01,
        gamma_prior: float = 0.001,
        Ainv_iters: int = 20,
        Ainv_rtol: float = 1e-4,
        Ainv_atol: float = 1e-6,
        safety_buffer: float = 0.1,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute FRInGe-2 attributions and trajectory diagnostics.

        With ``smoothing=False`` this is exactly the closed-form method in the
        draft.  Returned attributions satisfy the usual IG orientation,
        ``sum(attr) ~= F_t(x_0) - F_t(x_T)``.
        """
        if x.ndim < 2:
            raise ValueError("x must have a batch dimension and at least one feature dimension.")
        if self.target_idx.ndim != 1 or self.target_idx.shape[0] != x.shape[0]:
            raise ValueError("target_idx must have shape (B,) matching x.")
        if lam <= 0:
            raise ValueError("lam must be positive.")
        if delta_euc <= 0 or eta_max <= 0:
            raise ValueError("delta_euc and eta_max must be positive.")
        if Ainv_iters <= 0:
            raise ValueError("Ainv_iters must be positive.")
        if gamma_step < 0 or gamma_prior < 0:
            raise ValueError("gamma_step and gamma_prior must be non-negative.")

        batch_size = x.shape[0]
        eta_max_tensor = torch.full(
            (batch_size,), float(eta_max), device=x.device, dtype=x.dtype
        )

        x_prev = x.detach().clone().requires_grad_(True)
        path_integral = torch.zeros_like(x)

        with torch.no_grad():
            logits_start = self.model(x_prev)
            if logits_start.ndim != 2:
                raise ValueError(
                    f"model must return logits with shape (B, C), got {tuple(logits_start.shape)}"
                )
            logit_start = self.model_forward(x_prev, self.target_idx)
            p_start = F.softmax(logits_start, dim=1)

        _, num_classes = p_start.shape
        p_t0 = p_start.gather(1, self.target_idx[:, None]).squeeze(1)
        waypoints, waypoint_mask, n_steps = self._binary_slerp_waypoints(
            p_t0=p_t0,
            num_classes=num_classes,
            kl_target=kl_target,
            safety_buffer=safety_buffer,
        )
        max_steps = int(n_steps.max().item())

        stats: Dict[str, Any] = {
            "p_t": [],
            "target_logodds": [],
            "waypoint": [],
            "beta": [],
            "gradient_norm": [],
            "closed_form_denominator": [],
            "active_constraint": [],
            "tracking_error_spher": [],
            "fisher_rao_norm": [],
            "euclidean_norm_v": [],
            "step_size_euclidean": [],
            "regularizer_inverse_iters": [],
            "constraint_active": {"kl": 0, "euclid": 0, "max": 0},
            "n_steps": n_steps.detach().cpu().tolist(),
            "total_steps": max_steps - 1,
            "smoothing": bool(smoothing),
        }

        # One initial gradient, then roll it forward for trapezoidal quadrature.
        grad_prev = self._gradient_target_logit(x_prev)
        iteration = range(1, max_steps)
        if show_progress:
            iteration = tqdm(iteration, desc="FRInGe-2", leave=False)

        for waypoint_index in iteration:
            target_waypoint = waypoints[:, waypoint_index, :]
            active = waypoint_mask[:, waypoint_index]
            if not active.any():
                break

            logits = self.model(x_prev)
            probabilities = F.softmax(logits, dim=1)
            p_t = probabilities.gather(1, self.target_idx[:, None]).squeeze(1)
            p_t = self._safe_probability(p_t)
            binary_sqrt = torch.stack((p_t.sqrt(), (1.0 - p_t).sqrt()), dim=1)
            tracking_loss = 1.0 - (binary_sqrt * target_waypoint).sum(dim=1)

            logodds = self._logodds_one_vs_rest_from_logits(logits, self.target_idx)
            (u,) = torch.autograd.grad(
                logodds,
                x_prev,
                grad_outputs=torch.ones_like(logodds),
                retain_graph=False,
                create_graph=False,
            )
            u = u.detach()

            s_t = target_waypoint[:, 0]
            s_r = target_waypoint[:, 1]
            fisher_scalar = (p_t * (1.0 - p_t)).detach()
            beta = 0.5 * (
                p_t * torch.sqrt(1.0 - p_t) * s_r
                - (1.0 - p_t) * torch.sqrt(p_t) * s_t
            )
            beta = beta.detach()
            grad_loss = self._view_batch(beta, x_prev) * u

            u_sq = self._batch_dot(u, u).clamp_min(0.0)
            closed_form_v, closed_form_denom = self._closed_form_direction(
                u=u,
                fisher_scalar=fisher_scalar,
                beta=beta,
                lam=lam,
            )

            if smoothing:
                rhs = grad_loss
                if gamma_prior:
                    rhs = rhs + gamma_prior * biharmonic_matvec(x_prev.detach())

                # Sherman--Morrison around H = lam I + gamma_step Delta^2.
                paired_rhs = torch.cat((rhs, u), dim=0)
                paired_solution, inverse_iters = self._apply_regularizer_inverse(
                    rhs=paired_rhs,
                    lam=lam,
                    gamma_step=gamma_step,
                    use_sobolev_preconditioner=use_sobolev_preconditioner,
                    blur_kernel_size=blur_kernel_size,
                    blur_sigma=blur_sigma,
                    max_iters=Ainv_iters,
                    rtol=Ainv_rtol,
                    atol=Ainv_atol,
                )
                h_inv_rhs, h_inv_u = paired_solution.chunk(2, dim=0)
                inverse_iters_rhs, inverse_iters_u = inverse_iters.chunk(2, dim=0)

                u_hinv_rhs = self._batch_dot(u, h_inv_rhs)
                u_hinv_u = self._batch_dot(u, h_inv_u)
                sm_denom = (1.0 + fisher_scalar * u_hinv_u).clamp_min(1e-12)
                correction = fisher_scalar * u_hinv_rhs / sm_denom
                v = h_inv_rhs - self._view_batch(correction, x_prev) * h_inv_u
                inverse_iters_step = torch.maximum(inverse_iters_rhs, inverse_iters_u)
            else:
                v = closed_form_v
                inverse_iters_step = torch.zeros(
                    batch_size, device=x.device, dtype=torch.long
                )

            with torch.no_grad():
                v_norm = v.reshape(batch_size, -1).norm(dim=1).clamp_min(1e-12)
                u_dot_v = self._batch_dot(u, v)
                fisher_norm_sq = (fisher_scalar * u_dot_v.square()).clamp_min(1e-12)
                eta_kl = torch.sqrt(2.0 * float(kl_target) / fisher_norm_sq)
                eta_euclid = float(delta_euc) / v_norm
                constraint_indices, constraint_labels = self._constraint_labels(
                    eta_kl, eta_euclid, eta_max_tensor
                )
                constraint_labels = [
                    label if is_active else "inactive"
                    for label, is_active in zip(
                        constraint_labels, active.cpu().tolist()
                    )
                ]
                eta = torch.minimum(
                    eta_max_tensor, torch.minimum(eta_kl, eta_euclid)
                )
                for index, name in enumerate(("kl", "euclid", "max")):
                    stats["constraint_active"][name] += int(
                        ((constraint_indices == index) & active).sum().item()
                    )

                step = (
                    self._view_batch(active.to(x.dtype), x_prev)
                    * self._view_batch(eta, x_prev)
                    * v
                )
                x_next = x_prev - step

            if not torch.isfinite(x_next).all():
                raise RuntimeError(
                    f"FRInGe-2 produced a non-finite path state at waypoint {waypoint_index}."
                )

            x_next_req = x_next.detach().requires_grad_(True)
            grad_next = self._gradient_target_logit(x_next_req)
            path_integral += 0.5 * (grad_prev + grad_next) * (-step)

            stats["p_t"].append(p_t.detach().cpu().tolist())
            stats["target_logodds"].append(logodds.detach().cpu().tolist())
            stats["waypoint"].append(target_waypoint.detach().cpu().tolist())
            stats["beta"].append(beta.cpu().tolist())
            stats["gradient_norm"].append(u_sq.sqrt().cpu().tolist())
            stats["closed_form_denominator"].append(closed_form_denom.cpu().tolist())
            stats["active_constraint"].append(constraint_labels)
            stats["tracking_error_spher"].append(tracking_loss.detach().cpu().tolist())
            stats["fisher_rao_norm"].append(fisher_norm_sq.sqrt().cpu().tolist())
            stats["euclidean_norm_v"].append(v_norm.cpu().tolist())
            stats["step_size_euclidean"].append(
                step.reshape(batch_size, -1).norm(dim=1).cpu().tolist()
            )
            stats["regularizer_inverse_iters"].append(
                inverse_iters_step.cpu().tolist()
            )

            grad_prev = grad_next
            x_prev = x_next_req

        # Reverse the generated start->reference path so attribution has the
        # standard IG orientation reference->input.
        attributions = -path_integral

        with torch.no_grad():
            logits_end = self.model(x_prev)
            logit_end = self.model_forward(x_prev, self.target_idx)
            expected_difference = logit_start - logit_end
            attr_sum = attributions.reshape(batch_size, -1).sum(dim=1)
            completeness_delta = (attr_sum - expected_difference).abs()

            probabilities_end = F.softmax(logits_end, dim=1).clamp_min(1e-10)
            p_t_end = probabilities_end.gather(
                1, self.target_idx[:, None]
            ).squeeze(1)
            q_t = torch.full_like(p_t_end, 1.0 / float(num_classes))
            q_r = 1.0 - q_t
            binary_endpoint_kl = (
                p_t_end * (p_t_end.log() - q_t.log())
                + (1.0 - p_t_end)
                * ((1.0 - p_t_end).clamp_min(1e-10).log() - q_r.log())
            )
            uniform_log_prob = -math.log(float(num_classes))
            full_endpoint_kl = (
                probabilities_end * (probabilities_end.log() - uniform_log_prob)
            ).sum(dim=1)

            stats["completeness_delta"] = completeness_delta.cpu().tolist()
            stats["completeness_relative_error"] = (
                completeness_delta / expected_difference.abs().clamp_min(1e-9)
            ).cpu().tolist()
            stats["endpoint_error_probability"] = (
                p_t_end - q_t
            ).abs().cpu().tolist()
            stats["endpoint_error_kl"] = binary_endpoint_kl.cpu().tolist()
            stats["endpoint_full_kl_to_uniform"] = full_endpoint_kl.cpu().tolist()
            stats["endpoint_target_probability"] = p_t_end.cpu().tolist()
            stats["endpoint"] = x_prev.detach()

        return attributions.detach(), stats


# Clearer public alias while preserving the historic class name used by scripts.
BinaryFisherRaoIntegratedGradients = FisherRaoIntegratedGradients2Class
