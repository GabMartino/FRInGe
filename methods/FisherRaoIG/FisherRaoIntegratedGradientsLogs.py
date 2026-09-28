import os
from typing import Callable, Tuple, Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from methods.FisherRaoIG.FR_plot_utils import plot_fisher_rao_performance, plot_intermediate_steps, denormalize_n, \
    select_path_subset, \
    plot_entropy_with_snapshot_arrows_and_delta_attrs
# Adjust imports to match your project structure
from methods.FisherRaoIG.FR_utils import compute_waypoints, spherical_loss, fisher_mat_vect_product, compute_gradient, \
    preconditionedCG_solve, biharmonic_matvec, singular_psd_cg_solve
from common.utils import load_model, load_image, visualize_attributions, denormalize


def masked_to_list(values: torch.Tensor, active_mask: torch.Tensor):
    out = values.detach().clone()
    out[~active_mask] = float("nan")
    return out.cpu().tolist()
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

    def attribute_and_save(
            self,
            x: torch.Tensor,
            kl_target: float = None,
            T_desired: int = None,
            p_target: torch.Tensor = None,
            fisher=True,
            delta_euc: float = 10,
            eta_max: float = 10,
            use_sobolev_preconditioner: bool = True,
            lambda_ratio: float = 1,
            smoothing=False,
            gamma_step: float = 0.01,
            gamma_prior: float = 0.001,
            save_x_path: bool = False,
            save_every: int = 1,
            save_on_cpu: bool = True,
            save_indices=None,
    ) -> Tuple[torch.Tensor, Dict]:

        if kl_target is None and T_desired is None:
            raise ValueError("At least one of kl_target and T_desired should be set.")

        eta_max_tensor = torch.tensor(eta_max, device=x.device, dtype=x.dtype)
        x_prev = x.clone().requires_grad_(True)

        def _snapshot_tensor(z: torch.Tensor) -> torch.Tensor:
            z = z.detach()
            if save_indices is not None:
                z = z[save_indices]
            if save_on_cpu:
                z = z.cpu()
            return z.clone()

        # Initialize tracking tensors
        attributions = torch.zeros_like(x)
        chunk_attribution = torch.zeros_like(x)

        stats = {
            "completeness_delta": [], "completeness_relative_error": [], "endpoint_error_kl": [],
            "tracking_error_spher": [], "fisher_rao_norm": [], "euclidean_norm_v": [],
            "rayleigh_quotient": [], "step_size_euclidean": [], "target_logit": [], "entropy": [],
            "cg_iters": [], "delta_attr_path": [],
            "constraint_active": {"kl": 0, "euclid": 0, "max": 0},
            "total_steps": 0,
        }

        x_path = []
        saved_step_ids = []

        if save_x_path:
            x_path.append(_snapshot_tensor(x_prev))  # x_0
            saved_step_ids.append(0)
            # Note: We do NOT append to delta_attr_path here.
            # The first delta chunk corresponds to the transition finishing at step 1.

        with torch.no_grad():
            logit_start = self.model_forward(x, self.target_idx)
            p_start = self.compute_softmax(x_prev)

        C = p_start.shape[-1]
        B = p_start.shape[0]
        s_alphas, mask, n_steps = compute_waypoints(
            p_start=p_start,
            kl_target=kl_target,
            num_classes=C,
            p_target=p_target,
            T_desired=T_desired
        )

        max_T = int(n_steps.max().item())
        v = None
        cg_iter_history = []

        grad_prev = self._gradient_target_logit(x_prev)

        # Waypoint zero is the starting prediction; target the next waypoint.
        for i in tqdm(range(1, max_T), desc="FisherRao Ascent"):
            s_alpha = s_alphas[:, i, :]
            active_mask = mask[:, i]

            if not active_mask.any():
                break

            current_logits = self.model(x_prev)
            current_p = F.softmax(current_logits, dim=-1)

            current_p_safe = current_p.clamp(min=1e-10)
            current_p_sqrt = current_p_safe.sqrt()

            with torch.no_grad():
                current_loss = spherical_loss(current_p_sqrt, s_alpha)
                stats["tracking_error_spher"].append(current_loss.cpu().tolist())
                tgt_logit = current_logits.gather(1, self.target_idx.unsqueeze(1)).squeeze(1)
                stats["target_logit"].append(tgt_logit.cpu().tolist())
                eps = 1e-10
                entropy = -(current_p * (current_p + eps).log()).sum(dim=-1)
                stats["entropy"].append(entropy.cpu().tolist())

            spher_loss = spherical_loss(current_p_sqrt, s_alpha)
            grad_spher_loss = compute_gradient(spher_loss, x_prev, retain_graph=False)

            if fisher:
                with torch.no_grad():
                    _, vjp_fn = torch.func.vjp(lambda inp: self.model(inp), x_prev)

                def G_mv(v_in):
                    return fisher_mat_vect_product(
                        x=x_prev,
                        p=current_p_safe.detach(),
                        v=v_in,
                        f=self.model,
                        vjp_fn=vjp_fn
                    )

                def A_fn(v_in):
                    res = G_mv(v_in) + lambda_ratio * v_in
                    if smoothing:
                        res += gamma_step * biharmonic_matvec(v_in)
                    return res

                import torchvision.transforms.functional as TF
                if smoothing and use_sobolev_preconditioner:
                    if lambda_ratio == 0:
                        M_inv = None
                    else:
                        def M_inv(v_in):
                            v_blurred = TF.gaussian_blur(v_in, kernel_size=[5, 5], sigma=[2.0, 2.0])
                            return v_blurred / lambda_ratio
                else:
                    if lambda_ratio == 0:
                        M_inv = None
                    else:
                        def M_inv(v_in):
                            return v_in / lambda_ratio

                if smoothing:
                    total_loss = grad_spher_loss + gamma_prior * biharmonic_matvec(x_prev)
                else:
                    total_loss = grad_spher_loss

                extra_dims = [1] * (total_loss.ndim - 1)
                total_loss = total_loss * active_mask.view(B, *extra_dims).float()

                v, n_iterations, _, _, _, _ = singular_psd_cg_solve(
                    A_fn=A_fn,
                    b=total_loss,
                    max_iter=self.cg_max_iters,
                    rtol=self.cg_rtol,
                    atol=self.cg_atol
                )

                cg_iter_history.append(n_iterations.cpu().tolist())

                with torch.no_grad():
                    Gv = G_mv(v)
                    fisher_norm_squared = (v * Gv).reshape(B, -1).sum(-1).clamp_min(1e-12)
                    v_norm_sq_batch = (v * v).reshape(B, -1).sum(-1).clamp_min(1e-12)
                    rayleigh_batch = fisher_norm_squared / v_norm_sq_batch

                    stats["fisher_rao_norm"].append(fisher_norm_squared.cpu().tolist())
                    stats["euclidean_norm_v"].append(v_norm_sq_batch.sqrt().cpu().tolist())
                    stats["rayleigh_quotient"].append(rayleigh_batch.cpu().tolist())

                    eta_kl = torch.sqrt(2.0 * kl_target / fisher_norm_squared)
                    eta_euclid = delta_euc / v_norm_sq_batch.sqrt()

                    avg_kl = eta_kl.mean().item()
                    avg_euc = eta_euclid.mean().item()
                    avg_max = eta_max_tensor.mean().item()

                    if avg_kl < avg_euc and avg_kl < avg_max:
                        stats["constraint_active"]["kl"] += 1
                    elif avg_euc < avg_kl and avg_euc < avg_max:
                        stats["constraint_active"]["euclid"] += 1
                    else:
                        stats["constraint_active"]["max"] += 1

                    eta = torch.minimum(
                        eta_max_tensor.expand_as(eta_euclid),
                        torch.minimum(eta_kl, eta_euclid)
                    )
            else:
                v = grad_spher_loss + gamma_prior * biharmonic_matvec(x_prev)
                eta = eta_max_tensor.expand(B)
                with torch.no_grad():
                    v_norm = (v * v).reshape(B, -1).sum(-1).sqrt()
                    stats["euclidean_norm_v"].append(v_norm.cpu().tolist())
                    stats["fisher_rao_norm"].append([0.0] * B)
                    stats["rayleigh_quotient"].append([0.0] * B)

            extra_dims = [1] * (x_prev.ndim - 1)
            mask_b = active_mask.view(B, *extra_dims).float()
            eta_b = eta.view(B, *extra_dims)

            with torch.no_grad():
                step = mask_b * (eta_b * v)
                step_norm_batch = step.reshape(B, -1).norm(dim=-1)
                stats["step_size_euclidean"].append(step_norm_batch.cpu().tolist())

                x_next = x_prev - step

            x_next_req = x_next.detach().requires_grad_(True)
            grad_next = self._gradient_target_logit(x_next_req)

            # Update total attributions and accumulate the local chunk
            delta_attr = 0.5 * (grad_next + grad_prev) * (-step)
            attributions += delta_attr
            chunk_attribution += delta_attr

            # --- THE FIX: Snapshot logic is safely inside the loop ---
            if save_x_path and (i % save_every == 0):
                x_path.append(_snapshot_tensor(x_next))
                saved_step_ids.append(i)
                stats["delta_attr_path"].append(_snapshot_tensor(-chunk_attribution))

                # Reset the chunk for the next interval
                chunk_attribution = torch.zeros_like(x)

            grad_prev = grad_next
            x_prev = x_next_req

        # --- LOOP ENDS HERE ---

        stats["total_steps"] = len(cg_iter_history)
        stats["cg_iters"] = cg_iter_history

        with torch.no_grad():
            logit_end = self.model_forward(x_prev, self.target_idx)
            attr_sum = attributions.reshape(B, -1).sum(-1)
            logit_diff = logit_end - logit_start
            delta = (attr_sum - logit_diff).abs()
            stats["completeness_delta"] = delta.cpu().tolist()

            variation = logit_diff.abs().clamp_min(1e-9)
            stats["completeness_relative_error"] = (delta / variation).cpu().tolist()

            current_p = F.softmax(self.model(x_prev), dim=-1).clamp(min=1e-10)
            target_p = torch.full_like(current_p, 1.0 / current_p.shape[-1])
            kl_endpoint = (current_p * (current_p.log() - target_p.log())).sum(-1)
            stats["endpoint_error_kl"] = kl_endpoint.cpu().tolist()

        if save_x_path:
            stats["x_path"] = x_path
            stats["x_path_step_ids"] = saved_step_ids

        # Inject final attribution into stats for easy plotting
        stats["attr"] = -attributions.detach().cpu()

        print(f"Completeness Delta: {delta.mean().item():.4f}")

        return -attributions, stats

    def attribute(
            self,
            x: torch.Tensor,
            kl_target: float = None,
            T_desired: int = None,
            p_target: torch.Tensor = None,
            fisher: bool = True,
            # FISHER PARAMETERS
            delta_euc: float = 10,
            eta_max: float = 10,
            use_sobolev_preconditioner: bool = True,
            lambda_ratio: float = 1,
            # SMOOTHING PARAMETERS
            smoothing: bool = False,
            gamma_step: float = 0.01,
            gamma_prior: float = 0.001,
    ) -> Tuple[torch.Tensor, Dict]:

        if kl_target is None and T_desired is None:
            raise ValueError("At least one of kl_target and T_desired should be set.")

        def masked_to_list(values: torch.Tensor, active_mask: torch.Tensor) -> List[float]:
            out = values.detach().clone().to(torch.float32)
            out[~active_mask] = float("nan")
            return out.cpu().tolist()

        eta_max_tensor = torch.tensor(eta_max, device=x.device, dtype=x.dtype)
        x_prev = x.clone().detach().requires_grad_(True)
        attributions = torch.zeros_like(x)

        stats = {
            "completeness_delta": [],
            "completeness_relative_error": [],
            "endpoint_error_kl": [],
            "tracking_error_spher": [],
            "fisher_rao_norm": [],
            "euclidean_norm_v": [],
            "rayleigh_quotient": [],
            "step_size_euclidean": [],
            "target_logit": [],
            "entropy": [],
            "cg_iters": [],
            "cg_residual_norm": [],
            "cg_relative_residual_norm": [],
            "active_mask": [],
            "n_steps_per_sample": None,
            "constraint_active": {"kl": 0, "euclid": 0, "max": 0},
            "total_steps": 0,
        }

        with torch.no_grad():
            logit_start = self.model_forward(x, self.target_idx)
            p_start = self.compute_softmax(x_prev)

        C = p_start.shape[-1]
        B = p_start.shape[0]
        s_alphas, mask, n_steps = compute_waypoints(
            p_start=p_start,
            kl_target=kl_target,
            num_classes=C,
            p_target=p_target,
            T_desired=T_desired,
        )

        stats["n_steps_per_sample"] = n_steps.cpu().tolist()

        max_T = int(n_steps.max().item())
        v = None
        grad_prev = self._gradient_target_logit(x_prev)

        # Waypoint zero is the starting prediction; target the next waypoint.
        for i in tqdm(range(1, max_T), desc="FisherRao Ascent"):
            s_alpha = s_alphas[:, i, :]
            active_mask = mask[:, i]

            if not active_mask.any():
                break

            stats["active_mask"].append(active_mask.cpu().tolist())

            current_logits = self.model(x_prev)
            current_p = F.softmax(current_logits, dim=-1)
            current_p_safe = current_p.clamp(min=1e-10)
            current_p_sqrt = current_p_safe.sqrt()

            # --------------------------------------------------
            # Logging: model state (masked for inactive samples)
            # --------------------------------------------------
            with torch.no_grad():
                current_loss = spherical_loss(current_p_sqrt, s_alpha)
                tgt_logit = current_logits.gather(1, self.target_idx.unsqueeze(1)).squeeze(1)
                entropy = -(current_p * (current_p + 1e-10).log()).sum(dim=-1)

                stats["tracking_error_spher"].append(masked_to_list(current_loss, active_mask))
                stats["target_logit"].append(masked_to_list(tgt_logit, active_mask))
                stats["entropy"].append(masked_to_list(entropy, active_mask))

            spher_loss = spherical_loss(current_p_sqrt, s_alpha)
            grad_spher_loss = compute_gradient(spher_loss, x_prev, retain_graph=False)

            if fisher:
                with torch.no_grad():
                    _, vjp_fn = torch.func.vjp(lambda inp: self.model(inp), x_prev)

                def G_mv(v_in):
                    return fisher_mat_vect_product(
                        x=x_prev,
                        p=current_p_safe.detach(),
                        v=v_in,
                        f=self.model,
                        vjp_fn=vjp_fn,
                    )

                def A_fn(v_in):
                    res = G_mv(v_in) + lambda_ratio * v_in
                    if smoothing:
                        res = res + gamma_step * biharmonic_matvec(v_in)
                    return res

                import torchvision.transforms.functional as TF

                if lambda_ratio == 0:
                    M_inv = None
                elif smoothing and use_sobolev_preconditioner:
                    def M_inv(v_in):
                        v_blurred = TF.gaussian_blur(v_in, kernel_size=[5, 5], sigma=[2.0, 2.0])
                        return v_blurred / lambda_ratio
                else:
                    def M_inv(v_in):
                        return v_in / lambda_ratio

                if smoothing:
                    total_loss = grad_spher_loss + gamma_prior * biharmonic_matvec(x_prev)
                else:
                    total_loss = grad_spher_loss

                extra_dims = [1] * (total_loss.ndim - 1)
                active_mask_b = active_mask.view(B, *extra_dims).float()

                # Inactive samples should contribute nothing to the linear system
                total_loss = total_loss * active_mask_b

                # Warm start only active samples; zero-out inactive stale directions
                if v is None or lambda_ratio == 0:
                    x0_cg = None
                else:
                    x0_cg = v.detach().clone() * active_mask_b

                v, n_iterations, residual_norm, relative_residual_norm = preconditionedCG_solve(
                    A_fn=A_fn,
                    b=total_loss,
                    M_op=M_inv,
                    x0=x0_cg,
                    max_iter=self.cg_max_iters,
                    rtol=self.cg_rtol,
                    atol=self.cg_atol,
                )

                # Keep inactive directions zeroed to avoid stale carryover
                v = v * active_mask_b

                stats["cg_iters"].append(masked_to_list(n_iterations.to(torch.float32), active_mask))
                stats["cg_residual_norm"].append(masked_to_list(residual_norm, active_mask))
                stats["cg_relative_residual_norm"].append(masked_to_list(relative_residual_norm, active_mask))

                with torch.no_grad():
                    Gv = G_mv(v)
                    fisher_norm_squared = (v * Gv).reshape(B, -1).sum(-1).clamp_min(1e-12)
                    v_norm_sq_batch = (v * v).reshape(B, -1).sum(-1).clamp_min(1e-12)
                    rayleigh_batch = fisher_norm_squared / v_norm_sq_batch

                    stats["fisher_rao_norm"].append(masked_to_list(fisher_norm_squared, active_mask))
                    stats["euclidean_norm_v"].append(masked_to_list(v_norm_sq_batch.sqrt(), active_mask))
                    stats["rayleigh_quotient"].append(masked_to_list(rayleigh_batch, active_mask))

                    eta_euclid = delta_euc / v_norm_sq_batch.sqrt()

                    if kl_target is not None:
                        eta_kl = torch.sqrt(2.0 * kl_target / fisher_norm_squared)
                    else:
                        eta_kl = torch.full_like(eta_euclid, float("inf"))

                    # Constraint telemetry: evaluate only on active samples
                    eta_kl_active = eta_kl[active_mask]
                    eta_euclid_active = eta_euclid[active_mask]
                    eta_max_active = eta_max_tensor.expand_as(eta_euclid)[active_mask]

                    avg_kl = eta_kl_active.mean().item()
                    avg_euc = eta_euclid_active.mean().item()
                    avg_max = eta_max_active.mean().item()

                    if avg_kl < avg_euc and avg_kl < avg_max:
                        stats["constraint_active"]["kl"] += 1
                    elif avg_euc < avg_kl and avg_euc < avg_max:
                        stats["constraint_active"]["euclid"] += 1
                    else:
                        stats["constraint_active"]["max"] += 1

                    eta = torch.minimum(
                        eta_max_tensor.expand_as(eta_euclid),
                        torch.minimum(eta_kl, eta_euclid),
                    )

            else:
                v = grad_spher_loss
                eta = eta_max_tensor.expand(B)

                with torch.no_grad():
                    v_norm = (v * v).reshape(B, -1).sum(-1).sqrt()

                    stats["fisher_rao_norm"].append(
                        masked_to_list(torch.zeros(B, device=x.device, dtype=x.dtype), active_mask))
                    stats["euclidean_norm_v"].append(masked_to_list(v_norm, active_mask))
                    stats["rayleigh_quotient"].append(
                        masked_to_list(torch.zeros(B, device=x.device, dtype=x.dtype), active_mask))

                    # No CG solve in Euclidean mode
                    nan_vec = torch.full((B,), float("nan"), device=x.device, dtype=x.dtype)
                    stats["cg_iters"].append(masked_to_list(nan_vec, active_mask))
                    stats["cg_residual_norm"].append(masked_to_list(nan_vec, active_mask))
                    stats["cg_relative_residual_norm"].append(masked_to_list(nan_vec, active_mask))

                    stats["constraint_active"]["max"] += 1

            # --------------------------------------------------
            # Update
            # --------------------------------------------------
            extra_dims = [1] * (x_prev.ndim - 1)
            mask_b = active_mask.view(B, *extra_dims).float()
            eta_b = eta.view(B, *extra_dims)

            with torch.no_grad():
                step = mask_b * (eta_b * v)
                step_norm_batch = step.reshape(B, -1).norm(dim=-1)
                stats["step_size_euclidean"].append(masked_to_list(step_norm_batch, active_mask))

                x_next = x_prev - step

            x_next_req = x_next.detach().requires_grad_(True)
            grad_next = self._gradient_target_logit(x_next_req)
            attributions += 0.5 * (grad_next + grad_prev) * (-step)

            grad_prev = grad_next
            x_prev = x_next_req

        stats["total_steps"] = len(stats["active_mask"])

        with torch.no_grad():
            logit_end = self.model_forward(x_prev, self.target_idx)
            attr_sum = attributions.reshape(B, -1).sum(-1)
            logit_diff = logit_end - logit_start
            delta = (attr_sum - logit_diff).abs()
            stats["completeness_delta"] = delta.cpu().tolist()

            variation = logit_diff.abs().clamp_min(1e-9)
            stats["completeness_relative_error"] = (delta / variation).cpu().tolist()

            current_p = F.softmax(self.model(x_prev), dim=-1).clamp(min=1e-10)
            target_p = torch.full_like(current_p, 1.0 / current_p.shape[-1])
            kl_endpoint = (current_p * (current_p.log() - target_p.log())).sum(-1)
            stats["endpoint_error_kl"] = kl_endpoint.cpu().tolist()

        print(f"Completeness Delta: {delta.mean().item():.4f}")

        return -attributions, stats


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transformation = load_model("resnet18", device=device)

    # Update paths to your actual images
    base_path = "./data/ImageNet/"
    images_paths_list = os.listdir(base_path)[4:5]
    img_tensor_list = []
    for image_path in images_paths_list:
        img_tensor_list.append(load_image(os.path.join(base_path, image_path), transformation, device=device))

    img_path1 = "./data/ImageNet/n01580077_jay.JPEG"
    img_path2 = "./data/ImageNet/n04070727_refrigerator.JPEG"
    #img1 = load_image(img_path1, transformation, device=device)
    img2 = load_image(img_path1, transformation, device=device)
    batch = torch.cat([img2], dim=0)

    with torch.no_grad():
        logits = model(batch)
        target_idx = logits.argmax(dim=1)
    print(f"Targets: {target_idx}")


    def model_forward(x, target):
        return model(x).gather(1, target[:, None]).squeeze(1)


    fr_ig = FisherRaoIntegratedGradients(model=model,
                                         model_forward=model_forward,
                                         target_idx=target_idx,
                                         cg_max_iters=20)

    print("Running Attribution...")
    attributions, stats = fr_ig.attribute_and_save(x=batch,
                                          kl_target=0.00030339,#0.00030339
                                          fisher=True,
                                          delta_euc=0.61337,
                                          eta_max=13.56315,
                                          use_sobolev_preconditioner=True,
                                          lambda_ratio=4.7707e-11,
                                          smoothing=True,
                                          gamma_step=0.0099739, # 0.0099739
                                          gamma_prior=0.00097495, #00097495,
                                           save_x_path=True,
                                           save_every=1,
                                           save_on_cpu=True,
                                           save_indices=[0],  # only save the first sample in the batch
                                          )

    print("Generating visualizations...")
    # visualize_attributions(batch, attributions)
    # plot_entropy_with_snapshot_arrows_and_diffs(
    #     stats,
    #     denormalize_fn=denormalize,
    #     image_indices=None, #[0, 4, 8, 12, -1],
    #     n_show_images=5,
    #     diff_mode="blur",
    #     blur_sigma=2,
    #     attr_map=attributions.cpu().squeeze(),
    #     save_path="trajectory_example.png",
    #     title="Fringe Trajectory - ResNet18"
    # )
    plot_entropy_with_snapshot_arrows_and_delta_attrs(
        stats,
        denormalize_fn=denormalize,
        #image_indices=None,  # [0, 4, 8, 12, -1],
        n_show_images=5,
        #diff_mode="blur",
        #blur_sigma=2,
        #attr_map=attributions.cpu().squeeze(),
        save_path="trajectory_example.png",
        title="Fringe Trajectory - ResNet18"
    )
    exit()
    plot_entropy_with_snapshot_arrows(
        stats,
        denormalize_fn=denormalize,
        n_show_images=6,
        save_path="entropy_with_all_arrows.png",
    )
    plot_fisher_rao_performance(
        stats,
        num_classes=1000,
        save_path="fisher_rao_summary.png",
        normalize_x=True,
        original_image=batch.cpu(),  # shape HxW or HxWxC
        attribution_map=attributions.cpu().squeeze().sum(0).abs(),  # shape HxW preferred
    )
    # x_sub, step_sub = select_path_subset(
    #     x_path=stats["x_path"],
    #     step_ids=stats.get("x_path_step_ids"),
    #     indices=[0, 2, 5, -1],
    # )
    #
    # plot_intermediate_steps(
    #     x_path=x_sub,
    #     step_ids=step_sub,
    #     denormalize_fn=denormalize,
    #     max_cols=4,
    #     save_path="trajectory_subset.png",
    # )
    # exit()
