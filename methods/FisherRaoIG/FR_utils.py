import contextlib
import math
import weakref
from typing import Optional, Tuple, Union, Callable
import torch.nn.functional as F
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torchvision.transforms.functional import gaussian_blur

# Cache whether a model contains MHA-like modules
_HAS_MHA_CACHE = weakref.WeakKeyDictionary()

# Cache the working JVP mode per (model class, train/eval, device, dtype, shape)
_JVP_MODE_CACHE = {}


def _has_mha(model: torch.nn.Module) -> bool:
    cached = _HAS_MHA_CACHE.get(model)
    if cached is None:
        cached = any(isinstance(m, torch.nn.MultiheadAttention) for m in model.modules())
        _HAS_MHA_CACHE[model] = cached
    return cached


def _jvp_cache_key(model: torch.nn.Module, x: torch.Tensor):
    return (
        model.__class__,
        model.training,
        x.device.type,
        x.device.index,
        str(x.dtype),
        tuple(x.shape),
    )


@contextlib.contextmanager
def _jvp_mode_context(mode: str):
    old_mha = None

    if mode in ("func_no_fastpath", "func_math_sdpa"):
        old_mha = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)

    try:
        if mode == "func_math_sdpa":
            with sdpa_kernel([SDPBackend.MATH]):
                yield
        else:
            yield
    finally:
        if old_mha is not None:
            torch.backends.mha.set_fastpath_enabled(old_mha)


def _run_jvp(fn: Callable, x: torch.Tensor, v: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "autograd":
        _, jv = torch.autograd.functional.jvp(
            fn, x, v, create_graph=False, strict=False
        )
        return jv

    with _jvp_mode_context(mode):
        _, jv = torch.func.jvp(fn, (x,), (v,))
        return jv


def fast_jvp(fn: Callable, x: torch.Tensor, v: torch.Tensor, model: Optional[torch.nn.Module] = None) -> torch.Tensor:
    """
    Try the fastest compatible JVP mode first, then degrade only if needed.
    The successful mode is cached.
    """
    model_for_cache = model if model is not None else getattr(fn, "__self__", None)

    # If we cannot identify a model instance, do a simple generic fallback chain.
    if model_for_cache is None:
        for mode in ("func", "autograd"):
            try:
                return _run_jvp(fn, x, v, mode)
            except NotImplementedError:
                pass
        raise RuntimeError("No compatible JVP mode found.")

    key = _jvp_cache_key(model_for_cache, x)
    cached_mode = _JVP_MODE_CACHE.get(key)

    candidates = []
    if cached_mode is not None:
        candidates.append(cached_mode)

    # Fastest-first ordering
    candidates.append("func")

    if _has_mha(model_for_cache):
        candidates.append("func_no_fastpath")
        candidates.append("func_math_sdpa")

    candidates.append("autograd")

    # Deduplicate while preserving order
    seen = set()
    candidates = [m for m in candidates if not (m in seen or seen.add(m))]

    last_err = None
    for mode in candidates:
        try:
            jv = _run_jvp(fn, x, v, mode)
            _JVP_MODE_CACHE[key] = mode
            return jv
        except NotImplementedError as e:
            last_err = e
            continue

    raise RuntimeError(f"No compatible JVP mode found. Last error: {last_err}")

def spherical_loss(p_sqrt: torch.Tensor,
                   s_t: torch) -> torch.Tensor:

    return 1.0 - torch.einsum("bc, bc -> b", p_sqrt, s_t)

def compute_gradient(y: torch.Tensor,
                     x: torch.Tensor,
                     retain_graph: bool = False) -> torch.Tensor:
    (grad_y,) = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        retain_graph=retain_graph
    )
    return grad_y

def fisher_mat_vect_product(
    x: torch.Tensor,
    p: torch.Tensor,
    v: Union[torch.Tensor, Tuple[torch.Tensor]],
    f: torch.nn.Module,
    vjp_fn: Optional[Callable] = None,
):
    """
    Computes Fisher-vector product.
    Reuses vjp_fn when provided.
    """

    if isinstance(v, tuple):
        if len(v) != 1:
            raise ValueError(f"Expected a 1-tuple for v, got length {len(v)}")
        v = v[0]

    # Fastest compatible JVP, cached after first success
    Jv = fast_jvp(f, x, v, model=f)

    p = p.detach().clamp_min(1e-12)
    pJv = p * Jv
    S_Jv = pJv - p * pJv.sum(dim=1, keepdim=True)

    if vjp_fn is None:
        _, vjp_fn = torch.func.vjp(f, x)

    (JT_S_Jv,) = vjp_fn(S_Jv)
    return JT_S_Jv

@torch.no_grad
def compute_waypoints(
        p_start: torch.Tensor,
        kl_target: float = None,
        num_classes: int = None,
        p_target: torch.Tensor = None,
        safety_buffer: float = 0.1,
        eps: float = 1e-8,
        T_desired: int = None
):
    """
    Computes a Fisher–Rao (slerp-on-sqrt-simplex) geodesic path from p_start to p_target.

    Returns:
        s_list: [B, T, C] padded path in sqrt-space
        mask:   [B, T] True where timestep is valid per sample
        n_steps:[B] number of valid steps per sample
    """
    assert kl_target > 0.0
    assert kl_target > 0.0
    assert 0.0 <= safety_buffer < 1.0

    if p_start.ndim == 1:
        p_start = p_start.unsqueeze(0)

    if p_target is None and num_classes is None:
        raise ValueError("Define at least one of num_classes or p_target")

    device = p_start.device
    dtype = p_start.dtype

    # --- 1. Normalize + Clamp ---
    p_start = p_start.clamp_min(eps)
    p_start = p_start / p_start.sum(dim=-1, keepdim=True).clamp_min(eps)

    if p_target is not None:
        if p_target.ndim == 1:
            p_target = p_target.unsqueeze(0)
        if p_target.shape[0] == 1 and p_start.shape[0] > 1:
            p_target = p_target.expand(p_start.shape[0], -1)
        p_target = p_target.to(device=device, dtype=dtype).clamp_min(eps)
    else:
        if num_classes is None:
            raise ValueError("num_classes must be provided if p_target is None")
        p_target = torch.full_like(p_start, 1.0 / float(num_classes))

    p_start_sqrt = torch.sqrt(p_start)  # [B, C]
    p_target_sqrt = torch.sqrt(p_target)  # [B, C]

    # --- 2. Fisher–Rao Angle ---
    # Dot product of sqrt densities = cos(theta)
    rho = (p_start_sqrt * p_target_sqrt).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)  # [B]
    theta = torch.acos(rho)  # [B]
    total_arc_length = 2.0 * theta  # [B]

    if kl_target is None and T_desired is not None:

        kl_target = 0.5 * (total_arc_length/T_desired) ** 2

    # --- 3. Compute Step Counts ---
    # D_fr approx sqrt(2 * KL).
    step_size = math.sqrt(2.0 * kl_target)

    # Logic Fix: n_steps must be (segments + 1) to cover the distance
    num_intervals = torch.ceil((total_arc_length / step_size) * (1.0 + safety_buffer)).long()
    n_steps = num_intervals + 1
    n_steps = n_steps.clamp(min=2)  # Ensure at least start and end points exist


    # --- 4. Generate Timesteps [0, 1] ---
    B = n_steps.shape[0]
    T = int(n_steps.max().item())

    j = torch.arange(T, device=device, dtype=torch.long)
    mask = j[None, :] < n_steps[:, None]  # [B, T]

    denom = (n_steps - 1).clamp(min=1)  # [B]
    ts = j[None, :].to(dtype) / denom[:, None].to(dtype)  # [B, T]
    ts = ts.masked_fill(~mask, 0.0)

    # --- 5. Broadcast Helpers (CRITICAL FIX) ---
    # We need to reshape variables to [B, T, 1] or [B, 1, 1] to broadcast correctly

    # Calculate extra dims for features (e.g. if input is [B, C], extra_dims=1)
    extra_dims = p_start_sqrt.ndim - 1
    extra_ones = (1,) * extra_dims  # (1,)

    # ts: [B, T] -> [B, T, 1]
    ts_b = ts.view(B, T, *extra_ones)

    # theta: [B] -> [B, 1, 1] (Was [B, 1] previously, causing the crash)
    theta_b = theta.view(B, 1, *extra_ones)

    sin_theta = torch.sin(theta)
    # sin_theta: [B] -> [B, 1, 1]
    sin_theta_b = sin_theta.clamp_min(eps).view(B, 1, *extra_ones)

    # --- 6. Slerp Weights ---
    # Safe division for small angles
    small_angle = (theta < 1e-6).view(B, 1, *extra_ones)

    # Use safe denominator (1.0) where angle is small to avoid NaN
    safe_sin = torch.where(small_angle, torch.ones_like(sin_theta_b), sin_theta_b)

    # Standard Slerp formula
    w_1 = torch.sin((1.0 - ts_b) * theta_b) / safe_sin
    w_2 = torch.sin(ts_b * theta_b) / safe_sin

    # Linear fallback for small angles (limit sin(x)/x -> 1)
    w_1 = torch.where(small_angle, 1.0 - ts_b, w_1)
    w_2 = torch.where(small_angle, ts_b, w_2)

    # --- 7. Build Path ---
    # w_1: [B, T, 1], p_start_sqrt: [B, C] -> p_start_sqrt[:, None, ...]: [B, 1, C]
    # Result: [B, T, C]
    s_list = w_1 * p_start_sqrt[:, None, ...] + w_2 * p_target_sqrt[:, None, ...]

    # Zero out padded steps
    mask_b = mask.view(B, T, *extra_ones)
    s_list = s_list.masked_fill(~mask_b, 0.0)

    return s_list, mask, n_steps


import torch
from typing import Optional, Callable, Union, Tuple


import torch
from typing import Callable, Optional, Union, Tuple


@torch.no_grad()
def preconditionedCG_solve(
    A_fn: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    M_op: Optional[Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]] = None,
    x0: Optional[torch.Tensor] = None,
    max_iter: int = 20,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solves Ax = b using Batched Preconditioned Conjugate Gradient.

    Returns:
        x:
            Solution tensor, same shape as b.
        iterations:
            Tensor of shape (B,) with the number of iterations used per sample.
        residual_norm:
            Tensor of shape (B,) with final ||Ax - b||_2 per sample.
        relative_residual_norm:
            Tensor of shape (B,) with final ||Ax - b||_2 / (||b||_2 + eps) per sample.

    Notes:
        - This does NOT return the true solution error ||x - x*||, because x* is unknown.
        - residual_norm is the standard computable proxy for approximation quality.
    """
    if torch.isnan(b).any() or torch.isinf(b).any():
        raise ValueError("Input 'b' contains NaNs or Infs.")

    B = b.shape[0]
    flat_shape = (B, -1)
    view_shape = (B,) + (1,) * (b.ndim - 1)

    # Preconditioner
    if M_op is None:
        precond = lambda x: x
    elif isinstance(M_op, torch.Tensor):
        precond = lambda x: M_op * x
    else:
        precond = M_op

    # Initialization
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b.clone() if x0 is None else b - A_fn(x)
    z = precond(r)
    p = z.clone()

    b_norm = b.reshape(flat_shape).norm(dim=1)
    tol = torch.maximum(
        torch.full_like(b_norm, float(atol)),
        b_norm * float(rtol),
    )
    tol_sq = tol ** 2

    rz_old = (r * z).reshape(flat_shape).sum(dim=1)
    r_sq = (r * r).reshape(flat_shape).sum(dim=1)
    mask_active = r_sq > tol_sq

    iterations = torch.zeros(B, device=b.device, dtype=torch.long)

    for _ in range(max_iter):
        if not mask_active.any():
            break

        iterations[mask_active] += 1

        Ap = A_fn(p)
        pAp = (p * Ap).reshape(flat_shape).sum(dim=1)

        curvature_safe = pAp > eps
        pAp_denom = torch.where(curvature_safe, pAp, torch.ones_like(pAp))
        alpha = rz_old / pAp_denom

        mask_update = mask_active & curvature_safe
        alpha_v = torch.where(mask_update, alpha, torch.zeros_like(alpha)).view(view_shape)

        x = x + alpha_v * p
        r = r - alpha_v * Ap

        r_sq = (r * r).reshape(flat_shape).sum(dim=1)
        mask_active = (r_sq > tol_sq) & curvature_safe

        if not mask_active.any():
            break

        z = precond(r)
        rz_new = (r * z).reshape(flat_shape).sum(dim=1)

        denom_safe = torch.where(rz_old.abs() > eps, rz_old, torch.ones_like(rz_old))
        beta = rz_new / denom_safe
        beta_v = torch.where(mask_active, beta, torch.zeros_like(beta)).view(view_shape)

        p = z + beta_v * p
        rz_old = rz_new

    # Final approximation-error proxy
    final_residual = b - A_fn(x)
    residual_norm = final_residual.reshape(flat_shape).norm(dim=1)
    relative_residual_norm = residual_norm / b_norm.clamp_min(eps)

    return x, iterations, residual_norm, relative_residual_norm
import torch
from typing import Callable, Tuple


@torch.no_grad()
def singular_psd_cg_solve(
    A_fn: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    max_iter: int = 50,
    atol: float = 1e-6,
    rtol: float = 1e-4,
    eps: float = 1e-12,
    residual_refresh: int = 10,
) -> Tuple[
    torch.Tensor,  # x
    torch.Tensor,  # iterations
    torch.Tensor,  # residual_norm
    torch.Tensor,  # relative_residual_norm
    torch.Tensor,  # converged_mask
    torch.Tensor,  # breakdown_mask
]:
    """
    Batched CG for symmetric PSD possibly singular systems A x = b.

    Intended use:
        - A is symmetric positive semidefinite
        - system is consistent: b in Im(A)
        - zero-init branch to recover the minimum-norm solution x = A^dagger b

    Important:
        - no preconditioner
        - no warm start
        - if breakdown happens with large residual, the system is likely inconsistent
          or A_fn is not behaving like a symmetric PSD operator numerically.
    """
    if torch.isnan(b).any() or torch.isinf(b).any():
        raise ValueError("Input b contains NaNs or Infs.")

    B = b.shape[0]
    flat_shape = (B, -1)
    view_shape = (B,) + (1,) * (b.ndim - 1)
    dtype_eps = torch.finfo(b.dtype).eps

    def batch_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x * y).reshape(flat_shape).sum(dim=1)

    def batch_norm(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(flat_shape).norm(dim=1)

    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()

    b_norm = batch_norm(b)
    tol = torch.maximum(
        torch.full_like(b_norm, float(atol)),
        b_norm * float(rtol),
    )
    tol_sq = tol * tol

    rr = batch_dot(r, r)
    converged = rr <= tol_sq
    breakdown = torch.zeros(B, device=b.device, dtype=torch.bool)
    active = ~converged

    iterations = torch.zeros(B, device=b.device, dtype=torch.long)

    for k in range(max_iter):
        if not active.any():
            break

        Ap = A_fn(p)
        pAp = batch_dot(p, Ap)

        p_norm = batch_norm(p)
        Ap_norm = batch_norm(Ap)

        # Relative curvature threshold
        curv_tol = 10.0 * torch.maximum(
            torch.full_like(pAp, eps),
            (p_norm * Ap_norm) * dtype_eps
        )
        small_curv = pAp <= curv_tol

        # If curvature vanishes, distinguish true convergence from breakdown.
        suspicious = active & small_curv
        if suspicious.any():
            r_true = b - A_fn(x)
            rr_true = batch_dot(r_true, r_true)
            truly_converged = suspicious & (rr_true <= tol_sq)

            converged = converged | truly_converged
            active = active & (~truly_converged)

            bad = suspicious & (~truly_converged)
            if bad.any():
                breakdown = breakdown | bad
                active = active & (~bad)

            if not active.any():
                break

            # Keep residual synchronized for the remaining active samples
            r = torch.where(active.view(view_shape), r_true, r)
            rr = torch.where(active, rr_true, rr)

        alpha = torch.zeros_like(rr)
        alpha[active] = rr[active] / pAp[active]
        alpha_v = alpha.view(view_shape)

        x = x + alpha_v * p

        if residual_refresh > 0 and ((k + 1) % residual_refresh == 0):
            r = b - A_fn(x)
        else:
            r = r - alpha_v * Ap

        rr_new = batch_dot(r, r)
        iterations[active] += 1

        newly_converged = active & (rr_new <= tol_sq)
        converged = converged | newly_converged
        active = active & (~newly_converged)

        if not active.any():
            rr = rr_new
            break

        beta = torch.zeros_like(rr_new)
        safe_rr = rr.clamp_min(eps)
        beta[active] = rr_new[active] / safe_rr[active]
        beta_v = beta.view(view_shape)

        p = r + beta_v * p
        rr = rr_new

    final_residual = b - A_fn(x)
    residual_norm = batch_norm(final_residual)
    relative_residual_norm = residual_norm / b_norm.clamp_min(eps)

    return x, iterations, residual_norm, relative_residual_norm, converged, breakdown
@torch.no_grad
def estimate_grad_sq_precond(grad: torch.Tensor,
                              gauss_blur_kernel_diag: int = 5,
                              gauss_blur_sigma_diag: float = 2.0, ) -> torch.Tensor:
    """
    Returns the smoothed Empirical Fisher Diagonal (g^2).
    Removes normalization to respect the true scale of the problem.
    """
    # 1. Raw Empirical Fisher
    g_sq = grad ** 2

    # 2. Spatial Smoothing
    # Essential to bridge the gaps between pixels (consistency)
    # Using the boolean check to handle non-image tensors if necessary
    if g_sq.ndim == 4:
        g_sq_smooth = gaussian_blur(g_sq,
                                  kernel_size=gauss_blur_kernel_diag,
                                  sigma=gauss_blur_sigma_diag)
    else:
        g_sq_smooth = g_sq

    return g_sq_smooth

_LAPLACIAN_KERNEL = torch.tensor([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]], dtype=torch.float32)

@torch.no_grad
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


@torch.no_grad
@torch.compile
def biharmonic_matvec(v: torch.Tensor, pad_mode: str = "reflect") -> torch.Tensor:
    return laplacian_matvec(laplacian_matvec(v, pad_mode=pad_mode), pad_mode=pad_mode)


def batched_nystrom_preconditioner(G_mv, x, rank=20, damping=1e-3):
    """
    Builds a batched rank-k approximation of G, then constructs M^{-1} via Woodbury.
    x shape: [B, C, H, W]
    """
    B = x.shape[0]
    D = x.shape[1] * x.shape[2] * x.shape[3]  # Features per image

    # 1. Draw random Gaussian sketch per image in the batch
    # Shape: [B, D, rank]
    Omega = torch.randn(B, D, rank, device=x.device)

    # Orthonormalize each sketch in the batch
    Omega, _ = torch.linalg.qr(Omega)

    # 2. Apply G to each column of the sketch
    Y = torch.zeros_like(Omega)
    for j in range(rank):
        # Extract the j-th column for all batches: [B, D] -> [B, C, H, W]
        v_j = Omega[:, :, j].view_as(x)

        # Apply Fisher-vector product
        # G_mv must handle [B, C, H, W] and return [B, C, H, W]
        gv = G_mv(v_j)

        # Store result: [B, D]
        Y[:, :, j] = gv.view(B, D)

    # 3. Nyström approximation: G ≈ Y (Ω^T Y)^{-1} Y^T
    # B_mat = Ω^T Y  -> Shape: [B, rank, rank]
    B_mat = torch.bmm(Omega.transpose(1, 2), Y)

    # Symmetrize to prevent complex eigenvalues from floating point math
    B_mat = (B_mat + B_mat.transpose(1, 2)) / 2.0

    # Batched Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(B_mat)  # eigvals: [B, rank], eigvecs: [B, rank, rank]
    eigvals = eigvals.clamp_min(0)

    # 4. Compute U for Woodbury
    # U = Y @ eigvecs / sqrt(eigvals + eps) -> Shape: [B, D, rank]
    inv_sqrt_eigvals = 1.0 / (eigvals + 1e-8).sqrt()
    # Apply diag scaling via broadcasting
    U = torch.bmm(Y, eigvecs) * inv_sqrt_eigvals.unsqueeze(1)

    # 5. The fast batched application function
    def M_inv_apply(v):
        """v shape: [B, C, H, W]"""
        v_flat = v.view(B, D, 1)  # [B, D, 1]

        inv_damp = 1.0 / damping
        # correction shape: [B, rank, 1]
        correction = ((1.0 / (eigvals + damping)) - inv_damp).unsqueeze(-1)

        # UTv = U^T v -> [B, rank, 1]
        UTv = torch.bmm(U.transpose(1, 2), v_flat)

        # U @ (correction * UTv) -> [B, D, 1]
        woodbury_term = torch.bmm(U, correction * UTv)

        res_flat = (inv_damp * v_flat) + woodbury_term
        return res_flat.view_as(v)

    return M_inv_apply
