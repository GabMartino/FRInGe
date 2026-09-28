from typing import Dict, Union, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
# -------------------------
# Small helpers
# -------------------------

def _to_1d_trace(arr, b_idx=0, reduce="auto"):
    """
    Convert common logged shapes to a 1D per-step trace.

    Supported shapes:
      - (T,)                      -> returned
      - (T, B)                    -> arr[:, b_idx]
      - (T, B, ...)               -> reduced over non-(T,B) dims then arr[:, b_idx]
      - list of scalars / list of arrays -> stacked to ndarray, then processed

    reduce:
      - "auto": mean over remaining dims
      - "mean": mean over remaining dims
      - "median": median over remaining dims
      - "max": max over remaining dims
    """
    if arr is None:
        return None

    if isinstance(arr, list):
        if len(arr) == 0:
            return None
        try:
            arr = np.stack(arr, axis=0)
        except Exception:
            # list of python floats
            arr = np.asarray(arr)

    arr = np.asarray(arr)
    if arr.size == 0:
        return None

    if arr.ndim == 1:
        return arr

    # If 2D -> (T,B)
    if arr.ndim == 2:
        b = 0 if b_idx >= arr.shape[1] else b_idx
        return arr[:, b]

    # If >=3D -> (T,B,...) reduce over last dims
    reducer = reduce
    if reducer == "auto":
        reducer = "mean"

    tail = arr.reshape(arr.shape[0], arr.shape[1], -1)
    if reducer == "mean":
        tail = tail.mean(axis=2)
    elif reducer == "median":
        tail = np.median(tail, axis=2)
    elif reducer == "max":
        tail = tail.max(axis=2)
    else:
        raise ValueError(f"Unknown reduce={reduce}")

    b = 0 if b_idx >= tail.shape[1] else b_idx
    return tail[:, b]


def _get_trace(info, key, b_idx=0, reduce="auto"):
    if key not in info:
        return None
    return _to_1d_trace(info[key], b_idx=b_idx, reduce=reduce)


def _default_steps(*traces):
    """Return steps based on the first non-None trace length."""
    for tr in traces:
        if tr is not None:
            return np.arange(len(tr))
    return None


def _maybe_log_axis(ax, y, thresh=10.0):
    """Use log-y if series has large dynamic range or large spikes."""
    if y is None:
        return
    y = np.asarray(y)
    if y.size == 0:
        return
    y_pos = y[y > 0]
    if y_pos.size == 0:
        return
    ratio = y_pos.max() / max(y_pos.min(), 1e-30)
    if ratio > 1e3 or (y.max() > thresh and y.min() >= 0):
        ax.set_yscale("log")


# -------------------------
# Main dashboard
# -------------------------

def plot_all_diagnostics(info, b_idx=0, kl_target=None, title_prefix=""):
    """
    Dashboard for Fisher-Rao IG runs.

    Fixes vs previous version:
      - robust trace extraction for (T,B), (T,B,...) and list logs
      - safe handling when keys are missing
      - consistent step axis selection
      - kl_target displayed (if provided)
      - meta displayed if info["__meta__"] exists (no crashing)

    Expected keys (optional):
      spherical_loss, entropy, delta,
      fisher_norm_sq, eta_step,
      n_cg_steps,
      relative_residual, fraction_D_vs_G,
      alpha_t,
      diag_min/med/max, D_min/med/max, step_l2/step_linf, theta_err (if you log them)
    """
    print(f"--- Plotting Diagnostics for Batch Index {b_idx} ---")

    meta = info.get("__meta__", None)
    if meta is not None:
        print("--- Run meta ---")
        for k, v in meta.items():
            print(f"{k}: {v}")

    plot_optimization_dynamics(info, b_idx=b_idx, kl_target=kl_target, title_prefix=title_prefix)
    plot_trust_region_and_solver(info, b_idx=b_idx, title_prefix=title_prefix)
    plot_schedules(info, b_idx=b_idx, title_prefix=title_prefix)
    plot_optional_conditioning(info, b_idx=b_idx, title_prefix=title_prefix)


# -------------------------
# Panels
# -------------------------

def plot_optimization_dynamics(info, b_idx=0, kl_target=None, title_prefix=""):
    loss = _get_trace(info, "spherical_loss", b_idx)
    entropy = _get_trace(info, "entropy", b_idx)
    delta = _get_trace(info, "delta", b_idx)

    if loss is None and entropy is None and delta is None:
        return

    steps = _default_steps(loss, entropy, delta)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: objective + entropy
    if loss is not None:
        ax1.plot(steps, loss, lw=2, label="Spherical Loss")
        ax1.fill_between(steps, 0, loss, alpha=0.08)
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

    if entropy is not None:
        ax1b = ax1.twinx()
        ax1b.plot(steps, entropy, lw=1.8, linestyle="--", label="Entropy")
        ax1b.set_ylabel("Entropy")
        ax1b.grid(False)

    txt = "Objective & Uncertainty"
    if kl_target is not None:
        txt += f"  (kl_target={kl_target})"
    ax1.set_title(txt)

    # Panel 2: completeness / integration error
    if delta is not None:
        ax2.plot(steps, delta, lw=2, label="Completeness / Riemann Error")
        ax2.axhline(0, lw=0.7, color="black", alpha=0.6)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Error")
        ax2.grid(True, alpha=0.3)
        _maybe_log_axis(ax2, delta, thresh=10.0)
        ax2.set_title("Completeness Check")

    if title_prefix:
        fig.suptitle(title_prefix, y=1.02)

    plt.tight_layout()
    plt.show()


def plot_trust_region_and_solver(info, b_idx=0, title_prefix=""):
    fisher_norm = _get_trace(info, "fisher_norm_sq", b_idx)
    eta = _get_trace(info, "eta_step", b_idx)
    n_cg = _get_trace(info, "n_cg_steps", b_idx)
    rel_res = _get_trace(info, "relative_residual", b_idx)
    frac_D = _get_trace(info, "fraction_D_vs_G", b_idx)

    # require at least something
    if fisher_norm is None and eta is None and n_cg is None and rel_res is None and frac_D is None:
        return

    steps = _default_steps(fisher_norm, eta, n_cg, rel_res, frac_D)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax11, ax12, ax21, ax22 = axes.flatten()

    # (1) Fisher norm + eta
    if fisher_norm is not None:
        ax11.semilogy(steps, fisher_norm, lw=1.8, label=r"$v^\top A v$")
        ax11.set_ylabel("Fisher/Quadratic Form")
        ax11.grid(True, alpha=0.3)
    if eta is not None:
        ax11b = ax11.twinx()
        ax11b.plot(steps, eta, lw=1.8, linestyle="--", label="Eta")
        ax11b.set_ylabel("Eta")
        ax11b.grid(False)
    ax11.set_title("Trust Region Dynamics")
    ax11.set_xlabel("Step")

    # (2) PCG/CG steps
    if n_cg is not None:
        ax12.plot(steps, n_cg, lw=1.8)
        ax12.set_title("Actual CG/PCG Steps")
        ax12.set_xlabel("Step")
        ax12.set_ylabel("Iterations")
        ax12.grid(True, alpha=0.3)

    # (3) fraction D vs G
    if frac_D is not None:
        ax21.plot(steps, frac_D, lw=1.8)
        ax21.set_title("Fraction D vs (G + D)")
        ax21.set_xlabel("Step")
        ax21.set_ylabel("v^T D v / v^T (G+D) v")
        ax21.set_ylim(-0.05, 1.05)
        ax21.grid(True, alpha=0.3)

    # (4) relative residual
    if rel_res is not None:
        ax22.plot(steps, rel_res, lw=1.8)
        ax22.set_title("Relative Linear Solve Residual")
        ax22.set_xlabel("Step")
        ax22.set_ylabel("||b - A x|| / ||b||")
        ax22.grid(True, alpha=0.3)
        _maybe_log_axis(ax22, rel_res, thresh=0.1)

    if title_prefix:
        fig.suptitle(title_prefix, y=1.02)

    plt.tight_layout()
    plt.show()


def plot_schedules(info, b_idx=0, title_prefix=""):
    alpha = _get_trace(info, "alpha_t", b_idx)
    if alpha is None:
        return

    steps = np.arange(len(alpha))
    plt.figure(figsize=(10, 4))
    plt.plot(steps, alpha, lw=2)
    plt.xlabel("Step")
    plt.ylabel("Alpha")
    plt.title("Annealing Schedule (Alpha)")
    plt.grid(True, alpha=0.3)
    if title_prefix:
        plt.suptitle(title_prefix, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_optional_conditioning(info, b_idx=0, title_prefix=""):
    """
    Plots additional logs if present:
      - diag_min/med/max
      - D_min/med/max
      - D_cond_proxy
      - step_l2 / step_linf
      - theta_err
    """
    diag_min = _get_trace(info, "diag_min", b_idx)
    diag_med = _get_trace(info, "diag_med", b_idx)
    diag_max = _get_trace(info, "diag_max", b_idx)

    D_min = _get_trace(info, "D_min", b_idx)
    D_med = _get_trace(info, "D_med", b_idx)
    D_max = _get_trace(info, "D_max", b_idx)
    D_cond = _get_trace(info, "D_cond_proxy", b_idx)

    step_l2 = _get_trace(info, "step_l2", b_idx)
    step_linf = _get_trace(info, "step_linf", b_idx)
    theta_err = _get_trace(info, "theta_err", b_idx)

    any_present = any(x is not None for x in [
        diag_min, diag_med, diag_max, D_min, D_med, D_max, D_cond, step_l2, step_linf, theta_err
    ])
    if not any_present:
        return

    # layout depends on what is available
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax11, ax12, ax21, ax22 = axes.flatten()

    steps = _default_steps(diag_min, D_min, D_cond, step_l2, theta_err)

    # Diag stats
    if diag_min is not None or diag_med is not None or diag_max is not None:
        if diag_min is not None: ax11.semilogy(steps, diag_min, lw=1.6, label="diag_min")
        if diag_med is not None: ax11.semilogy(steps, diag_med, lw=1.6, label="diag_med")
        if diag_max is not None: ax11.semilogy(steps, diag_max, lw=1.6, label="diag_max")
        ax11.set_title("diag_shape Stats")
        ax11.set_xlabel("Step")
        ax11.grid(True, alpha=0.3)
        ax11.legend()

    # D stats
    if D_min is not None or D_med is not None or D_max is not None:
        if D_min is not None: ax12.semilogy(steps, D_min, lw=1.6, label="D_min")
        if D_med is not None: ax12.semilogy(steps, D_med, lw=1.6, label="D_med")
        if D_max is not None: ax12.semilogy(steps, D_max, lw=1.6, label="D_max")
        ax12.set_title("Damping D Stats")
        ax12.set_xlabel("Step")
        ax12.grid(True, alpha=0.3)
        ax12.legend()

    # Condition proxy
    if D_cond is not None:
        ax21.semilogy(steps, D_cond, lw=1.8)
        ax21.set_title("D Condition Proxy (max/min)")
        ax21.set_xlabel("Step")
        ax21.grid(True, alpha=0.3)

    # Step norms + theta error
    if step_l2 is not None or step_linf is not None or theta_err is not None:
        if step_l2 is not None:
            ax22.plot(steps, step_l2, lw=1.6, label="step_l2")
        if step_linf is not None:
            ax22.plot(steps, step_linf, lw=1.6, label="step_linf")
        if theta_err is not None:
            ax22b = ax22.twinx()
            ax22b.plot(steps, theta_err, lw=1.6, linestyle="--", label="theta_err")
            ax22b.set_ylabel("Theta error (rad)")
        ax22.set_title("Euclidean Step Norms / Geodesic Error")
        ax22.set_xlabel("Step")
        ax22.grid(True, alpha=0.3)
        ax22.legend(loc="upper left")

    if title_prefix:
        fig.suptitle(title_prefix, y=1.02)

    plt.tight_layout()
    plt.show()


# -------------------------
# Attribution visualization utilities
# -------------------------

def _normalize_heatmap(hm, robust_pct=99.0, eps=1e-12):
    """
    Robustly normalize a heatmap to [0,1] using percentile scaling.
    """
    hm = np.asarray(hm, dtype=np.float32)
    hm = np.maximum(hm, 0.0)
    denom = np.percentile(hm, robust_pct) + eps
    return np.clip(hm / denom, 0.0, 1.0)


def visualize_input_and_heatmap(
    x,                  # torch Tensor or numpy, shape (C,H,W) or (H,W,C)
    attributions,        # torch Tensor or numpy, shape (C,H,W) or (H,W)
    denormalize_fn=None, # callable: torch/numpy -> torch/numpy image in [0,1]
    b_idx=0,
    reduce="abs_sum",    # how to reduce channels: "abs_sum", "sum", "l2", "max"
    cmap="inferno",
    robust_pct=99.0,
    alpha=0.45,
    title=None,
):
    """
    Shows:
      - input image
      - heatmap
      - overlay (input * heatmap or alpha-blend overlay)
      - side-by-side with colorbar

    Notes:
      - If x/attr are batched (B,C,H,W), pick b_idx.
      - If you want "input * heatmap", set mode="multiply" below (default).
    """
    # --- to numpy, select batch if needed
    def _to_np(t):
        if hasattr(t, "detach"):
            t = t.detach().cpu().float().numpy()
        return np.asarray(t)

    x_np = _to_np(x)
    a_np = _to_np(attributions)

    if x_np.ndim == 4:  # B,C,H,W or B,H,W,C
        x_np = x_np[b_idx]
    if a_np.ndim == 4:
        a_np = a_np[b_idx]

    # --- channels to HWC for image
    if x_np.ndim == 3 and x_np.shape[0] in (1, 3):  # CHW
        x_chw = x_np
        if denormalize_fn is not None:
            # allow denorm on CHW
            x_chw = _to_np(denormalize_fn(x_chw))
        img = np.transpose(x_chw, (1, 2, 0))
    elif x_np.ndim == 3 and x_np.shape[-1] in (1, 3):  # HWC
        img = x_np
        if denormalize_fn is not None:
            img = _to_np(denormalize_fn(img))
    else:
        raise ValueError(f"Unsupported x shape: {x_np.shape}")

    img = np.clip(img, 0.0, 1.0)

    # --- attribution to 2D heatmap
    if a_np.ndim == 3 and a_np.shape[0] in (1, 3):  # CHW
        if reduce == "abs_sum":
            hm = np.abs(a_np).sum(axis=0)
        elif reduce == "sum":
            hm = a_np.sum(axis=0)
        elif reduce == "l2":
            hm = np.sqrt((a_np * a_np).sum(axis=0))
        elif reduce == "max":
            hm = np.abs(a_np).max(axis=0)
        else:
            raise ValueError(f"Unknown reduce={reduce}")
    elif a_np.ndim == 2:
        hm = a_np
    else:
        raise ValueError(f"Unsupported attributions shape: {a_np.shape}")

    hm_n = _normalize_heatmap(hm, robust_pct=robust_pct)

    # --- overlay variants
    # multiply overlay: input * (1 + hm_n * alpha) clipped
    overlay_mul = np.clip(img * (hm_n[..., None]), 0.0, 1.0)

    # alpha-blend overlay using colormap
    cm = plt.get_cmap(cmap)
    hm_rgb = cm(hm_n)[..., :3]
    overlay_blend = np.clip((1 - alpha) * img + alpha * hm_rgb, 0.0, 1.0)

    # --- plot
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    ax0, ax1, ax2, ax3 = axes

    ax0.imshow(img)
    ax0.set_title("Input")
    ax0.axis("off")

    im1 = ax1.imshow(hm_n, cmap=cmap, vmin=0.0, vmax=1.0)
    ax1.set_title("Heatmap (norm)")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2.imshow(overlay_mul)
    ax2.set_title("Input * Heatmap")
    ax2.axis("off")

    ax3.imshow(overlay_blend)
    ax3.set_title("Alpha-blend Overlay")
    ax3.axis("off")

    if title:
        fig.suptitle(title, y=1.05)

    plt.tight_layout()
    plt.show()


def visualize_attr_triplet(
    x,
    attributions,
    denormalize_fn=None,
    b_idx=0,
    cmap="inferno",
    robust_pct=99.0,
    alpha=0.45,
    title_prefix="",
):
    """
    Convenience wrapper: shows input, heatmap, and overlays.
    Intended to be called inside your main loop every N steps.
    """
    visualize_input_and_heatmap(
        x=x,
        attributions=attributions,
        denormalize_fn=denormalize_fn,
        b_idx=b_idx,
        reduce="abs_sum",
        cmap=cmap,
        robust_pct=robust_pct,
        alpha=alpha,
        title=title_prefix,
    )



def visualize_geometry2( riemann, euclidean, steps, riemann_reg = None, entropy = None):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Riemannian Distance (The "Cost" of the Linear Step)
    color = 'tab:blue'
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Riemannian Distance (Prob Change)', color=color, fontweight='bold')
    ax1.plot(steps, riemann, color=color, label='Riemannian Cost (IG)')
    # ax1.plot(steps, info_ig["logit"])
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(bottom=0, top=np.max(riemann) * 1.5)
    if riemann_reg is not None:
        ax1.plot(steps, riemann_reg, color="green", label='Riemannian Regularized')
    ax2 = ax1.twinx()

    # Plot Euclidean Distance (Constant for IG)
    color = 'tab:red'
    ax2.set_ylabel('Euclidean Distance (Pixel Change)', color=color, fontweight='bold')
    ax2.plot(steps, euclidean, color=color, linestyle='--', label='Euclidean Step (IG)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=0, top=np.max(euclidean) * 1.5)  # Scale to see the line clearly

    plt.title(
        'Geometry of a Linear Path (Integrated Gradients)')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    #plt.legend()

    plt.savefig("ig_geometry_analysis.png", dpi=300)
    print("Plot saved to 'ig_geometry_analysis.png'")

    plt.show()

# --- ICML-ish typography defaults (call once in your script/notebook) ---
def set_icml_matplotlib_style(
    base_fontsize=13,     # increase/decrease globally
    tick_fontsize=11,
    legend_fontsize=11,
    title_fontsize=14,
    line_width=2.0
):
    mpl.rcParams.update({
        # Typography
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_fontsize,

        # Axes / ticks
        "axes.titlesize": title_fontsize,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
        "axes.linewidth": 1.0,

        # Lines / markers
        "lines.linewidth": line_width,
        "lines.markersize": 5,

        # Legend
        "legend.fontsize": legend_fontsize,
        "legend.frameon": True,
        "legend.framealpha": 0.9,

        # Figure
        "figure.dpi": 300,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def visualize_geometry(
        riemann: Union[List, np.ndarray],
        euclidean: Union[List, np.ndarray],
        steps: Optional[Union[List, np.ndarray]] = None,
        riemann_reg: Optional[Union[List, np.ndarray]] = None,
        entropy: Optional[Union[List, np.ndarray]] = None,
        euclidean_kind: str = "delta",  # "delta" | "cumulative" | "auto"
        title: str = "",
        figsize: Tuple[float, float] = (6.6, 3.8),
        entropy_axis_offset: float = 1.14,
        grid_alpha: float = 0.25,
        normalize_progress: bool = True,
) -> plt.Figure:
    """
    Geometry plot (Riemannian vs Euclidean step) optionally with Entropy on a third axis.

    IMPORTANT:
      - This function does NOT touch matplotlib rcParams.
      - Call `set_icml_matplotlib_style(...)` ONCE outside (e.g. at script start)
        to control fonts/linewidths/DPI consistently.
    """

    # -------------------- helpers -------------------- #
    def as_1d(x, name: str) -> np.ndarray:
        a = np.asarray(x, dtype=float)
        if a.ndim == 0:
            a = a.reshape(1)
        return a.squeeze()

    def align(*arrs):
        lens = [len(a) for a in arrs if a is not None]
        T = min(lens) if lens else 0
        out = []
        for a in arrs:
            out.append(None if a is None else a[:T])
        return T, out

    def robust_bubble_sizes(delta: np.ndarray, s_min: float = 18.0, s_max: float = 160.0) -> np.ndarray:
        if len(delta) == 0:
            return np.array([])
        lo, hi = np.percentile(delta, [5, 95])
        if hi - lo < 1e-12:
            return np.full_like(delta, (s_min + s_max) / 2.0)
        z = np.clip((delta - lo) / (hi - lo), 0.0, 1.0)
        return s_min + (s_max - s_min) * z

    # -------------------- data prep -------------------- #
    riemann = as_1d(riemann, "riemann")
    euclidean = as_1d(euclidean, "euclidean")
    riemann_reg = None if riemann_reg is None else as_1d(riemann_reg, "riemann_reg")
    entropy = None if entropy is None else np.clip(as_1d(entropy, "entropy"), 0.0, 1.05)

    # X axis
    if steps is None:
        x_vals = np.linspace(0.0, 1.0, len(riemann))
        xlabel = r"Path progress $\alpha \in [0,1]$"
    else:
        x_vals = as_1d(steps, "steps")
        _, (x_vals, riemann, euclidean, riemann_reg, entropy) = align(
            x_vals, riemann, euclidean, riemann_reg, entropy
        )
        if normalize_progress:
            ptp = np.ptp(x_vals)
            if ptp < 1e-12:
                x_vals = np.linspace(0.0, 1.0, len(riemann))
            else:
                x_vals = (x_vals - x_vals.min()) / (ptp + 1e-12)
            xlabel = r"Path progress $\alpha \in [0,1]$"
        else:
            xlabel = "Step"

    # Euclidean interpretation
    if euclidean_kind == "auto":
        is_cum = (len(euclidean) > 1) and (abs(euclidean[0]) < 1e-8) and np.all(np.diff(euclidean) >= -1e-10)
    elif euclidean_kind == "cumulative":
        is_cum = True
    elif euclidean_kind == "delta":
        is_cum = False
    else:
        raise ValueError("euclidean_kind must be 'delta', 'cumulative', or 'auto'.")

    if is_cum:
        delta_euc = np.diff(euclidean, prepend=euclidean[0])
    else:
        delta_euc = euclidean.copy()

    delta_euc = np.clip(delta_euc, 0.0, None)
    bubble_sizes = robust_bubble_sizes(delta_euc)

    # -------------------- plotting -------------------- #
    fig, ax1 = plt.subplots(figsize=figsize)

    c_riem = "tab:blue"
    c_euc = "tab:red"
    c_ent = "tab:purple"

    # Axis 1: Riemannian
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Riemannian distance", color=c_riem, fontweight="bold")
    (l1,) = ax1.plot(x_vals, riemann, color=c_riem, label="Riemannian distance")

    lines = [l1]
    if riemann_reg is not None:
        (l1r,) = ax1.plot(x_vals, riemann_reg, color="teal", alpha=0.75, label="Riemannian (reg)")
        lines.append(l1r)

    ax1.tick_params(axis="y", labelcolor=c_riem)
    ax1.grid(True, which="major", axis="both", alpha=grid_alpha, linestyle="--")

    r_min, r_max = float(np.min(riemann)), float(np.max(riemann)) if len(riemann) else (0.0, 1.0)
    r_pad = 0.35 * (r_max - r_min + 1e-12)
    ax1.set_ylim(0.0, r_max + r_pad)

    # Axis 2: Euclidean
    ax2 = ax1.twinx()
    ax2.set_ylabel("Per-step Euclidean increment", color=c_euc, fontweight="bold")
    (l2,) = ax2.plot(x_vals, delta_euc, color=c_euc, linestyle="--", alpha=0.75, label="Euclidean step")
    ax2.scatter(x_vals, delta_euc, s=bubble_sizes, color=c_euc, alpha=0.35, edgecolors="none", zorder=10)
    ax2.tick_params(axis="y", labelcolor=c_euc)
    lines.append(l2)

    e_min = float(np.min(delta_euc)) if len(delta_euc) else 0.0
    e_max = float(np.max(delta_euc)) if len(delta_euc) else 1.0
    e_pad = 0.35 * (e_max - e_min + 1e-12)
    ax2.set_ylim(0.0, e_max + e_pad)

    # Axis 3: Entropy (offset)
    if entropy is not None:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", entropy_axis_offset))
        ax3.set_frame_on(True)
        ax3.patch.set_visible(False)
        ax3.set_ylabel("Normalized Entropy", color=c_ent, fontweight="bold")
        (l3,) = ax3.plot(x_vals, entropy, color=c_ent, linestyle=":", alpha=0.95, label="Entropy")
        ax3.tick_params(axis="y", labelcolor=c_ent)
        ax3.set_ylim(0.0, 1.)
        lines.append(l3)

        # make room for the offset entropy axis
        fig.subplots_adjust(right=0.74)
    else:
        fig.subplots_adjust(right=0.84)

    # -------------------- LEGEND & TITLE FIXES -------------------- #

    # 1. Title Strategy: Move it UP to avoid collision with the legend
    if title:
        # y=1.20 pushes the title safely above the plot area but below the legend text
        # (Assuming legend anchor is at 1.14 bottom-aligned)
        ax1.set_title(title, y=1.20, zorder=1)

    # 2. Legend Strategy: Absolute Z-Order domination and forced opacity
    labels = [ln.get_label() for ln in lines]

    leg = ax3.legend(lines, labels,
                     loc="upper left",
                     bbox_to_anchor=(-0.04, 1.14),
                     fancybox=False,
                     shadow=False,
                     ncol=1,
                     frameon=True)  # <--- Essential: Enables the background box

    # 3. Manual Style Override (The "Nuclear" Option)
    # We access the patch object of the legend directly
    frame = leg.get_frame()
    # frame.set_color("white")
    # frame.set_facecolor('white')  # <--- Sets background to white
    frame.set_alpha(1.0)  # <--- Sets background OPACITY to 100% (Not 0.0!)
    # frame.set_edgecolor('black')  # Optional: Adds a thin black border
    # frame.set_linewidth(0.5)
    #
    # # 4. Force Legend to be the top-most layer
    # leg.set_zorder(1000)

    fig.tight_layout()
    fig.show()


def plot_gradient_analysis(info: Dict[str, np.ndarray], title: str = "Gradient Analysis"):
    """
    Replicates Figure 3 from the Guided IG paper:
    Compares the useful Directional Derivative vs the potentially noisy Total Gradient Norm.
    """
    grad_norm = info["total_grad_norm"].squeeze()
    dir_deriv = info["directional_derivative"].squeeze()

    # Normalize steps to [0, 1] alpha
    steps = np.linspace(0, 1, len(grad_norm))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Directional Derivative (The Signal)
    # The area under this curve is the Total Attribution.
    ax1.plot(steps, dir_deriv, color='tab:blue', linewidth=2)
    ax1.fill_between(steps, dir_deriv, alpha=0.3, color='tab:blue')
    ax1.set_title("(b) Directional Derivative (Attribution contribution)")
    ax1.set_xlabel(r"Path progress $\alpha$")
    ax1.set_ylabel("Signal Magnitude")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Total Gradient Norm (The Sensitivity)
    # High values here when (b) is near zero indicate "Noise" (orthogonality).
    ax2.plot(steps, grad_norm, color='tab:red', linewidth=2)
    ax2.set_title("(c) Total Gradient Magnitude (L2 Norm)")
    ax2.set_xlabel(r"Path progress $\alpha$")
    ax2.set_ylabel("Gradient L2 Norm")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()