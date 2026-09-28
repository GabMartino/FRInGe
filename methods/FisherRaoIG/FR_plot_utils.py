import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from common.utils import denormalize


def plot_fisher_rao_performance(stats: dict, num_classes: int = 1000, save_path: str = None):
    """
    Generates aggregated ribbon plots (Mean +/- Range) for Fisher-Rao telemetry.
    Includes Logits and Entropy visualization.
    """

    def to_np(key):
        if key not in stats or len(stats[key]) == 0: return None
        data = stats[key]
        if isinstance(data[0], list):
            return np.array(data)
        else:
            return np.array(data)[:, None]

    # --- Extract Data ---
    track_err = to_np("tracking_error_spher")
    fisher_norm = to_np("fisher_rao_norm")
    rayleigh = to_np("rayleigh_quotient")
    step_size = to_np("step_size_euclidean")
    cg_iters = to_np("cg_iters")
    tgt_logit = to_np("target_logit")  # <--- NEW
    entropy = to_np("entropy")  # <--- NEW

    if track_err is None:
        print("No telemetry data found.")
        return

    steps = np.arange(track_err.shape[0])

    # --- Setup Plot Grid (2 Rows x 4 Columns) ---
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle('Fisher-Rao Optimization Dynamics & Path Evolution', fontsize=18)

    def plot_ribbon(ax, data, title, y_label, color, log_scale=False, y_limit=None):
        if data is None: return
        mean = np.mean(data, axis=1)
        min_val = np.min(data, axis=1)
        max_val = np.max(data, axis=1)

        ax.plot(steps, mean, color=color, linewidth=2, label='Batch Mean')
        ax.fill_between(steps, min_val, max_val, color=color, alpha=0.2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Step (t)')
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

        if log_scale:
            # only enable log-scale if there is at least one positive value
            # and avoid 0/negative bounds in autoscaling.
            if np.any(max_val > 0):
                # clamp nonpositive to a tiny epsilon for plotting purposes
                eps = 1e-12
                # re-plot with clamped values to avoid log(0)
                ax.cla()
                mean_c = np.maximum(mean, eps)
                min_c = np.maximum(min_val, eps)
                max_c = np.maximum(max_val, eps)
                ax.plot(steps, mean_c, color=color, linewidth=2, label='Batch Mean')
                ax.fill_between(steps, min_c, max_c, color=color, alpha=0.2)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel('Step (t)')
                ax.set_ylabel(y_label)
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
            else:
                # fallback: linear if nothing positive
                pass

        if y_limit: ax.set_ylim(y_limit)

    # --- Row 1: Optimization Health ---

    # 1. Tracking Error
    norm_factor = np.pi
    plot_ribbon(axes[0, 0], track_err / norm_factor, "Geodesic Tracking Error", "Error (% of $\pi$)", '#E24A33',
                y_limit=(0, None))

    # 2. Rayleigh Quotient
    plot_ribbon(axes[0, 1], rayleigh, "Rayleigh Quotient", r"Ratio $\frac{v^T G v}{v^T v}$ (Log)", '#348ABD',
                log_scale=True)

    # 3. Fisher Norm
    plot_ribbon(axes[0, 2], fisher_norm, "Fisher-Rao Norm", r"$v^T G v$", '#FBC15E', log_scale=True)

    # 4. Solver Complexity
    plot_ribbon(axes[0, 3], cg_iters, "Solver Complexity", "CG Iterations", '#8EBA42')

    # --- Row 2: Physical Path Evolution ---

    # 5. Step Size
    plot_ribbon(axes[1, 0], step_size, "Realized Step Size", r"$||\delta x||_2$", '#988ED5')

    # 6. Target Logit (NEW) - Should decrease
    plot_ribbon(axes[1, 1], tgt_logit, "Target Class Logit", "Logit Value", '#D62728')  # Red

    # 7. Entropy (NEW) - Should increase
    # Normalize by Log(C) -> [0, 1] range
    max_ent = np.log(num_classes)
    norm_entropy = entropy / max_ent if max_ent > 0 else entropy
    plot_ribbon(axes[1, 2], norm_entropy, "Predictive Entropy", "% of Max Entropy", '#1F77B4',
                y_limit=(0, 1.05))  # Blue

    # 8. Summary Text
    ax_text = axes[1, 3]
    ax_text.axis('off')

    avg_rel_err = np.mean(stats.get("completeness_relative_error", [0])) * 100
    constraints = stats.get("constraint_active", {})
    total = sum(constraints.values()) if constraints else 1
    if total == 0:
        total = 1
    text_str = (
        f"--- Batch Summary ---\n\n"
        f"Completeness Error:\n"
        f"  {avg_rel_err:.2f}% (Rel)\n\n"
        f"Active Constraints:\n"
        f"  Fisher (KL): {constraints.get('kl', 0) / total * 100:.1f}%\n"
        f"  Euclid (L2): {constraints.get('euclid', 0) / total * 100:.1f}%\n"
        f"  Hard Cap:    {constraints.get('max', 0) / total * 100:.1f}%\n"
    )
    ax_text.text(0.1, 0.5, text_str, fontsize=14, family='monospace', va='center')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    if save_path: plt.savefig(save_path, dpi=300)
    plt.show()


def plot_fisher_rao_performance(stats: dict, num_classes: int = 1000, save_path: str = None, normalize_x: bool = True):
    """
    Generates aggregated ribbon plots (Mean +/- Range) for Fisher-Rao telemetry.
    Interpolates variable-length batch trajectories to a normalized 0-100% progress scale.
    """

    def to_np(key):
        if key not in stats or len(stats[key]) == 0: return None
        data = stats[key]
        if isinstance(data[0], list):
            return np.array(data)
        else:
            return np.array(data)[:, None]

    # --- Extract Raw Data ---
    track_err = to_np("tracking_error_spher")
    fisher_norm = to_np("fisher_rao_norm")
    rayleigh = to_np("rayleigh_quotient")
    step_size = to_np("step_size_euclidean")
    cg_iters = to_np("cg_iters")
    tgt_logit = to_np("target_logit")
    entropy = to_np("entropy")

    if track_err is None:
        print("No telemetry data found.")
        return

    # --- 1. Fix the X-Axis (Trajectory Interpolation) ---
    def normalize_trajectory_batch(data, num_points=100):
        """Interpolates variable-length batch trajectories onto a 0-100% grid."""
        if data is None or not normalize_x: return data

        steps, batch_size = data.shape
        normalized_data = np.zeros((num_points, batch_size))
        x_target = np.linspace(0, 1, num_points)

        for i in range(batch_size):
            # Isolate valid data (assuming padding uses np.nan)
            # If your code pads with 0s at the end, change this condition to filter trailing 0s.
            valid_idx = ~np.isnan(data[:, i])
            y_valid = data[valid_idx, i]

            if len(y_valid) == 0:
                normalized_data[:, i] = np.nan
                continue

            x_original = np.linspace(0, 1, len(y_valid))
            normalized_data[:, i] = np.interp(x_target, x_original, y_valid)

        return normalized_data

    # Apply x-axis normalization
    track_err = normalize_trajectory_batch(track_err)
    fisher_norm = normalize_trajectory_batch(fisher_norm)
    rayleigh = normalize_trajectory_batch(rayleigh)
    step_size = normalize_trajectory_batch(step_size)
    cg_iters = normalize_trajectory_batch(cg_iters)
    tgt_logit = normalize_trajectory_batch(tgt_logit)
    entropy = normalize_trajectory_batch(entropy)

    steps = np.linspace(0, 100, track_err.shape[0]) if normalize_x else np.arange(track_err.shape[0])
    x_label = "Integration Progress (%)" if normalize_x else "Step (t)"

    # --- 2. Fix the Logit Data (Min-Max Normalization) ---
    if tgt_logit is not None:
        # Normalize each trajectory independently to [0, 1] relative to its own min/max
        col_min = np.nanmin(tgt_logit, axis=0, keepdims=True)
        col_max = np.nanmax(tgt_logit, axis=0, keepdims=True)
        # Avoid division by zero if min == max
        tgt_logit = (tgt_logit - col_min) / (col_max - col_min + 1e-8)

    # --- Setup Plot Grid ---
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle('Fisher-Rao Optimization Dynamics & Path Evolution', fontsize=18)

    def plot_ribbon(ax, data, title, y_label, color, log_scale=False, y_limit=None):
        if data is None: return
        # Use nanmean to ignore padded values if they exist
        mean = np.nanmean(data, axis=1)
        min_val = np.nanmin(data, axis=1)
        max_val = np.nanmax(data, axis=1)

        ax.plot(steps, mean, color=color, linewidth=2, label='Batch Mean')
        ax.fill_between(steps, min_val, max_val, color=color, alpha=0.2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

        if log_scale and np.any(max_val > 0):
            eps = 1e-12
            ax.cla()
            mean_c = np.maximum(mean, eps)
            min_c = np.maximum(min_val, eps)
            max_c = np.maximum(max_val, eps)
            ax.plot(steps, mean_c, color=color, linewidth=2, label='Batch Mean')
            ax.fill_between(steps, min_c, max_c, color=color, alpha=0.2)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        if y_limit: ax.set_ylim(y_limit)

    # --- Row 1: Optimization Health ---
    plot_ribbon(axes[0, 0], track_err / np.pi, "Geodesic Tracking Error", "Error (% of $\pi$)", '#E24A33',
                y_limit=(0, None))
    plot_ribbon(axes[0, 1], rayleigh, "Rayleigh Quotient", r"Ratio $\frac{v^T G v}{v^T v}$ (Log)", '#348ABD',
                log_scale=True)
    plot_ribbon(axes[0, 2], fisher_norm, "Fisher-Rao Norm", r"$v^T G v$", '#FBC15E', log_scale=True)

    # CG Iterations retained here as requested
    plot_ribbon(axes[0, 3], cg_iters, "Solver Complexity", "CG Iterations", '#8EBA42')

    # --- Row 2: Physical Path Evolution ---
    plot_ribbon(axes[1, 0], step_size, "Realized Step Size", r"$||\delta x||_2$", '#988ED5')

    # Normalized Target Logit
    plot_ribbon(axes[1, 1], tgt_logit, "Normalized Target Logit", "Relative Logit Scale [0,1]", '#D62728')

    # Normalized Entropy
    max_ent = np.log(num_classes)
    norm_entropy = entropy / max_ent if max_ent > 0 else entropy
    plot_ribbon(axes[1, 2], norm_entropy, "Predictive Entropy", "% of Max Entropy", '#1F77B4', y_limit=(0, 1.05))

    # --- Summary Text ---
    ax_text = axes[1, 3]
    ax_text.axis('off')

    avg_rel_err = np.mean(stats.get("completeness_relative_error", [0])) * 100
    constraints = stats.get("constraint_active", {})
    total = sum(constraints.values()) if constraints else 1
    if total == 0: total = 1

    text_str = (
        f"--- Batch Summary ---\n\n"
        f"Completeness Error:\n"
        f"  {avg_rel_err:.2f}% (Rel)\n\n"
        f"Active Constraints:\n"
        f"  Fisher (KL): {constraints.get('kl', 0) / total * 100:.1f}%\n"
        f"  Euclid (L2): {constraints.get('euclid', 0) / total * 100:.1f}%\n"
        f"  Hard Cap:    {constraints.get('max', 0) / total * 100:.1f}%\n"
    )
    ax_text.text(0.1, 0.5, text_str, fontsize=14, family='monospace', va='center')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    if save_path: plt.savefig(save_path, dpi=300)
    plt.show()


import numpy as np
from matplotlib import pyplot as plt


import numpy as np
from matplotlib import pyplot as plt


def configure_icml_plot_style(use_tex=False):
    """
    Approximate ICML paper typography in matplotlib.
    Set use_tex=True only if a working LaTeX installation is available.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,

        "figure.titlesize": 12,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "text.usetex": use_tex,
    })


def plot_fisher_rao_performance(
    stats: dict,
    num_classes: int = 1000,
    save_path: str = None,
    normalize_x: bool = True,
    original_image=None,
    attribution_map=None,
    attribution_cmap="seismic",
    use_tex: bool = False,
):
    """
    Generates aggregated ribbon plots (Mean +/- Range) for Fisher-Rao telemetry.
    Interpolates variable-length batch trajectories to a normalized 0-100% progress scale.

    Parameters
    ----------
    stats : dict
        Telemetry dictionary.
    num_classes : int
        Number of classes, used to normalize entropy.
    save_path : str, optional
        If provided, saves the figure.
    normalize_x : bool
        Whether to normalize trajectory progress to [0, 100].
    original_image : np.ndarray, optional
        Representative input image to display. Expected shape HxW or HxWxC.
    attribution_map : np.ndarray, optional
        Representative attribution map to display. Expected shape HxW or HxWxC.
    attribution_cmap : str
        Colormap used for the attribution map.
    use_tex : bool
        Whether to use LaTeX text rendering for an even closer paper style.
    """
    configure_icml_plot_style(use_tex=use_tex)

    def to_np(key):
        if key not in stats or len(stats[key]) == 0:
            return None
        data = stats[key]
        if isinstance(data[0], list):
            return np.array(data)
        return np.array(data)[:, None]

    # --- Extract Raw Data ---
    track_err = to_np("tracking_error_spher")
    fisher_norm = to_np("fisher_rao_norm")
    step_size = to_np("step_size_euclidean")
    tgt_logit = to_np("target_logit")
    entropy = to_np("entropy")

    if track_err is None:
        print("No telemetry data found.")
        return

    # --- 1. Fix the X-Axis (Trajectory Interpolation) ---
    def normalize_trajectory_batch(data, num_points=100):
        """Interpolates variable-length batch trajectories onto a 0-100% grid."""
        if data is None or not normalize_x:
            return data

        steps, batch_size = data.shape
        normalized_data = np.zeros((num_points, batch_size))
        x_target = np.linspace(0, 1, num_points)

        for i in range(batch_size):
            valid_idx = ~np.isnan(data[:, i])
            y_valid = data[valid_idx, i]

            if len(y_valid) == 0:
                normalized_data[:, i] = np.nan
                continue

            x_original = np.linspace(0, 1, len(y_valid))
            normalized_data[:, i] = np.interp(x_target, x_original, y_valid)

        return normalized_data

    # Apply x-axis normalization
    track_err = normalize_trajectory_batch(track_err)
    fisher_norm = normalize_trajectory_batch(fisher_norm)
    step_size = normalize_trajectory_batch(step_size)
    tgt_logit = normalize_trajectory_batch(tgt_logit)
    entropy = normalize_trajectory_batch(entropy)

    steps = np.linspace(0, 100, track_err.shape[0]) if normalize_x else np.arange(track_err.shape[0])
    x_label = "Integration Progress (%)" if normalize_x else "Step (t)"

    # --- 2. Fix the Logit Data (Min-Max Normalization) ---
    if tgt_logit is not None:
        col_min = np.nanmin(tgt_logit, axis=0, keepdims=True)
        col_max = np.nanmax(tgt_logit, axis=0, keepdims=True)
        tgt_logit = (tgt_logit - col_min) / (col_max - col_min + 1e-8)

    # --- Setup Plot Grid ---
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle("Fisher-Rao Optimization Dynamics & Attribution Overview")

    def plot_ribbon(ax, data, title, y_label, color, log_scale=False, y_limit=None):
        if data is None:
            ax.axis("off")
            return

        mean = np.nanmean(data, axis=1)
        min_val = np.nanmin(data, axis=1)
        max_val = np.nanmax(data, axis=1)

        if log_scale and np.any(max_val > 0):
            eps = 1e-12
            mean = np.maximum(mean, eps)
            min_val = np.maximum(min_val, eps)
            max_val = np.maximum(max_val, eps)
            ax.set_yscale("log")

        ax.plot(steps, mean, color=color, linewidth=2, label="Batch Mean")
        ax.fill_between(steps, min_val, max_val, color=color, alpha=0.2)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

        if y_limit is not None:
            ax.set_ylim(y_limit)

    def plot_image(ax, image, title, cmap=None):
        if image is None:
            ax.axis("off")
            ax.set_title(title, fontweight="bold")
            return

        image = np.asarray(image)

        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]

        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontweight="bold")
        ax.axis("off")

    # --- Row 1 ---
    plot_ribbon(
        axes[0, 0],
        track_err / np.pi,
        "Geodesic Tracking Error",
        r"Error (% of $\pi$)",
        "#E24A33",
        y_limit=(0, None),
    )

    plot_ribbon(
        axes[0, 1],
        fisher_norm,
        "Fisher-Rao Norm",
        r"$v^T G v$",
        "#FBC15E",
        log_scale=True,
    )

    plot_image(
        axes[0, 2],
        denormalize(original_image.squeeze(0)),
        "Original Image",
        cmap="gray" if original_image is not None and np.asarray(original_image).ndim == 2 else None,
    )

    if attribution_map is not None:
        attr = np.asarray(attribution_map)

        if attr.ndim == 3 and attr.shape[-1] > 1:
            attr = np.mean(attr, axis=-1)
        elif attr.ndim == 3 and attr.shape[-1] == 1:
            attr = attr[..., 0]

        vmax = np.max(np.abs(attr)) + 1e-12
        axes[0, 3].imshow(attr, cmap=attribution_cmap, vmin=-vmax, vmax=vmax)
        axes[0, 3].set_title("Attribution Map", fontweight="bold")
        axes[0, 3].axis("off")
    else:
        axes[0, 3].axis("off")
        axes[0, 3].set_title("Attribution Map", fontweight="bold")

    # --- Row 2 ---
    plot_ribbon(
        axes[1, 0],
        step_size,
        "Realized Step Size",
        r"$||\delta x||_2$",
        "#988ED5",
    )

    plot_ribbon(
        axes[1, 1],
        tgt_logit,
        "Normalized Target Logit",
        "Relative Logit Scale [0,1]",
        "#D62728",
    )

    max_ent = np.log(num_classes)
    norm_entropy = entropy / max_ent if (entropy is not None and max_ent > 0) else entropy
    plot_ribbon(
        axes[1, 2],
        norm_entropy,
        "Predictive Entropy",
        "% of Max Entropy",
        "#1F77B4",
        y_limit=(0, 1.05),
    )

    # --- Summary Text ---
    ax_text = axes[1, 3]
    ax_text.axis("off")

    avg_rel_err = np.mean(stats.get("completeness_relative_error", [0])) * 100
    constraints = stats.get("constraint_active", {})
    total = sum(constraints.values()) if constraints else 1
    if total == 0:
        total = 1

    text_str = (
        f"--- Batch Summary ---\n\n"
        f"Completeness Error:\n"
        f"  {avg_rel_err:.2f}% (Rel)\n\n"
        f"Active Constraints:\n"
        f"  Fisher (KL): {constraints.get('kl', 0) / total * 100:.1f}%\n"
        f"  Euclid (L2): {constraints.get('euclid', 0) / total * 100:.1f}%\n"
        f"  Hard Cap:    {constraints.get('max', 0) / total * 100:.1f}%\n"
    )
    ax_text.text(0.1, 0.5, text_str, fontsize=11, family="monospace", va="center")

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    if save_path:
        plt.savefig(save_path, dpi=330, bbox_inches="tight")

    plt.show()


import math
import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_intermediate_steps(
    x_path,
    step_ids=None,
    denormalize_fn=None,
    max_cols=5,
    figsize_per_panel=(3.0, 3.0),
    cmap=None,
    save_path=None,
):
    """
    Plot saved intermediate states x_k.

    Parameters
    ----------
    x_path : list[Tensor or ndarray]
        Typically stats["x_path"].
        Each element can be:
          - [C,H,W]
          - [1,C,H,W]  (first element will be taken)
          - [H,W]
          - [H,W,C]
    step_ids : list[int], optional
        Typically stats["x_path_step_ids"].
    denormalize_fn : callable, optional
        Function applied to each image before plotting.
        Example: denormalize_fn = denormalize
    max_cols : int
        Maximum number of columns in the grid.
    figsize_per_panel : tuple
        Size multiplier per subplot.
    cmap : str, optional
        Colormap for grayscale images.
    save_path : str, optional
        If provided, save the figure.
    """

    def to_display_image(x):
        if torch.is_tensor(x):
            x = x.detach().cpu()

        x = np.array(x)

        # If batch dimension is present, take the first example
        if x.ndim == 4:
            x = x[0]

        if denormalize_fn is not None:
            x = denormalize_fn(torch.tensor(x)) if not isinstance(x, torch.Tensor) else denormalize_fn(x)

        # CHW -> HWC
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = np.transpose(x, (1, 2, 0))

        # Squeeze grayscale channel
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]

        return x

    n = len(x_path)
    if n == 0:
        print("Empty x_path.")
        return

    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
        squeeze=False,
    )
    axes = axes.flatten()

    for i, xk in enumerate(x_path):
        ax = axes[i]
        img = to_display_image(xk)

        if img.ndim == 2:
            ax.imshow(img, cmap=cmap or "gray")
        else:
            ax.imshow(np.clip(img, 0, 1) if np.issubdtype(img.dtype, np.floating) else img)

        title = f"$x_{{{i}}}$"
        if step_ids is not None:
            title += f"\n(step {step_ids[i]})"

        ax.set_title(title)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


import numpy as np
import torch
from typing import List, Union

import numpy as np

def select_path_subset(x_path, step_ids=None, indices=None, n_show=None):
    n = len(x_path)
    if n == 0:
        return [], [] if step_ids is not None else None

    if indices is not None and n_show is not None:
        raise ValueError("Use either indices or n_show, not both.")

    if indices is None:
        if n_show is None or n_show >= n:
            indices = list(range(n))
        else:
            indices = np.linspace(0, n - 1, n_show)
            indices = np.unique(np.round(indices).astype(int)).tolist()
    else:
        fixed = []
        for i in indices:
            if i < 0:
                i = n + i
            if i < 0 or i >= n:
                raise IndexError(f"Index {i} out of range for path of length {n}")
            fixed.append(i)
        indices = fixed

    x_sub = [x_path[i] for i in indices]
    step_ids_sub = [step_ids[i] for i in indices] if step_ids is not None else None
    return x_sub, step_ids_sub
def denormalize_n(
    x: Union[torch.Tensor, np.ndarray],
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """
    Reverse standard ImageNet normalization for visualization.

    Accepts:
      - torch.Tensor or np.ndarray
      - shapes:
          [3, H, W]
          [H, W, 3]
          [1, 3, H, W]
          [1, H, W, 3]

    Returns:
      - np.ndarray of shape [H, W, 3] in [0, 1]
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    # Remove batch dimension if present
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 when passing 4D input, got shape {x.shape}")
        x = x[0]

    if x.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {x.shape}")

    # Convert CHW -> HWC if needed
    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    elif x.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, got shape {x.shape}")

    mean = np.asarray(mean, dtype=x.dtype)
    std = np.asarray(std, dtype=x.dtype)

    x = (x * std) + mean
    return np.clip(x, 0.0, 1.0)

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.gridspec import GridSpec


def _to_display_image(x, denormalize_fn=None):
    if denormalize_fn is not None:
        return denormalize_fn(x)

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    if x.ndim == 4:
        x = x[0]

    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    return x


def _select_evenly_spaced_indices(n_total, n_show):
    if n_show >= n_total:
        return list(range(n_total))
    idx = np.linspace(0, n_total - 1, n_show)
    return np.unique(np.round(idx).astype(int)).tolist()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _safe_reduce_attr(attr):
    """Safely reduces a (B, C, H, W) or (C, H, W) tensor to a 2D (H, W) heatmap."""
    if hasattr(attr, "cpu"):
        attr = attr.detach().cpu().numpy()
    else:
        attr = np.asarray(attr)

    if attr.ndim == 4 and attr.shape[0] == 1:
        attr = attr[0]

    if attr.ndim == 3 and attr.shape[0] in [1, 3]:
        attr = np.sum(attr, axis=0)
    elif attr.ndim == 3 and attr.shape[-1] in [1, 3]:
        attr = np.sum(attr, axis=-1)

    return attr


def plot_entropy_with_snapshot_arrows_and_delta_attrs(
        stats,
        denormalize_fn=None,
        spacing_power=1.0,  # Keeping linear distribution for equal time gaps
        n_show_images=6,
        image_indices=None,
        num_classes=1000,
        save_path=None,
        title=None,
):
    """
    Static figure plotting entropy, selected snapshots, and Cumulative DELTA attributions.
    Requires `stats["delta_attr_path"]` to be properly populated by the generator.
    """
    if not all(k in stats for k in ("entropy", "x_path", "x_path_step_ids", "delta_attr_path")):
        raise ValueError("stats must contain 'entropy', 'x_path', 'x_path_step_ids', and 'delta_attr_path'")

    entropy = np.asarray(stats["entropy"], dtype=float)
    if entropy.ndim == 1:
        entropy = entropy[:, None]
    entropy_mean = np.nanmean(entropy, axis=1)

    max_ent = np.log(num_classes)
    if max_ent > 0:
        entropy_mean = entropy_mean / max_ent

    x_path = stats["x_path"]
    delta_attrs = stats["delta_attr_path"]
    step_ids = np.asarray(stats["x_path_step_ids"], dtype=int)
    n_total_imgs = len(x_path)

    # 1. Selection Logic
    if image_indices is None:
        if n_show_images >= n_total_imgs:
            show_idx = list(range(n_total_imgs))
        else:
            t = np.linspace(0, 1, n_show_images) ** spacing_power
            raw_idx = np.round(t * (n_total_imgs - 1)).astype(int)

            show_idx = []
            for idx in raw_idx:
                while idx in show_idx and idx < n_total_imgs - 1:
                    idx += 1
                if idx not in show_idx:
                    show_idx.append(idx)
                else:
                    idx = n_total_imgs - 1
                    while idx in show_idx and idx > 0:
                        idx -= 1
                    show_idx.append(idx)

            show_idx = sorted(show_idx)
    else:
        show_idx = [n_total_imgs + idx if idx < 0 else idx for idx in image_indices]

    shown_images = [x_path[i] for i in show_idx]
    shown_step_ids = step_ids[show_idx]

    telemetry_steps = np.clip(shown_step_ids, 0, len(entropy_mean) - 1)
    telemetry_vals = entropy_mean[telemetry_steps]

    n_panels = len(shown_images)
    fig = plt.figure(figsize=(3.2 * n_panels, 9))
    gs = GridSpec(3, n_panels, height_ratios=[0.8, 1.25, 1.4], figure=fig)

    # --- Top row: Entropy curve ---
    ax_entropy = fig.add_subplot(gs[0, :])
    ax_entropy.plot(np.arange(len(entropy_mean)), entropy_mean, linewidth=2)
    ax_entropy.set_title("Predictive Entropy")
    ax_entropy.set_xlabel("Step")
    ax_entropy.set_ylabel("Normalized Entropy")
    ax_entropy.grid(True, alpha=0.3)

    for step in telemetry_steps:
        ax_entropy.axvline(step, linestyle=":", alpha=0.18)

    # --- Middle row: Selected images ---
    image_axes = []
    for j, (img, sid) in enumerate(zip(shown_images, shown_step_ids)):
        ax = fig.add_subplot(gs[1, j])
        disp = _to_display_image(img, denormalize_fn=denormalize_fn)

        if disp.ndim == 2:
            ax.imshow(disp, cmap="gray")
        else:
            ax.imshow(np.clip(disp, 0, 1) if np.issubdtype(disp.dtype, np.floating) else disp)

        ax.set_title(f"step {sid}", fontweight="bold")
        ax.axis("off")
        image_axes.append(ax)

    # --- Bottom row: Delta Attributions ---

    # Panel 0: The Anchor (Final Total Attribution)
    ax0 = fig.add_subplot(gs[2, 0])
    final_attr = stats.get("attr", stats.get("attribution", None))

    if final_attr is not None:
        final_attr = _safe_reduce_attr(final_attr)
        attr_disp = _prepare_attr_for_display(final_attr)
        final_vmax = max(np.percentile(np.abs(attr_disp), 99), 1e-5)

        im0 = ax0.imshow(attr_disp, cmap="seismic", vmin=-final_vmax, vmax=final_vmax)
        ax0.set_title("Final Total Attribution", fontsize=10)
        ax0.axis("off")

        divider = make_axes_locatable(ax0)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax)
    else:
        ax0.text(0.5, 0.5, "no final attr provided", ha="center", va="center")
        ax0.axis("off")

    # Pre-computation: Accumulate gaps and calculate Absolute Volumes
    plotted_deltas = []
    chunk_magnitudes = []

    for j in range(1, n_panels):
        start_idx = show_idx[j - 1]
        end_idx = show_idx[j]

        if end_idx <= len(delta_attrs):
            chunks_to_sum = delta_attrs[start_idx:end_idx]

            d_attr = chunks_to_sum[0].clone() if hasattr(chunks_to_sum[0], 'clone') else chunks_to_sum[0].copy()
            for chunk in chunks_to_sum[1:]:
                d_attr = d_attr + chunk

            d_attr = _safe_reduce_attr(d_attr)
            chunk_volume = np.sum(np.abs(d_attr))
            chunk_magnitudes.append(chunk_volume)

            d_attr_disp = _prepare_attr_for_display(d_attr)
            plotted_deltas.append(d_attr_disp)
        else:
            plotted_deltas.append(None)
            chunk_magnitudes.append(0.0)

    # ---> THE FIX: Calculate the Total Path Volume for the Progress Bar <---
    total_path_volume = sum(chunk_magnitudes) if chunk_magnitudes else 1e-9
    running_volume = 0.0

    valid_deltas = [d for d in plotted_deltas if d is not None]
    if valid_deltas:
        delta_vmax = max(np.percentile(np.abs(valid_deltas[0]), 99), 1e-5)
    else:
        delta_vmax = 1.0

    # Plotting the Delta Panels
    for j in range(1, n_panels):
        ax = fig.add_subplot(gs[2, j])
        d_attr_disp = plotted_deltas[j - 1]
        chunk_vol = chunk_magnitudes[j - 1]

        if d_attr_disp is not None:
            im = ax.imshow(d_attr_disp, cmap="seismic", vmin=-final_vmax, vmax=final_vmax)

            # ---> THE FIX: Update the Running Tally and calculate Progress <---
            running_volume += chunk_vol
            progress_pct = (running_volume / total_path_volume) * 100

            prev_sid = shown_step_ids[j - 1]
            curr_sid = shown_step_ids[j]
            ax.set_title(f"Cumulative Δ Attr\n(steps {prev_sid} → {curr_sid})",
                         fontsize=10)
            ax.axis("off")

            # if j == n_panels - 1:
            #     divider = make_axes_locatable(ax)
                #cax = divider.append_axes("right", size="5%", pad=0.05)
                #plt.colorbar(im, cax=cax)
        else:
            ax.text(0.5, 0.5, "no delta data", ha="center", va="center")
            ax.axis("off")

    # --- Arrows and Scatter Points ---
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_panels, 1)))
    for step, yval, target_ax, sid, color in zip(telemetry_steps, telemetry_vals, image_axes, shown_step_ids, colors):
        ax_entropy.scatter([step], [yval], s=70, color=color, zorder=6)

        con = ConnectionPatch(
            xyA=(step, yval), coordsA=ax_entropy.transData,
            xyB=(0.5, 1.18), coordsB=target_ax.transAxes,
            arrowstyle="->", mutation_scale=16, linewidth=2, color=color, alpha=0.9
        )
        fig.add_artist(con)

        for spine in target_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_edgecolor(color)

    fig.suptitle(title or "Entropy trajectory, snapshots, and step-wise Delta Attributions", y=0.985, fontsize=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
#
# def plot_entropy_with_snapshot_arrows(
#     stats,
#     denormalize_fn=None,
#     n_show_images=6,
#     image_indices=None,
#     num_classes=1000,
#     save_path=None,
# ):
#     """
#     Static figure:
#       - top row: full entropy curve
#       - bottom row: selected intermediate images
#       - one arrow from each selected entropy point to its corresponding snapshot
#
#     Parameters
#     ----------
#     stats : dict
#         Must contain:
#           - "entropy"
#           - "x_path"
#           - "x_path_step_ids"
#     denormalize_fn : callable, optional
#         Function used to convert normalized tensors to displayable RGB images.
#     n_show_images : int
#         Number of evenly spaced images to show if image_indices is None.
#     image_indices : list[int], optional
#         Explicit indices into x_path to display.
#         Example: [0, 3, 7, 12, 20, -1]
#     num_classes : int
#         Used to normalize entropy by log(num_classes).
#     save_path : str, optional
#         Path to save the figure.
#     """
#
#     if "entropy" not in stats or "x_path" not in stats or "x_path_step_ids" not in stats:
#         raise ValueError("stats must contain 'entropy', 'x_path', and 'x_path_step_ids'")
#
#     entropy = np.asarray(stats["entropy"], dtype=float)
#     if entropy.ndim == 1:
#         entropy = entropy[:, None]
#
#     entropy_mean = np.nanmean(entropy, axis=1)
#
#     max_ent = np.log(num_classes)
#     if max_ent > 0:
#         entropy_mean = entropy_mean / max_ent
#
#     x_path = stats["x_path"]
#     step_ids = np.asarray(stats["x_path_step_ids"], dtype=int)
#
#     n_total_imgs = len(x_path)
#     if n_total_imgs == 0:
#         raise ValueError("x_path is empty.")
#
#     # Select which saved snapshots to show
#     if image_indices is None:
#         if n_show_images >= n_total_imgs:
#             show_idx = list(range(n_total_imgs))
#         else:
#             show_idx = np.linspace(0, n_total_imgs - 1, n_show_images)
#             show_idx = np.unique(np.round(show_idx).astype(int)).tolist()
#     else:
#         show_idx = []
#         for idx in image_indices:
#             if idx < 0:
#                 idx = n_total_imgs + idx
#             if idx < 0 or idx >= n_total_imgs:
#                 raise IndexError(f"image index {idx} out of range for x_path of length {n_total_imgs}")
#             show_idx.append(idx)
#
#     shown_images = [x_path[i] for i in show_idx]
#     shown_step_ids = step_ids[show_idx]
#
#     # Map saved step ids to valid telemetry indices
#     telemetry_steps = np.clip(shown_step_ids, 0, len(entropy_mean) - 1)
#     telemetry_vals = entropy_mean[telemetry_steps]
#
#     n_panels = len(shown_images)
#
#     fig = plt.figure(figsize=(2.8 * n_panels, 6.5))
#     gs = GridSpec(2, n_panels, height_ratios=[2.3, 1.25], figure=fig)
#
#     # --- Top row: entropy curve ---
#     ax_entropy = fig.add_subplot(gs[0, :])
#     xs = np.arange(len(entropy_mean))
#
#     ax_entropy.plot(xs, entropy_mean, linewidth=2)
#     ax_entropy.scatter(telemetry_steps, telemetry_vals, s=55, zorder=5)
#
#     ax_entropy.set_title("Predictive Entropy")
#     ax_entropy.set_xlabel("Step")
#     ax_entropy.set_ylabel("Normalized Entropy")
#     ax_entropy.grid(True, alpha=0.3)
#
#     # Optional vertical guides
#     for step in telemetry_steps:
#         ax_entropy.axvline(step, linestyle=":", alpha=0.18)
#
#     # --- Bottom row: selected images ---
#     image_axes = []
#     for j, (img, sid) in enumerate(zip(shown_images, shown_step_ids)):
#         ax = fig.add_subplot(gs[1, j])
#         disp = _to_display_image(img, denormalize_fn=denormalize_fn)
#
#         if disp.ndim == 2:
#             ax.imshow(disp, cmap="gray")
#         else:
#             if np.issubdtype(disp.dtype, np.floating):
#                 disp = np.clip(disp, 0, 1)
#             ax.imshow(disp)
#
#         ax.set_title(f"step {sid}", fontweight="bold")
#         ax.axis("off")
#         image_axes.append(ax)
#
#     # --- One arrow per selected snapshot ---
#     colors = plt.cm.tab10(np.linspace(0, 1, max(n_panels, 1)))
#
#     for j, (step, yval, target_ax, sid, color) in enumerate(
#         zip(telemetry_steps, telemetry_vals, image_axes, shown_step_ids, colors)
#     ):
#         # color the corresponding point
#         ax_entropy.scatter([step], [yval], s=70, color=color, zorder=6)
#
#         # small label near the entropy point
#         ax_entropy.annotate(
#             f"{sid}",
#             xy=(step, yval),
#             xytext=(0, 8),
#             textcoords="offset points",
#             ha="center",
#             fontsize=9,
#             color=color,
#         )
#
#         con = ConnectionPatch(
#             xyA=(step, yval),
#             coordsA=ax_entropy.transData,
#             xyB=(0.5, 1.02),
#             coordsB=target_ax.transAxes,
#             arrowstyle="->",
#             mutation_scale=16,
#             linewidth=2,
#             color=color,
#             alpha=0.9,
#         )
#         fig.add_artist(con)
#
#         # match image border color to arrow color
#         for spine in target_ax.spines.values():
#             spine.set_visible(True)
#             spine.set_linewidth(2.0)
#             spine.set_edgecolor(color)
#
#     fig.suptitle("Entropy trajectory linked to selected intermediate snapshots", y=0.98)
#     plt.tight_layout()
#
#     if save_path is not None:
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#
#     plt.show()
#
#
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from matplotlib.patches import ConnectionPatch
#
#
# def _local_sharpness_map(img_rgb: np.ndarray, blur_sigma: float = 5.0) -> np.ndarray:
#     """
#     Local sharpness estimate from smoothed Laplacian energy.
#     Higher = sharper / more high-frequency structure.
#     """
#     if img_rgb.ndim == 2:
#         gray = img_rgb.astype(np.float32)
#     else:
#         if np.issubdtype(img_rgb.dtype, np.floating):
#             img_u8 = np.clip(img_rgb, 0, 1)
#             img_u8 = (img_u8 * 255).astype(np.uint8)
#         else:
#             img_u8 = img_rgb.astype(np.uint8)
#         gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
#
#     lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
#     energy = lap ** 2
#
#     k = int(max(3, 2 * round(3 * blur_sigma) + 1))
#     sharp = cv2.GaussianBlur(energy, (k, k), blur_sigma)
#     return sharp
#
#
# def _normalize_map(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
#     x = x.astype(np.float32)
#     x = x - np.min(x)
#     return x / (np.max(x) + eps)
#
#
# def _blur_increase_map(prev_img: np.ndarray, curr_img: np.ndarray, blur_sigma: float = 5.0) -> np.ndarray:
#     """
#     Positive where curr_img is blurrier than prev_img.
#     """
#     sharp_prev = _local_sharpness_map(prev_img, blur_sigma=blur_sigma)
#     sharp_curr = _local_sharpness_map(curr_img, blur_sigma=blur_sigma)
#     blur_inc = np.maximum(sharp_prev - sharp_curr, 0.0)
#     return blur_inc
#
#
#
def _prepare_attr_for_display(attr):
    """
    Convert an attribution tensor/array into a 2D signed map for display.

    Accepted shapes:
      - [H, W]
      - [C, H, W]
      - [H, W, C]

    Returns
    -------
    arr : np.ndarray, shape [H, W]
        Signed 2D attribution map.
    """
    arr = np.asarray(attr)

    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim == 3:
        # CHW
        if arr.shape[0] in (1, 3):
            return arr.mean(axis=0).astype(np.float32)
        # HWC
        if arr.shape[-1] in (1, 3):
            return arr.mean(axis=-1).astype(np.float32)

    raise ValueError(f"Unsupported attribution shape: {arr.shape}")
#
#
# def plot_entropy_with_snapshot_arrows_and_diffs(
#     stats,
#     denormalize_fn=None,
#     n_show_images=6,
#     image_indices=None,
#     num_classes=1000,
#     save_path=None,
#     diff_mode="blur",   # "blur" or "rgb"
#     blur_sigma=5.0,
#     attr_map=None,      # NEW: optional final attribution map
#     title = None,
# ):
#     """
#     Static figure:
#       - top row: full entropy curve
#       - middle row: selected intermediate images
#       - bottom row:
#           * first panel: final attribution map
#           * remaining panels: difference map wrt the first shown image
#       - one arrow from each selected entropy point to its corresponding snapshot
#
#     diff_mode:
#       - "blur": local blur-increase map wrt the first shown snapshot
#       - "rgb": mean absolute RGB difference wrt the first shown snapshot
#
#     IMPORTANT:
#       All bottom-row diff maps use a *global* normalization across shown panels,
#       so intensities are directly comparable across steps.
#     """
#
#     if "entropy" not in stats or "x_path" not in stats or "x_path_step_ids" not in stats:
#         raise ValueError("stats must contain 'entropy', 'x_path', and 'x_path_step_ids'")
#
#     entropy = np.asarray(stats["entropy"], dtype=float)
#     if entropy.ndim == 1:
#         entropy = entropy[:, None]
#
#     entropy_mean = np.nanmean(entropy, axis=1)
#
#     max_ent = np.log(num_classes)
#     if max_ent > 0:
#         entropy_mean = entropy_mean / max_ent
#
#     x_path = stats["x_path"]
#     step_ids = np.asarray(stats["x_path_step_ids"], dtype=int)
#
#     n_total_imgs = len(x_path)
#     if n_total_imgs == 0:
#         raise ValueError("x_path is empty.")
#
#     # Select which saved snapshots to show
#     if image_indices is None:
#         if n_show_images >= n_total_imgs:
#             show_idx = list(range(n_total_imgs))
#         else:
#             show_idx = np.linspace(0, n_total_imgs - 1, n_show_images)
#             show_idx = np.unique(np.round(show_idx).astype(int)).tolist()
#     else:
#         show_idx = []
#         for idx in image_indices:
#             if idx < 0:
#                 idx = n_total_imgs + idx
#             if idx < 0 or idx >= n_total_imgs:
#                 raise IndexError(f"image index {idx} out of range for x_path of length {n_total_imgs}")
#             show_idx.append(idx)
#
#     shown_images = [x_path[i] for i in show_idx]
#     shown_step_ids = step_ids[show_idx]
#
#     # Map saved step ids to valid telemetry indices
#     telemetry_steps = np.clip(shown_step_ids, 0, len(entropy_mean) - 1)
#     telemetry_vals = entropy_mean[telemetry_steps]
#
#     n_panels = len(shown_images)
#
#     fig = plt.figure(figsize=(2.8 * n_panels, 8.8))
#     gs = GridSpec(3, n_panels, height_ratios=[2.3, 1.25, 1.15], figure=fig)
#
#     # --- Top row: entropy curve ---
#     ax_entropy = fig.add_subplot(gs[0, :])
#     xs = np.arange(len(entropy_mean))
#
#     ax_entropy.plot(xs, entropy_mean, linewidth=2)
#     ax_entropy.scatter(telemetry_steps, telemetry_vals, s=55, zorder=5)
#
#     ax_entropy.set_title("Predictive Entropy")
#     ax_entropy.set_xlabel("Step")
#     ax_entropy.set_ylabel("Normalized Entropy")
#     ax_entropy.grid(True, alpha=0.3)
#
#     for step in telemetry_steps:
#         ax_entropy.axvline(step, linestyle=":", alpha=0.18)
#
#     # --- Middle row: selected images ---
#     image_axes = []
#     shown_disp = []
#
#     for j, (img, sid) in enumerate(zip(shown_images, shown_step_ids)):
#         ax = fig.add_subplot(gs[1, j])
#         disp = _to_display_image(img, denormalize_fn=denormalize_fn)
#
#         if disp.ndim == 2:
#             ax.imshow(disp, cmap="gray")
#         else:
#             if np.issubdtype(disp.dtype, np.floating):
#                 disp = np.clip(disp, 0, 1)
#             ax.imshow(disp)
#
#         shown_disp.append(disp)
#         ax.set_title(f"step {sid}", fontweight="bold")
#         ax.axis("off")
#         image_axes.append(ax)
#
#     # Use the first shown image as the fixed reference
#     ref_disp = shown_disp[0]
#     ref_sid = shown_step_ids[0]
#
#     # ---------------------------------------------------------
#     # PRECOMPUTE ALL DIFF MAPS FIRST -> GLOBAL NORMALIZATION
#     # ---------------------------------------------------------
#     diff_maps = [None]  # panel 0 reserved for attribution map
#     diff_strengths = [None]
#
#     for j in range(1, n_panels):
#         curr_disp = shown_disp[j]
#
#         if diff_mode == "blur":
#             # IMPORTANT: _blur_increase_map should return a RAW map, not normalized.
#             diff_map = _blur_increase_map(ref_disp, curr_disp, blur_sigma=blur_sigma)
#
#         elif diff_mode == "rgb":
#             ref_f = ref_disp.astype(np.float32)
#             curr_f = curr_disp.astype(np.float32)
#
#             if ref_f.ndim == 3:
#                 diff_map = np.mean(np.abs(curr_f - ref_f), axis=2)
#             else:
#                 diff_map = np.abs(curr_f - ref_f)
#
#         else:
#             raise ValueError("diff_mode must be 'blur' or 'rgb'")
#
#         diff_maps.append(diff_map)
#         diff_strengths.append(float(np.mean(diff_map)))
#
#     valid_maps = [m for m in diff_maps[1:] if m is not None]
#     global_vmax = max(float(np.max(m)) for m in valid_maps) if valid_maps else 1.0
#     if global_vmax <= 0:
#         global_vmax = 1.0
#
#     # --- Bottom row ---
#     diff_axes = []
#
#     # Panel 0: attribution map instead of empty reference
#     ax0 = fig.add_subplot(gs[2, 0])
#
#     if attr_map is None:
#         attr_map = stats.get("attr", stats.get("attribution", None))
#
#     if attr_map is not None:
#         attr_disp = _prepare_attr_for_display(attr_map)
#         vmax_attr = np.max(np.abs(attr_disp))
#         if vmax_attr <= 0:
#             vmax_attr = 1.0
#         im0 = ax0.imshow(attr_disp, cmap="seismic", vmin=-vmax_attr, vmax=vmax_attr)
#         ax0.set_title("Final Attribution", fontsize=10)
#         ax0.axis("off")
#         plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02 , orientation="horizontal")
#     else:
#         ax0.text(
#             0.5, 0.5, "no attribution\nprovided",
#             ha="center", va="center", fontsize=10
#         )
#         ax0.axis("off")
#
#     diff_axes.append(ax0)
#
#     # Remaining panels: cumulative difference wrt first shown image
#     for j in range(1, n_panels):
#         ax = fig.add_subplot(gs[2, j])
#         diff_map = diff_maps[j]
#
#         if diff_mode == "blur":
#             im = ax.imshow(diff_map, cmap="jet", vmin=0, vmax=global_vmax)
#             ax.set_title(
#                 f"Blur Increase\nvs step {ref_sid}",
#                 fontsize=10
#             )
#         else:  # rgb
#             im = ax.imshow(diff_map, cmap="magma", vmin=0, vmax=global_vmax)
#             ax.set_title(
#                 f"|RGB diff|\nvs step {ref_sid}\nmean={diff_strengths[j]:.4f}",
#                 fontsize=9
#             )
#
#         ax.axis("off")
#         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, orientation="horizontal")
#         diff_axes.append(ax)
#
#     # --- One arrow per selected snapshot ---
#     colors = plt.cm.tab10(np.linspace(0, 1, max(n_panels, 1)))
#
#     for j, (step, yval, target_ax, sid, color) in enumerate(
#         zip(telemetry_steps, telemetry_vals, image_axes, shown_step_ids, colors)
#     ):
#         ax_entropy.scatter([step], [yval], s=70, color=color, zorder=6)
#
#         ax_entropy.annotate(
#             f"{sid}",
#             xy=(step, yval),
#             xytext=(0, 8),
#             textcoords="offset points",
#             ha="center",
#             fontsize=9,
#             color=color,
#         )
#
#         con = ConnectionPatch(
#             xyA=(step, yval),
#             coordsA=ax_entropy.transData,
#             xyB=(0.5, 1.18),
#             coordsB=target_ax.transAxes,
#             arrowstyle="->",
#             mutation_scale=16,
#             linewidth=2,
#             color=color,
#             alpha=0.9,
#         )
#         fig.add_artist(con)
#
#         for spine in target_ax.spines.values():
#             spine.set_visible(True)
#             spine.set_linewidth(2.0)
#             spine.set_edgecolor(color)
#
#     fig.suptitle(
#         f"Entropy trajectory, selected snapshots, cumulative difference maps, and final attribution. {title}",
#         y=0.985,
#         fontsize=15,
#     )
#     plt.tight_layout()
#
#     if save_path is not None:
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#
#     plt.show()
#
#
# def plot_integration_dynamics(stats, batch_idx=0):
#     """
#     Plots the step-by-step dynamics of the path integration to diagnose
#     coherence, gradient accumulation, and model confidence.
#
#     Args:
#         stats (dict): The stats dictionary returned by the attribute method.
#         batch_idx (int): The index of the image in the batch to plot.
#     """
#     # Number of steps actually taken
#     num_steps = stats["total_steps"]
#     steps = np.arange(num_steps)
#
#     # Extract the metrics for the specific image across all steps
#     # stats[metric] is a list of length 'steps', where each element is a list of length 'batch_size'
#     cos_sim = [step_vals[batch_idx] for step_vals in stats.get("cosine_similarity", [])]
#     dir_deriv = [step_vals[batch_idx] for step_vals in stats.get("directional_derivative", [])]
#     target_logit = [step_vals[batch_idx] for step_vals in stats.get("target_logit", [])]
#     step_size = [step_vals[batch_idx] for step_vals in stats.get("step_size_euclidean", [])]
#
#     # Create a 2x2 grid of plots
#     fig, axs = plt.subplots(2, 2, figsize=(14, 10))
#     fig.suptitle(f"Path Integration Dynamics (Batch Index: {batch_idx})", fontsize=16, fontweight='bold')
#
#     # 1. Cosine Similarity (The "Coherence" check)
#     axs[0, 0].plot(steps, cos_sim, color='purple', linewidth=2)
#     axs[0, 0].set_title("Cosine Similarity: ∇F(x) vs Path Velocity")
#     axs[0, 0].set_xlabel("Integration Step")
#     axs[0, 0].set_ylabel("Cosine Similarity")
#     axs[0, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
#     axs[0, 0].grid(True, alpha=0.3)
#
#     # 2. Directional Derivative (The "Attribution" check)
#     axs[0, 1].plot(steps, dir_deriv, color='teal', linewidth=2)
#     axs[0, 1].set_title("Directional Derivative (Credit Accumulated)")
#     axs[0, 1].set_xlabel("Integration Step")
#     axs[0, 1].set_ylabel("∇F(x) · dx")
#     axs[0, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
#     axs[0, 1].grid(True, alpha=0.3)
#
#     # 3. Target Logit (The "Confidence" check)
#     axs[1, 0].plot(steps, target_logit, color='darkorange', linewidth=2)
#     axs[1, 0].set_title("Target Class Logit F_t(x)")
#     axs[1, 0].set_xlabel("Integration Step")
#     axs[1, 0].set_ylabel("Logit Value")
#     axs[1, 0].grid(True, alpha=0.3)
#
#     # 4. Euclidean Step Size (The "Pacing" check)
#     axs[1, 1].plot(steps, step_size, color='crimson', linewidth=2)
#     axs[1, 1].set_title("Euclidean Step Size ||dx||")
#     axs[1, 1].set_xlabel("Integration Step")
#     axs[1, 1].set_ylabel("Step Magnitude")
#     axs[1, 1].grid(True, alpha=0.3)
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()