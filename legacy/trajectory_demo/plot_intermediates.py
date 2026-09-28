import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def _pick_equally_spaced_indices(lo: int, hi_excl: int, k: int) -> list[int]:
    """
    Pick k indices in [lo, hi_excl-1], always including endpoints (lo and hi_excl-1).
    Returns a sorted list of unique ints. If the range is too short, returns all indices.
    """
    if hi_excl <= lo:
        return []
    n = hi_excl - lo
    if n <= k:
        return list(range(lo, hi_excl))

    raw = np.linspace(lo, hi_excl - 1, k)
    idx = np.rint(raw).astype(int)
    idx[0] = lo
    idx[-1] = hi_excl - 1

    # unique while preserving order
    idx = list(dict.fromkeys(idx.tolist()))

    # if rounding collapsed duplicates, fill with missing indices to reach k
    if len(idx) < k:
        missing = [i for i in range(lo, hi_excl) if i not in idx]
        # add closest missing indices (in order) until we hit k
        for i in missing:
            idx.append(i)
            if len(idx) == k:
                break
        idx = sorted(idx)

    return idx


def visualize_steps_prev_delta(
        info,
        steps=None,  # explicit list of indices
        start=0,
        stop=None,
        stride=1,
        k_samples=5,
        cols=10,
        C=1000,
        tick_decimals=2,  # Increased decimals to see small progress diffs
        cmap_delta="coolwarm",
        delta_vmin=-1,
        delta_vmax=1,
        dpi=200,
        left_label_pad=0.02,
        savepath=None,
):
    icml_rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.25,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with mpl.rc_context(icml_rc):
        entropy = np.asarray(info["intermediate_entropy"]).squeeze()
        xs = info["intermediate_x"]
        deltas = info["intermediate_attr"]
        logit = np.asarray(info["current_logit"]).squeeze()

        T = min(len(entropy), len(xs), len(deltas), len(logit))
        if T <= 0: return

        # Normalize metrics
        entropy = entropy[:T] / (np.log(C) + 1e-12)
        logit = logit[:T]
        logit = logit / (np.max(np.abs(logit)) + 1e-12)

        # ----- 1. Select Indices -----
        if stop is None: stop = T
        lo = max(0, int(start))
        hi = min(int(stop), T)

        if steps is None:
            # Auto-sample if no steps provided
            idx_all = _pick_equally_spaced_indices(lo, hi, k_samples)
        else:
            # Use provided steps, filtered by bounds
            idx_all = [int(i) for i in steps if lo <= int(i) < hi]
            if stride and stride > 1:
                idx_all = idx_all[::int(stride)]
            print(idx_all)
        if not idx_all: return

        # Paginate if too many columns
        cols = min(cols, len(idx_all))
        print("cols", cols)
        cols = 7
        page_id = 0
        for page_start in range(0, len(idx_all), cols):
            idx_page = idx_all[page_start:page_start + cols]
            ncols = len(idx_page)
            print("NUM COLS", ncols)

            fig = plt.figure(figsize=(3.0 * ncols, 6.9), constrained_layout=True)
            gs = fig.add_gridspec(nrows=3, ncols=ncols, height_ratios=[1.15, 2.6, 2.6])

            # ---------- Row 1: Metrics (The Fix) ----------
            # 1. We plot on integers (0, 1, 2...) so points align with image columns
            x_plot_coords = np.arange(ncols)

            # 2. Extract values for these specific steps
            y_ent = np.array([entropy[i] for i in idx_page], dtype=float)
            y_log = np.array([logit[i] for i in idx_page], dtype=float)

            ax_ent = fig.add_subplot(gs[0, :])
            ax_ent.plot(x_plot_coords, y_ent, marker="o", label="Entropy (norm.)")
            ax_ent.plot(x_plot_coords, y_log, marker="o", label="Logit (norm.)")

            # 3. Create Labels: Calculate fraction based on total steps T
            # If explicit steps were [0, 2, 8] and T=100, labels become 0.00, 0.02, 0.08
            progress_labels = [f"{i / (T - 1):.{tick_decimals}f}" for i in idx_page]

            ax_ent.set_xlim(-0.5, ncols - 0.5)
            ax_ent.set_xticks(x_plot_coords)
            ax_ent.set_xticklabels(progress_labels)

            ax_ent.set_xlabel("Progress (Step / Total)", fontsize=16)
            ax_ent.set_ylabel("Normalized value", fontsize=16)
            ax_ent.set_title("Intermediate steps along integration path", pad=6, fontsize=18)
            ax_ent.grid(True, axis="y")
            ax_ent.grid(True, axis="x", alpha=0.12)
            ax_ent.spines["top"].set_visible(False)
            ax_ent.spines["right"].set_visible(False)
            ax_ent.legend(frameon=False, loc="upper right", fontsize=14)

            # ---------- Row 2: Images ----------
            for col, i in enumerate(idx_page):
                ax = fig.add_subplot(gs[1, col])
                ax.imshow(xs[i])
                ax.set_axis_off()

            # ---------- Row 3: Deltas ----------
            first_in_seq = idx_all[0]  # The very first step of the whole sequence

            for col, i in enumerate(idx_page):
                ax = fig.add_subplot(gs[2, col])
                delta = np.asarray(deltas[i])

                # Logic: If it's the absolute start, show raw attribution or zeros?
                # Usually zeros for delta, or raw for "first view".
                # Here we ensure 0-th step is zeroed delta.
                if i == 0:
                    delta = np.zeros_like(delta)

                # Plot (always plot, just zeroed if needed)
                ax.imshow(delta, cmap=cmap_delta, vmin=delta_vmin, vmax=delta_vmax) if i > 0 else None
                ax.set_axis_off()

            # ---------- Labels ----------
            fig.text(left_label_pad, 0.55, "Intermediate images", rotation=90, ha="center", va="center", fontsize=18)
            fig.text(left_label_pad, 0.20, r"$\Delta$ attribution", rotation=90, ha="center", va="center", fontsize=18)

            if savepath is not None:
                out = savepath.format(page=page_id)
                fig.savefig(out, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()

            page_id += 1
