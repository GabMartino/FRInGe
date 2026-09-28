import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T

# =============================================================================
# 1. Plotting Configuration
# =============================================================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"""
        \usepackage{newtxtext}
        \usepackage{newtxmath}
    """,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})


def tex_escape(s: str) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_"}
    return "".join(replacements.get(ch, ch) for ch in str(s))


# =============================================================================
# 2. Paths & Micro-Grid Targets
# =============================================================================
RESULTS_ROOT = "./results/methods"
IMAGE_ROOT = "./data/ImageNet"
OUTPUT_DIR = "./analyses/outputs"

OURS_METHOD = "FisherRaoIG/3_fisher_smooth"
BASELINE_METHODS = ["IG", "IG2", "GuidedIG", "SmoothGrad", "AdversarialIG", "GGIG", "GIG"]

# The exact columns to display in the paper
MAIN_TEXT_METHODS = [OURS_METHOD, "IG", "SmoothGrad", "GGIG", "GIG"]

# Pin the exact paper examples. Recomputing the rank dynamically is useful for
# exploration, but it is not a stable specification of a publication figure:
# adding or replacing result files can change the median example while leaving
# the requested stratum unchanged.
# Format: (Model_ID, Display_Name, Stratum, Index_in_stratum, Join_Key)
MAIN_TEXT_SELECTION = [
    ("inception_v3", "Inception-V3", "Top (Win)", 0, "n02814860_beacon.JPEG"),
    ("resnet101", "ResNet-101", "Median (Typical)", 0, "n07871810_meat_loaf.JPEG"),
    ("resnet18", "ResNet-18", "Bottom (Failure)", 0, "n03937543_pill_bottle.JPEG"),
]

ATTR_CLIP_Q = 0.99
JOIN_MODE = "basename"

METRIC_SPECS = {
    "mas_insertion_auc_mean": {"higher_is_better": True},
    "mas_deletion_auc_mean": {"higher_is_better": False},
}


# =============================================================================
# 3. Core Utilities
# =============================================================================
def get_geometry_preprocess_no_norm(model_name: str) -> T.Compose:
    if "inception" in model_name.lower():
        return T.Compose([T.Resize(342, antialias=True), T.CenterCrop(299)])
    return T.Compose([T.Resize(256, antialias=True), T.CenterCrop(224)])


def preprocess_attributions(attr, clip_q: float = 0.99) -> np.ndarray:
    if isinstance(attr, torch.Tensor):
        attr = attr.detach().cpu().numpy()
    else:
        attr = np.asarray(attr)

    if attr.ndim == 4 and attr.shape[0] == 1: attr = attr[0]
    if attr.ndim == 3:
        heat = np.abs(attr).sum(axis=0 if attr.shape[0] in (1, 3) else -1)
    elif attr.ndim == 2:
        heat = np.abs(attr)
    else:
        return np.zeros((1, 1), dtype=np.float32)

    if heat.size == 0: return np.zeros((1, 1), dtype=np.float32)

    heat = np.clip(heat, 0.0, np.nanquantile(heat.reshape(-1), clip_q))
    hmin, hmax = float(np.nanmin(heat)), float(np.nanmax(heat))
    denom = hmax - hmin
    if not np.isfinite(denom) or denom < 1e-12: return np.zeros_like(heat, dtype=np.float32)
    return np.clip((heat - hmin) / denom, 0.0, 1.0).astype(np.float32)


def load_attributions_robust(attrib_dir: str, target_keys: set) -> Dict[str, np.ndarray]:
    subdir = os.path.join(attrib_dir, "attributions")
    attrib_dir = subdir if os.path.isdir(subdir) else attrib_dir
    if not os.path.isdir(attrib_dir): return {}

    attr_map = {}
    for fname in os.listdir(attrib_dir):
        fpath = os.path.join(attrib_dir, fname)
        if fname.endswith(".pt"):
            try:
                data = torch.load(fpath, map_location="cpu", weights_only=True)
                filenames = data.get("filenames", data.get("image_paths", []))
                attrs = data.get("attr", data.get("attributions", []))
                for raw_name, attr in zip(filenames, attrs):
                    key = os.path.basename(str(raw_name))
                    if key in target_keys: attr_map[key] = attr
            except Exception:
                pass
    return attr_map


def get_stratified_image_for_model(
        model: str,
        stratum: str,
        idx: int,
        join_key: Optional[str] = None,
) -> Optional[pd.Series]:
    """Compute relative scores and return a pinned or stratified example."""
    # 1. Load Metrics
    store = {}
    for method in [OURS_METHOD] + BASELINE_METHODS:
        mdir = os.path.join(RESULTS_ROOT, method, model)
        if not os.path.isdir(mdir): continue
        metric_files = sorted(
            f for f in os.listdir(mdir)
            if f.endswith(".parquet") and "metrics" in f
        )
        # The corrected MAS-only evaluation is the source used by the paper's
        # telemetry when it exists. Concatenating it with the older aggregate
        # file makes duplicate rows and leaves ``groupby.first`` dependent on
        # filesystem enumeration order.
        preferred_name = (
            "mas_only_results_metrics.parquet"
            if method == OURS_METHOD
            and "mas_only_results_metrics.parquet" in metric_files
            else "results_metrics.parquet"
        )
        if preferred_name in metric_files:
            metric_files = [preferred_name]
        frames = [
            pd.read_parquet(os.path.join(mdir, filename))
            for filename in metric_files
        ]
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df["join_key"] = df["image_path"].apply(lambda p: os.path.basename(str(p)))
            store[method] = df.groupby("join_key", as_index=False).first()

    if OURS_METHOD not in store: return None
    merged = store[OURS_METHOD].rename(columns={m: f"{m}__ours" for m in METRIC_SPECS})

    # 2. Calculate Dominance
    dominance_sum = pd.Series(0.0, index=merged.index)
    valid_bases = 0
    for base in BASELINE_METHODS:
        if base not in store: continue
        valid_bases += 1
        base_df = store[base].rename(columns={m: f"{m}__{base}" for m in METRIC_SPECS})
        merged = pd.merge(merged, base_df[["join_key"] + [f"{m}__{base}" for m in METRIC_SPECS]], on="join_key",
                          how="inner")

        per_base_score = pd.Series(0.0, index=merged.index)
        for metric, spec in METRIC_SPECS.items():
            delta = merged[f"{metric}__ours"] - merged[f"{metric}__{base}"]
            per_base_score += delta if spec["higher_is_better"] else -delta
        dominance_sum += per_base_score

    if valid_bases == 0: return None
    merged["global_dominance_score"] = dominance_sum / valid_bases
    merged = merged.dropna(subset=["global_dominance_score"])
    merged = merged.sort_values("global_dominance_score", ascending=False).reset_index(drop=True)

    # Publication mode: fail explicitly if the retained paper example is no
    # longer present instead of silently substituting a different image.
    if join_key is not None:
        selected = merged.loc[merged["join_key"] == join_key]
        if selected.empty:
            return None
        row = selected.iloc[0].copy()
        row["stratum_label"] = stratum
        return row

    # 3. Stratify
    n = len(merged)
    if n < 6: return None
    stratum_map = {
        "Top (Win)": merged.iloc[0:2],
        "Median (Typical)": merged.iloc[(n // 2) - 1: (n // 2) + 1],
        "Bottom (Failure)": merged.iloc[n - 2: n]
    }

    target_df = stratum_map.get(stratum)
    if target_df is not None and len(target_df) > idx:
        row = target_df.iloc[idx].copy()
        row["stratum_label"] = stratum
        return row
    return None


# =============================================================================
# 4. Plotting the Micro-Grid
# =============================================================================
def plot_main_text_micro_grid(
        global_rows,
        attr_maps,
        output_dir: str = OUTPUT_DIR,
        output_format: str = "jpeg",
) -> str:
    n_rows, n_cols = len(global_rows), 1 + len(MAIN_TEXT_METHODS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.8 * n_rows), squeeze=False)

    axes[0, 0].set_title("Original", fontsize=16, fontweight="bold")
    for j, method in enumerate(MAIN_TEXT_METHODS, start=1):
        if method == "GIG":
            method = "GeoIG"
        axes[0, j].set_title("FRInGe (Ours)" if method == OURS_METHOD else tex_escape(method), fontsize=16,
                             fontweight="bold")

    resample_filter = getattr(Image, "Resampling", Image).BILINEAR

    for i, data in enumerate(global_rows):
        row, model_id, model_label = data["row"], data["model_id"], data["model_label"]
        join_key = str(row["join_key"])

        img_path = str(row["image_path"])
        if not os.path.isabs(img_path): img_path = os.path.join(IMAGE_ROOT, img_path)

        ax_img = axes[i, 0]
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        try:
            pil = get_geometry_preprocess_no_norm(model_id)(Image.open(img_path).convert("RGB"))
            img_array = np.asarray(pil).astype(np.float32) / 255.0
            target_h, target_w = img_array.shape[:2]
            ax_img.imshow(img_array)
        except Exception:
            ax_img.set_facecolor("black")
            target_h, target_w = 224, 224

        ax_img.set_ylabel(f"{model_label}\n{row.get('stratum_label', '')}", rotation=0, labelpad=50, va="center",
                          fontsize=12, fontweight="bold")

        for j, method in enumerate(MAIN_TEXT_METHODS, start=1):
            ax = axes[i, j]
            ax.set_xticks([]);
            ax.set_yticks([])
            attr = attr_maps.get(method, {}).get(model_id, {}).get(join_key)

            if attr is not None:
                try:
                    heat_pil = Image.fromarray((preprocess_attributions(attr) * 255).astype(np.uint8))
                    ax.imshow(np.asarray(heat_pil.resize((target_w, target_h), resample=resample_filter)) / 255.0,
                              cmap="inferno", vmin=0, vmax=1)
                except Exception:
                    ax.set_facecolor("black")
            else:
                ax.set_facecolor("black")

            # Add Mini-Telemetry Box
            suffix = "__ours" if method == OURS_METHOD else f"__{method}"
            text = f"MI {row.get(f'mas_insertion_auc_mean{suffix}', np.nan):.2f}\nMD {row.get(f'mas_deletion_auc_mean{suffix}', np.nan):.2f}"
            ax.text(0.03, 0.03, text, transform=ax.transAxes, ha="left", va="bottom", color="white", fontsize=10,
                    bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", boxstyle="round,pad=0.2"))

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    normalized_format = output_format.lower().lstrip(".")
    format_map = {"jpg": ("jpeg", "jpg"), "jpeg": ("jpeg", "jpg"),
                  "png": ("png", "png"), "pdf": ("pdf", "pdf")}
    if normalized_format not in format_map:
        raise ValueError("output_format must be one of: jpeg, jpg, png, pdf.")
    extension, matplotlib_format = format_map[normalized_format]

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"main_text_micro_grid.{extension}")
    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
        format=matplotlib_format,
    )
    print(f"[Saved] {save_path}")
    plt.close(fig)
    return save_path


# =============================================================================
# 5. Main Execution
# =============================================================================
def main(
        output_dir: str = OUTPUT_DIR,
        output_format: str = "jpeg",
) -> None:
    print("=== Generating Main Text Micro-Grid ===")

    global_rows = []
    required_keys = {m_id: set() for m_id, _, _, _, _ in MAIN_TEXT_SELECTION}

    # Extract target images
    for model_id, model_label, stratum, idx, join_key in MAIN_TEXT_SELECTION:
        row = get_stratified_image_for_model(
            model_id,
            stratum,
            idx,
            join_key=join_key,
        )
        if row is not None:
            global_rows.append({"model_id": model_id, "model_label": model_label, "row": row})
            required_keys[model_id].add(row["join_key"])
            print(f"[*] Found {stratum} image for {model_id}: {row['join_key']}")
        else:
            print(f"[!] Could not find valid data for {model_id} ({stratum})")

    # Load only necessary attributions
    attr_maps = {method: {} for method in MAIN_TEXT_METHODS}
    for method in MAIN_TEXT_METHODS:
        for model_id in required_keys:
            if not required_keys[model_id]: continue
            attr_maps[method][model_id] = load_attributions_robust(os.path.join(RESULTS_ROOT, method, model_id),
                                                                   required_keys[model_id])

    # Plot
    if global_rows:
        plot_main_text_micro_grid(
            global_rows,
            attr_maps,
            output_dir=output_dir,
            output_format=output_format,
        )
    else:
        print("[!] No images were successfully loaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the main-text qualitative micro-grid."
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--format",
        default="jpeg",
        choices=("jpeg", "jpg", "png", "pdf"),
        dest="output_format",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir, output_format=args.output_format)
