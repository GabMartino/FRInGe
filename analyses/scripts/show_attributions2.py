import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",

    # Match the paper's Times-like look
    "text.latex.preamble": r"""
        \usepackage{newtxtext}
        \usepackage{newtxmath}
    """,

    # Reasonable sizes for ICML two-column figures
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})


def tex_escape(s: str) -> str:
    s = str(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


# =============================================================================
# Configuration
# =============================================================================
RESULTS_ROOT = "./results/methods"
IMAGE_ROOT = "./data/ImageNet"
OUTPUT_DIR = "./analyses/outputs"

MODELS_TO_RUN = [
    "resnet18",
    "resnet50",
    "resnet101",
    "resnet152",
    "inception_v3",
    "vgg19",
]

model_name_to_label = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "resnet152": "ResNet-152",
    "inception_v3": "Inception-V3",
    "vgg19": "VGG19",
}
OURS_METHOD = "FisherRaoIG/3_fisher_smooth"
BASELINE_METHODS = [
    "IG",
    "IG2",
    "GuidedIG",
    "SmoothGrad",
    "AdversarialIG",
    "GGIG",
    "GIG",
]

ALL_METHODS = [OURS_METHOD] + BASELINE_METHODS

TOP_K_QUAL = 5
ATTR_CLIP_Q = 0.99
JOIN_MODE = "basename"  # "basename" or "normalized_path"

METRIC_SPECS = {
    "mas_insertion_auc_mean": {"higher_is_better": True, "label": "MAS-Ins"},
    "mas_deletion_auc_mean": {"higher_is_better": False, "label": "MAS-Del"},
    "insertion_auc_mean": {"higher_is_better": True, "label": "Ins-AUC"},
    "deletion_auc_mean": {"higher_is_better": False, "label": "Del-AUC"},
}

GLOBAL_QUAL_WEIGHTS = {
    "mas_insertion_auc_mean": 1.0,
    "mas_deletion_auc_mean": 1.0,
    "insertion_auc_mean": 0.10,
    "deletion_auc_mean": 0.10,
}


# =============================================================================
# Utilities
# =============================================================================
def make_join_key(path: str, mode: str = "basename") -> str:
    path = str(path).replace("\\", "/").strip()
    if mode == "basename":
        return os.path.basename(path)
    if mode == "normalized_path":
        return os.path.normpath(path.lstrip("./")).replace("\\", "/")
    raise ValueError(f"Unsupported join mode: {mode}")


def sanitize_path_series(path_series: pd.Series, mode: str = "basename") -> pd.Series:
    return path_series.astype(str).apply(lambda p: make_join_key(p, mode=mode))


def resolve_image_path(root_image_path: str, image_path: str) -> str:
    image_path = str(image_path)
    if os.path.isabs(image_path):
        return image_path
    return os.path.join(root_image_path, image_path)


def clean_numeric(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return s.astype(float)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def coerce_attr_array(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.dtype == object:
        try:
            arr = np.stack(list(x))
        except Exception:
            arr = np.asarray(arr)
    return np.asarray(arr)


def preprocess_attributions(attr, clip_q: float = 0.99) -> np.ndarray:
    arr = coerce_attr_array(attr)

    if arr.ndim == 4:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            raise ValueError(f"Unsupported 4D attribution shape: {arr.shape}")

    if arr.ndim == 3:
        if arr.shape[0] in (1, 3):  # CHW
            heat = np.abs(arr).sum(axis=0)
        elif arr.shape[-1] in (1, 3):  # HWC
            heat = np.abs(arr).sum(axis=-1)
        else:
            raise ValueError(f"Unsupported 3D attribution shape: {arr.shape}")
    elif arr.ndim == 2:
        heat = np.abs(arr)
    else:
        raise ValueError(f"Unsupported attribution ndim: {arr.ndim}")

    if heat.size == 0:
        return np.zeros((1, 1), dtype=np.float32)

    q = np.nanquantile(heat.reshape(-1), clip_q)
    heat = np.clip(heat, 0.0, q)

    hmin = float(np.nanmin(heat))
    hmax = float(np.nanmax(heat))
    denom = hmax - hmin

    if not np.isfinite(denom) or denom < 1e-12:
        return np.zeros_like(heat, dtype=np.float32)

    heat = (heat - hmin) / denom
    return np.clip(heat, 0.0, 1.0).astype(np.float32)


def torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_attributions_robust(
        attrib_dir: str,
        target_keys: set,
        join_mode: str = "basename",
) -> Dict[str, np.ndarray]:
    subdir = os.path.join(attrib_dir, "attributions")
    if os.path.isdir(subdir):
        attrib_dir = subdir

    if not os.path.isdir(attrib_dir):
        return {}

    attr_map: Dict[str, np.ndarray] = {}

    for fname in os.listdir(attrib_dir):
        fpath = os.path.join(attrib_dir, fname)

        if fname.endswith(".pt"):
            try:
                data = torch_load_cpu(fpath)
                if not isinstance(data, dict):
                    continue

                filenames = data.get("filenames", data.get("image_paths", None))
                attrs = data.get("attr", data.get("attributions", None))
                if filenames is None or attrs is None or len(filenames) != len(attrs):
                    continue

                for raw_name, attr in zip(filenames, attrs):
                    key = make_join_key(raw_name, mode=join_mode)
                    if key in target_keys:
                        attr_map[key] = coerce_attr_array(attr)

            except Exception as e:
                print(f"[Warning] Could not read attribution file {fpath}: {e}")

        elif fname.endswith(".parquet"):
            try:
                df = pd.read_parquet(fpath)
                if not {"image_path", "attr"}.issubset(df.columns):
                    continue

                df = df[["image_path", "attr"]].copy()
                df["join_key"] = sanitize_path_series(df["image_path"], mode=join_mode)
                df = df[df["join_key"].isin(target_keys)]

                for _, row in df.iterrows():
                    attr_map[row["join_key"]] = coerce_attr_array(row["attr"])

            except Exception as e:
                print(f"[Warning] Could not read parquet attribution file {fpath}: {e}")

    return attr_map


# =============================================================================
# Metric loading
# =============================================================================
def load_method_metrics(metrics_dir: str, join_mode: str = "basename") -> pd.DataFrame:
    if not os.path.isdir(metrics_dir):
        return pd.DataFrame()

    parquet_files = [
        f for f in os.listdir(metrics_dir)
        if f.endswith(".parquet") and "metrics" in f
    ]
    if not parquet_files:
        return pd.DataFrame()

    frames = []
    for fname in parquet_files:
        fpath = os.path.join(metrics_dir, fname)
        try:
            frames.append(pd.read_parquet(fpath))
        except Exception as e:
            print(f"[Warning] Could not read metrics file {fpath}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    if df.empty or "image_path" not in df.columns:
        return pd.DataFrame()

    keep_cols = ["image_path"] + list(METRIC_SPECS.keys())
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    df["join_key"] = sanitize_path_series(df["image_path"], mode=join_mode)

    for metric in METRIC_SPECS:
        if metric not in df.columns:
            df[metric] = np.nan
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    agg_map = {"image_path": "first"}
    for metric in METRIC_SPECS:
        agg_map[metric] = "mean"

    return df.groupby("join_key", as_index=False).agg(agg_map)


def build_model_metric_store(
        results_root: str,
        model: str,
        methods: List[str],
        join_mode: str = "basename",
) -> Dict[str, pd.DataFrame]:
    store = {}
    for method in methods:
        method_dir = os.path.join(results_root, method, model)
        df = load_method_metrics(method_dir, join_mode=join_mode)
        if not df.empty:
            store[method] = df
    return store


# =============================================================================
# Summary table
# =============================================================================
def bootstrap_ci_mean(
        values: np.ndarray,
        n_boot: int = 2000,
        alpha: float = 0.05,
        seed: int = 0,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return values[0], values[0]

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)

    n = len(values)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()

    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


def build_summary_table(
        metric_store: Dict[str, pd.DataFrame],
        n_boot: int = 2000,
) -> pd.DataFrame:
    rows = []
    for method, df in metric_store.items():
        row = {"method": method, "n": len(df)}
        for metric, spec in METRIC_SPECS.items():
            vals = df[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            mean = float(np.mean(vals)) if len(vals) > 0 else np.nan
            lo, hi = bootstrap_ci_mean(vals, n_boot=n_boot)
            row[f"{metric}__mean"] = mean
            row[f"{metric}__ci_low"] = lo
            row[f"{metric}__ci_high"] = hi
        rows.append(row)

    out = pd.DataFrame(rows)
    for metric in METRIC_SPECS:
        out[f"{metric}__display"] = out.apply(
            lambda
                r: f"{r[f'{metric}__mean']:.4f} [{r[f'{metric}__ci_low']:.4f}, {r[f'{metric}__ci_high']:.4f}]" if pd.notna(
                r[f"{metric}__mean"]) else "nan",
            axis=1,
        )
    return out


# =============================================================================
# Global qualitative selection across all methods
# =============================================================================
def build_all_method_comparison_frame(
        metric_store: Dict[str, pd.DataFrame],
        ours_method: str,
        baseline_methods: List[str],
) -> pd.DataFrame:
    if ours_method not in metric_store:
        return pd.DataFrame()

    ours = metric_store[ours_method].copy()
    ours = ours.rename(columns={m: f"{m}__ours" for m in METRIC_SPECS})
    merged = ours

    for baseline in baseline_methods:
        if baseline not in metric_store:
            continue
        base = metric_store[baseline][["join_key"] + list(METRIC_SPECS.keys())].copy()
        base = base.rename(columns={m: f"{m}__{baseline}" for m in METRIC_SPECS})
        merged = pd.merge(merged, base, on="join_key", how="inner")

    if merged.empty:
        return pd.DataFrame()

    dominance_sum = pd.Series(0.0, index=merged.index, dtype=float)
    mas_both_wins = pd.Series(0, index=merged.index, dtype=int)
    num_valid_baselines = 0

    for baseline in baseline_methods:
        baseline_present = all(f"{m}__{baseline}" in merged.columns for m in METRIC_SPECS)
        if not baseline_present:
            continue
        num_valid_baselines += 1
        per_base_score = pd.Series(0.0, index=merged.index, dtype=float)

        for metric, spec in METRIC_SPECS.items():
            ours_col = f"{metric}__ours"
            base_col = f"{metric}__{baseline}"
            if spec["higher_is_better"]:
                delta = merged[ours_col] - merged[base_col]
            else:
                delta = merged[base_col] - merged[ours_col]

            merged[f"delta__{metric}__{baseline}"] = delta
            per_base_score += GLOBAL_QUAL_WEIGHTS.get(metric, 0.0) * delta.fillna(0.0)

        merged[f"dominance__{baseline}"] = per_base_score
        dominance_sum += per_base_score

        mas_ins_win = merged[f"delta__mas_insertion_auc_mean__{baseline}"] > 0
        mas_del_win = merged[f"delta__mas_deletion_auc_mean__{baseline}"] > 0
        mas_both_wins += (mas_ins_win & mas_del_win).astype(int)

    if num_valid_baselines == 0:
        return pd.DataFrame()

    merged["global_dominance_score"] = dominance_sum / num_valid_baselines
    merged["mas_both_win_fraction"] = mas_both_wins / num_valid_baselines
    merged["mas_both_win_count"] = mas_both_wins

    return merged


def select_stratified_qualitative_examples(
        all_method_df: pd.DataFrame,
        k_per_stratum: int = 2,
) -> pd.DataFrame:
    """
    Selects images representing the best, median, and worst relative
    performance of our method to ensure a fair, non-cherry-picked visual evaluation.
    """
    df = all_method_df.copy()

    if df.empty:
        return df

    # Sort strictly by the global dominance score
    df = df.sort_values("global_dominance_score", ascending=False).reset_index(drop=True)
    n = len(df)

    if n < k_per_stratum * 3:
        print("[Warning] Not enough samples for clean stratification. Returning all.")
        return df

    # Define indices for our strata
    top_indices = list(range(0, k_per_stratum))
    median_start = (n // 2) - (k_per_stratum // 2)
    median_indices = list(range(median_start, median_start + k_per_stratum))
    bottom_start = n - k_per_stratum
    bottom_indices = list(range(bottom_start, n))

    # Combine and label them for the plot
    selected_indices = top_indices + median_indices + bottom_indices
    stratified_df = df.iloc[selected_indices].copy()

    # Add a label column so you know what you are looking at in the plot
    labels = (["Top (Win)"] * k_per_stratum +
              ["Median (Typical)"] * k_per_stratum +
              ["Bottom (Failure)"] * k_per_stratum)
    stratified_df["stratum_label"] = labels

    return stratified_df


# =============================================================================
# Plot all methods in one panel
# =============================================================================
def get_geometry_preprocess_no_norm(model_name: str) -> T.Compose:
    if "inception" in model_name.lower():
        return T.Compose([T.Resize(342, antialias=True), T.CenterCrop(299)])
    return T.Compose([T.Resize(256, antialias=True), T.CenterCrop(224)])


def plot_all_method_qualitative_examples(
        selected_df: pd.DataFrame,
        geom_tfm: T.Compose,
        root_image_path: str,
        attr_maps: Dict[str, Dict[str, np.ndarray]],
        methods: List[str],
        model_name=str,
        save_path: Optional[str] = None,
        show: bool = True,
):
    if selected_df.empty:
        return

    n_rows = len(selected_df)
    n_cols = 1 + len(methods)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )

    axes[0, 0].set_title("Original", fontsize=25, fontweight="bold")
    for j, method in enumerate(methods, start=1):
        title = "Ours" if method == methods[0] else tex_escape(method)
        axes[0, j].set_title(title, fontsize=25, fontweight="bold")

    resampling = getattr(Image, "Resampling", Image).BILINEAR

    for i, (_, row) in enumerate(selected_df.iterrows()):
        join_key = str(row["join_key"])
        img_path = resolve_image_path(root_image_path, row["image_path"])

        ax_img = axes[i, 0]
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        target_h, target_w = 224, 224

        try:
            pil = Image.open(img_path).convert("RGB")
            pil = geom_tfm(pil)
            img_array = np.asarray(pil).astype(np.float32) / 255.0
            target_h, target_w = img_array.shape[:2]
            ax_img.imshow(img_array)
        except Exception:
            ax_img.text(0.5, 0.5, "Img Not Found", ha="center", va="center")
            ax_img.set_facecolor("black")

        # Apply the explicit strata labeling here
        stratum = str(row.get("stratum_label", f"Rank #{i + 1}"))
        score = row.get("global_dominance_score", 0.0)
        row_label = f"{stratum}\n$\\Delta$ Score: {score:.3f}"

        ax_img.set_ylabel(
            row_label,
            rotation=0,
            labelpad=50,
            va="center",
            fontsize=12,
            fontweight="bold"
        )

        for j, method in enumerate(methods, start=1):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])

            attr = attr_maps.get(method, {}).get(join_key)
            if attr is not None:
                try:
                    heat = preprocess_attributions(attr, clip_q=ATTR_CLIP_Q)
                    heat_pil = Image.fromarray((heat * 255).astype(np.uint8))
                    heat_pil = heat_pil.resize((target_w, target_h), resample=resampling)
                    ax.imshow(np.asarray(heat_pil) / 255.0, cmap="inferno", vmin=0.0, vmax=1.0)
                except Exception:
                    ax.text(0.5, 0.5, "Attr Err", ha="center", va="center")
                    ax.set_facecolor("black")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", color="gray")
                ax.set_facecolor("black")

            suffix = "__ours" if method == methods[0] else f"__{method}"
            text = (
                f"MI {row.get(f'mas_insertion_auc_mean{suffix}', np.nan):.3f}\n"
                f"MD {row.get(f'mas_deletion_auc_mean{suffix}', np.nan):.3f}\n"
                f"I  {row.get(f'insertion_auc_mean{suffix}', np.nan):.3f}\n"
                f"D  {row.get(f'deletion_auc_mean{suffix}', np.nan):.3f}"
            )
            ax.text(
                0.03,
                0.03,
                text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                color="white",
                fontsize=12,
                bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", boxstyle="round,pad=0.22"),
            )

    fig.suptitle(
        f"Stratified qualitative comparison across all methods for {model_name}",
        y=0.995,
        fontsize=25,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        extension = os.path.splitext(save_path)[1].lower()
        output_format = {
            ".jpeg": "jpg",
            ".jpg": "jpg",
            ".png": "png",
            ".pdf": "pdf",
        }.get(extension)
        if output_format is None:
            raise ValueError(
                f"Unsupported qualitative-figure extension: {extension!r}."
            )
        fig.savefig(
            save_path,
            dpi=160,
            bbox_inches="tight",
            format=output_format,
        )

    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Main analysis pipeline
# =============================================================================
def main(
        output_dir: str = OUTPUT_DIR,
        output_format: str = "jpeg",
        write_tables: bool = True,
        show: bool = True,
) -> None:
    """Generate the per-architecture qualitative comparison figures.

    PDF output keeps labels and telemetry as vector text while rasterizing only
    the underlying images and heatmaps. This is preferable for manuscript use.
    """
    normalized_format = output_format.lower().lstrip(".")
    extension_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "pdf": "pdf"}
    if normalized_format not in extension_map:
        raise ValueError("output_format must be one of: jpeg, jpg, png, pdf.")
    extension = extension_map[normalized_format]

    ensure_dir(output_dir)

    for model in MODELS_TO_RUN:
        print(f"\n=== MODEL: {model} ===")

        model_out_dir = os.path.join(output_dir, model)
        ensure_dir(model_out_dir)

        metric_store = build_model_metric_store(
            results_root=RESULTS_ROOT,
            model=model,
            methods=ALL_METHODS,
            join_mode=JOIN_MODE,
        )

        if OURS_METHOD not in metric_store:
            print(f"[Warning] Missing our method for model={model}, skipping.")
            continue

        all_method_df = build_all_method_comparison_frame(
            metric_store=metric_store,
            ours_method=OURS_METHOD,
            baseline_methods=BASELINE_METHODS,
        )

        # Utilize the stratified selection logic
        selected_global = select_stratified_qualitative_examples(
            all_method_df=all_method_df,
            k_per_stratum=2
        )

        target_keys = set(selected_global["join_key"].tolist())
        attr_maps = {}

        for method in ALL_METHODS:
            method_dir = os.path.join(RESULTS_ROOT, method, model)
            attr_maps[method] = load_attributions_robust(
                attrib_dir=method_dir,
                target_keys=target_keys,
                join_mode=JOIN_MODE,
            )

        geom_tfm = get_geometry_preprocess_no_norm(model)

        plot_all_method_qualitative_examples(
            selected_df=selected_global,
            geom_tfm=geom_tfm,
            root_image_path=IMAGE_ROOT,
            attr_maps=attr_maps,
            methods=ALL_METHODS,
            model_name=model_name_to_label.get(model, model),
            save_path=os.path.join(
                model_out_dir,
                f"qualitative_all_methods_{model}.{extension}",
            ),
            show=show,
        )

        if write_tables:
            summary_df = build_summary_table(metric_store, n_boot=1000)
            summary_csv = os.path.join(model_out_dir, "summary_table.csv")
            summary_df.to_csv(summary_csv, index=False)
            print(f"[Saved] {summary_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate stratified qualitative attribution figures."
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--format",
        default="jpeg",
        choices=("jpeg", "jpg", "png", "pdf"),
        dest="output_format",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Do not emit the auxiliary summary CSV files.",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    main(
        output_dir=args.output_dir,
        output_format=args.output_format,
        write_tables=not args.figures_only,
        show=not args.no_show,
    )
