import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def repair_mas_means_inplace(df: pd.DataFrame):
    # find baselines present
    ins_cols = [c for c in df.columns if c.startswith("mas_insertion_auc__") and "__[" not in c]
    del_cols = [c for c in df.columns if c.startswith("mas_deletion_auc__") and "__[" not in c]
    val_cols = [c for c in df.columns if c.startswith("mas_valid__") and "__[" not in c]

    # need matching baselines
    baselines = []
    for c in ins_cols:
        baselines.append(c.split("mas_insertion_auc__")[1])

    if not baselines:
        return df

    # mask invalid -> NaN
    for b in baselines:
        vcol = f"mas_valid__{b}"
        if vcol in df.columns:
            valid = df[vcol].astype(bool)
            icol = f"mas_insertion_auc__{b}"
            dcol = f"mas_deletion_auc__{b}"
            if icol in df.columns:
                df.loc[~valid, icol] = np.nan
            if dcol in df.columns:
                df.loc[~valid, dcol] = np.nan

    # recompute means if missing or if you want to enforce correctness
    ins_present = [f"mas_insertion_auc__{b}" for b in baselines if f"mas_insertion_auc__{b}" in df.columns]
    del_present = [f"mas_deletion_auc__{b}" for b in baselines if f"mas_deletion_auc__{b}" in df.columns]

    if ins_present:
        df["mas_insertion_auc_mean"] = df[ins_present].mean(axis=1, skipna=True)
    if del_present:
        df["mas_deletion_auc_mean"] = df[del_present].mean(axis=1, skipna=True)

    return df


# --- 1. Style Helper ---
def set_icml_matplotlib_style(base_fontsize=12, tick_fontsize=10, legend_fontsize=10, title_fontsize=12,
                              line_width=1.5):
    """
    Sets Matplotlib style to mimic ICML/LaTeX standards (Times New Roman).
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": base_fontsize,
        "axes.titlesize": title_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
        "legend.fontsize": legend_fontsize,
        "lines.linewidth": line_width,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })


def plot_icml_grouped_by_model(
        df,
        save_path="icml_batch_score_by_model.pdf",
        model_rename_map=None,
        # --- NEW: ordering controls ---
        model_order=None,  # e.g. ["ResNet-18","InceptionV3",...]
        method_order=None,  # e.g. ["FRInGe (ours)","GuidedIG","IG",...]
        order_methods_by_overall_score=False,  # if True and method_order is None -> auto sort
        # --- NEW: palette controls ---
        palette_name="colorblind",  # "colorblind", "deep", "muted", "Set2", ...
):
    plot_df = df[df["Metric"] == "batch_score"].copy()

    if model_rename_map:
        plot_df["Model"] = plot_df["Model"].replace(model_rename_map)

    val_col = "RawResult" if "RawResult" in plot_df.columns else "Result"
    plot_df[val_col] = pd.to_numeric(plot_df[val_col], errors="coerce")

    # ---------- ORDERING ----------
    if model_order is None:
        # keep "as appears" order (often nicer than alphabetical)
        model_order = list(pd.unique(plot_df["Model"]))

    if method_order is None:
        if order_methods_by_overall_score:
            # order by overall mean batch_score (descending)
            method_order = (
                plot_df.groupby("Method")[val_col]
                .mean()
                .sort_values(ascending=False)
                .index.tolist()
            )
        else:
            method_order = list(pd.unique(plot_df["Method"]))

    # Make ordering “stick” (important if you do any sorting later)
    plot_df["Model"] = pd.Categorical(plot_df["Model"], categories=model_order, ordered=True)
    plot_df["Method"] = pd.Categorical(plot_df["Method"], categories=method_order, ordered=True)

    # ---------- STYLE ----------
    # Avoid seaborn global defaults fighting your rcParams: set theme lightly.
    sns.set_theme(style="whitegrid", context="paper")  # "paper" reads more like a publication

    # Colorblind-safe, publication-friendly palette
    palette = sns.color_palette(palette_name, n_colors=len(method_order))

    fig, ax = plt.subplots(figsize=(22, 5))
    sns.barplot(
        data=plot_df,
        x="Model",
        y=val_col,
        hue="Method",
        order=model_order,
        hue_order=method_order,
        palette=palette,
        # Lighter, less “boxy” bars for papers
        edgecolor=(0, 0, 0, 0.55),
        linewidth=0.6,
        ax=ax
    )

    # Subtle grid: y-only, light
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    ax.grid(False, axis="x")

    ax.set_ylabel("Aggregated Score")
    ax.set_xlabel("")
    ax.axhline(0, color="black", linewidth=1.0)

    # ---------- STAR LOGIC (your original, kept) ----------
    best_scores = plot_df.groupby("Model")[val_col].max().to_dict()
    x_labels = [label.get_text() for label in ax.get_xticklabels()]

    for patch in ax.patches:
        height = patch.get_height()
        if np.isnan(height):
            continue

        c_x = patch.get_x() + patch.get_width() / 2
        group_idx = int(round(c_x))
        if 0 <= group_idx < len(x_labels):
            model_name = x_labels[group_idx]
            max_val = best_scores.get(model_name, -np.inf)
            if abs(height - max_val) < 1e-6:
                is_negative = height < 0
                offset_dir = -1 if is_negative else 1
                va_align = "top" if is_negative else "bottom"
                y_offset = height + (offset_dir * (0.05 * abs(height) + 0.02))
                ax.text(
                    c_x, y_offset, "★",
                    ha="center", va=va_align,
                    color="black", fontsize=16, weight="bold"
                )

    # Legend: white background, thin border (reads clean in PDFs)
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=min(7, len(method_order)),
        frameon=True,
        title=None,
        borderpad=0.4,
        handlelength=1.2,
        columnspacing=1.0,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor((0, 0, 0, 0.25))
    leg.get_frame().set_linewidth(0.6)

    sns.despine(ax=ax, left=True, bottom=False)
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


# --- 3. Data Processing ---
def print_metrics_table(results_dict, model_rename_map=None):
    rows = []

    for method, models in results_dict.items():
        # Clean method names for ICML standards
        method_label = method

        for model_name, metrics in models.items():
            # Helper to safely extract means from the nested stats dict
            def get_mean(k):
                val = metrics.get(k, {}).get('mean')
                # Return NaN if missing or non-numeric to avoid silent errors
                return float(val) if val is not None and not np.isnan(val) else np.nan

            # 1. Individual Metrics Extraction
            for metric_name, stats in metrics.items():
                m = stats.get('mean')
                low = stats.get('ci_lower_bound')
                high = stats.get('ci_upper_bound')

                if m is None or low is None or high is None or np.isnan(m):
                    formatted_val = "∅"
                    raw_val = np.nan
                else:
                    formatted_val = f"{m:.3f} ± {(high - low) / 2:.3f}"
                    raw_val = m

                rows.append({
                    "Method": method_label,
                    "Model": model_name,
                    "Metric": metric_name,
                    "Result": formatted_val,
                    "RawResult": raw_val
                })

    df = pd.DataFrame(rows)

    # Set style
    set_icml_matplotlib_style(base_fontsize=15, tick_fontsize=18)
    desired_model_order = ["InceptionV3", "ResNet-152", "ResNet-101", "ResNet-50"]  # after renaming
    desired_method_order = ["FRInGe (ours)", "GuidedIG", "IG", "IG2", "AIG", "SmoothGrad"]
    # Plot with renaming
    # plot_icml_grouped_by_model(
    #     df,
    #     model_rename_map=model_rename_map,
    #     model_order=desired_model_order,
    #     method_order=desired_method_order,
    #     palette_name="colorblind",
    # )

    pivot_df = df.pivot(index=["Model", "Metric"], columns="Method", values="Result").fillna("N/A")
    print(pivot_df.to_string())
    return pivot_df


# --- 4. Helpers ---
def bootstrap_interval(data, confidence=0.95, n_bootstraps=2000, seed=42):
    rng = np.random.default_rng(seed)
    data = np.array(data)
    if len(data) == 0: return np.nan, np.nan, np.nan
    indices = rng.integers(0, len(data), size=(n_bootstraps, len(data)))
    resampled_data = data[indices]
    bootstrap_means = np.mean(resampled_data, axis=1)

    alpha = 1 - confidence
    lower, upper = np.percentile(bootstrap_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return np.mean(data), lower, upper


def main():
    # --- HARD LIMITER FOR IMAGES ---
    N_IMAGES_TO_KEEP = 1001

    results_path = "./results/methods/"

    avail_methods = ["FisherRaoIG"]#, "GIG", "GGIG"]#, "IG", "IG2", "SmoothGrad", "AdversarialIG", "GuidedIG"]

    # Dynamic model discovery intersection (simplified to avoid checking nested FisherRao directories incorrectly)
    avail_models = ["resnet18", "resnet50", "resnet101", "resnet152", "inception_v3", "vgg19"]
    print(f"Using models: {avail_models}")

    results = {}

    # --- FLATTENING LOGIC (Replacing Recursion) ---
    # We create a list of tuples: (Display Name, Base Folder, Sub Folder)
    expanded_methods = []
    for method in avail_methods:
        if method == "FisherRaoIG":
            expanded_methods.extend([
                #("FisherRaoIG_1_euclidean", "FisherRaoIG", "1_euclidean"),
                #("FisherRaoIG_2_fisher_raw", "FisherRaoIG", "2_fisher_raw"),
                ("FisherRaoIG_3_fisher_smooth", "FisherRaoIG", "3_fisher_smooth"),
                ("FisherRaoIG_3_fisher_smooth_binary", "FisherRaoIG", "fringe2_binary_binary"),
            ])
        else:
            expanded_methods.append((method, method, None))

    # Iterate over our newly flattened list
    for display_name, base_method, submethod in expanded_methods:
        print("Method: ", display_name)
        results[display_name] = {}

        for model in avail_models:
            print(" \t Model: ", model)
            results[display_name][model] = {}

            # Construct the target directory based on whether a submethod exists
            if submethod:
                target_dir = os.path.join(results_path, base_method, submethod, model)
            else:
                target_dir = os.path.join(results_path, base_method, model)

            results_metrics_path = os.path.join(target_dir, "results_metrics.parquet")

            # --- Robust File Loading ---
            try:
                results_metrics = pd.read_parquet(results_metrics_path)
                results_metrics = repair_mas_means_inplace(results_metrics)
            except FileNotFoundError:
                if not os.path.exists(target_dir):
                    print(f"\t -> Directory not found: {target_dir}")
                    continue

                results_metrics_paths = os.listdir(target_dir)
                # Concatenate all parquet files in folder
                dfs = [pd.read_parquet(os.path.join(target_dir, file))
                       for file in results_metrics_paths if "metrics" in file and file.endswith(".parquet")]

                if not dfs:
                    print("\t -> No metrics parquet files found in", target_dir)
                    continue

                results_metrics = pd.concat(dfs, ignore_index=True)

                # Deduplicate
                if "image_path" in results_metrics.columns:
                    results_metrics = results_metrics[
                        ~results_metrics["image_path"].duplicated(keep=False)].reset_index(drop=True)

            if results_metrics.empty:
                continue

            # --- DETERMINISTIC TRUNCATION ---
            if "image_path" in results_metrics.columns:
                results_metrics = results_metrics.sort_values(by="image_path").reset_index(drop=True)
            results_metrics = results_metrics.head(N_IMAGES_TO_KEEP)

            req_cols = ["insertion_auc__blur", "deletion_auc__blur", "infidelity_score"]
            if all(c in results_metrics.columns for c in req_cols):
                ins_auc = results_metrics["insertion_auc__blur"]
                dele = results_metrics["deletion_auc__blur"]
                inf = results_metrics["infidelity_score"]
                eps = 1e-6
                inv_del = 1.0 - dele
                fidelity = np.exp(-2 * inf)
                w_ins = 1.0
                w_del = 1.0
                w_fid = 0.5
                numerator = w_ins + w_del + w_fid
                denominator = (w_ins / (ins_auc + eps)) + (w_del / (inv_del + eps)) + (w_fid / (fidelity + eps))

                results_metrics["batch_score"] = numerator / denominator

            # --- Calculate Stats ---
            columns_metrics = [
                'insertion_auc_mean', 'deletion_auc_mean', 'insertion_auc_raw_mean',
                'deletion_auc_raw_mean', 'mas_insertion_auc_mean',
                'mas_deletion_auc_mean', 'infidelity_score', 'sparseness_score',
                'insertion_auc__blur', 'deletion_auc__blur', 'insertion_auc_raw__blur',
                'deletion_auc_raw__blur', 'insertion_auc__black', 'deletion_auc__black',
                'insertion_auc_raw__black', 'deletion_auc_raw__black',
                'insertion_auc__white', 'deletion_auc__white',
                'insertion_auc_raw__white', 'deletion_auc_raw__white',
                'mas_insertion_auc__blur', 'batch_score'
            ]

            for metric in columns_metrics:
                if metric not in results_metrics.columns:
                    results[display_name][model][metric] = {
                        "metric": metric, "mean": None, "ci_lower_bound": None,
                        "ci_upper_bound": None, "n": 0, "status": "missing_column",
                    }
                    continue

                values = results_metrics[metric].dropna().values

                if len(values) == 0:
                    results[display_name][model][metric] = {
                        "metric": metric, "mean": None, "ci_lower_bound": None,
                        "ci_upper_bound": None, "n": 0, "status": "empty_or_all_nan",
                    }
                    continue

                mean, low, up = bootstrap_interval(values, confidence=0.95)

                results[display_name][model][metric] = {
                    "metric": metric,
                    "mean": float(mean),
                    "ci_lower_bound": float(low),
                    "ci_upper_bound": float(up),
                    "n": int(len(values)),
                    "status": "ok",
                }

    # --- RENAME MAP ---
    my_model_renames = {
        "resnet50": "ResNet-50",
        "resnet101": "ResNet-101",
        "resnet152": "ResNet-152",
        "inception_v3": "InceptionV3",
    }

    if results:
        print_metrics_table(results, model_rename_map=my_model_renames)


if __name__ == "__main__":
    main()
