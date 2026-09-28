import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
results_path = "./results/methods/"
avail_models = ["resnet18"]

avail_methods = [
    'BiharmonicFisherRaoMaxEntropyAscent', 'IG2', 'IG', 'GuidedIG', 'BlurIG', 'AIG', 'SmoothGrad'
]

# Format: 'column_name': ('Plot Title', True_if_Higher_is_Better)
METRICS_CONFIG = {
    'insertion_auc': ('Insertion AUC', True),
    'deletion_auc': ('Deletion AUC', False),
    'mas_insertion_auc': ('MAS Insertion', True),
    'mas_deletion_auc': ('MAS Deletion', False),
    'infidelity_score': ('Infidelity', False),
    'sparseness_score': ('Sparseness', True)
}


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_mean_only(data):
    """Robustly flattens nested data to return a single mean float."""
    try:
        flat_data = np.hstack(data)
        flat_data = np.array(flat_data, dtype=float)
        flat_data = flat_data[~np.isnan(flat_data)]
        if len(flat_data) == 0: return 0.0
        return float(np.mean(flat_data))
    except Exception:
        return 0.0


# ==========================================
# 3. DATA EXTRACTION
# ==========================================
plot_data = []

for method in avail_methods:
    for model in avail_models:
        method_model_path = os.path.join(results_path, method, model)
        files = os.listdir(method_model_path)
        files = [f for f in files if "results_metrics" in f]

        try:
            if len(files) == 1:
                df_raw = pd.read_parquet(os.path.join(method_model_path, files[0]))
            else:
                df_raw = pd.concat([pd.read_parquet(os.path.join(method_model_path, f)) for f in files])
        except Exception:
            continue

        for col_name in METRICS_CONFIG.keys():
            if col_name not in df_raw.columns: continue
            vals = df_raw[col_name].dropna().values
            if len(vals) == 0: continue

            plot_data.append({
                "Method": method,
                "Model": model,
                "Metric_Key": col_name,
                "Mean": get_mean_only(vals)
            })

if not plot_data:
    raise ValueError("No data found. Check paths.")

df = pd.DataFrame(plot_data)

# ==========================================
# 4. RANKING LOGIC (Winner & Runner-Up)
# ==========================================
df['RankStatus'] = 'Other'  # Default

for key, (_, higher_is_better) in METRICS_CONFIG.items():
    mask = df['Metric_Key'] == key
    subset = df[mask]
    if subset.empty: continue

    # Sort unique values to find 1st and 2nd best scores
    unique_means = sorted(subset['Mean'].unique())
    if higher_is_better:
        unique_means = unique_means[::-1]  # Descending (Max is best)

    # Identify Score Thresholds
    best_score = unique_means[0]
    second_best_score = unique_means[1] if len(unique_means) > 1 else None

    # Assign Status (using epsilon for float comparison)
    # Winner
    df.loc[mask & (np.abs(df['Mean'] - best_score) < 1e-9), 'RankStatus'] = 'Winner'

    # Second Place (if exists)
    if second_best_score is not None:
        df.loc[mask & (np.abs(df['Mean'] - second_best_score) < 1e-9), 'RankStatus'] = 'RunnerUp'

# ==========================================
# 5. ICML-STYLE PLOTTING
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    # Increased Height slightly to fit legend below
    "figure.figsize": (14, 3.5)
})

active_metrics = [k for k in METRICS_CONFIG.keys() if k in df['Metric_Key'].unique()]
num_metrics = len(active_metrics)

# Create Subplots
fig, axes = plt.subplots(1, num_metrics, figsize=(num_metrics * 2.5, 4))
if num_metrics == 1: axes = [axes]

# Colors
unique_methods = sorted(df['Method'].unique())
palette = sns.color_palette("tab10", n_colors=len(unique_methods))
method_color_map = dict(zip(unique_methods, palette))

# ---------------------------------------------------------
# LAYOUT FIX: Reserve space at bottom for Legend & Label
# ---------------------------------------------------------
fig.subplots_adjust(bottom=0.35, wspace=0.3)

for ax, metric_key in zip(axes.flat, active_metrics):
    custom_title, _ = METRICS_CONFIG[metric_key]
    data_sub = df[df['Metric_Key'] == metric_key]

    # Plot Bars
    for i, (_, row) in enumerate(data_sub.iterrows()):
        status = row['RankStatus']

        # --- VISUAL HIERARCHY ---
        if status == 'Winner':
            alpha_val = 1.0
            edge_width = 2.0
            edge_col = 'black'
            z_order = 3
        elif status == 'RunnerUp':
            alpha_val = 0.65  # Distinctly lighter than winner, but visible
            edge_width = 1.0
            edge_col = 'black'
            z_order = 2
        else:
            alpha_val = 0.25  # Faded background
            edge_width = 0.5
            edge_col = 'gray'
            z_order = 1

        ax.bar(
            i,
            row['Mean'],
            color=method_color_map[row['Method']],
            alpha=alpha_val,
            edgecolor=edge_col,
            linewidth=edge_width,
            width=0.75,
            zorder=z_order
        )

    ax.set_xticks([])
    ax.set_title(custom_title, fontweight='bold', pad=10)

    # Minimalist Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

# ==========================================
# 6. LABELS & LEGEND
# ==========================================

# Big X-Axis Label (Model Name) - Positioned in the reserved bottom space
model_label = avail_models[0].upper() if avail_models else "MODEL"
fig.supxlabel(f"Evaluated Model: {model_label}", fontsize=12, fontweight='bold', y=0.02)

# Legend - Positioned between charts and Model Label
handles = [plt.Rectangle((0, 0), 1, 1, color=method_color_map[m]) for m in unique_methods]
fig.legend(
    handles,
    unique_methods,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.12),  # Coordinates relative to figure (0,0 is bottom-left)
    ncol=min(len(unique_methods), 5),
    frameon=False
)

# Save with tight bounding box to trim EXTRA white space,
# but keep the bottom margin we reserved.
plt.savefig("icml_custom_metrics.pdf", bbox_inches='tight')
plt.show()
