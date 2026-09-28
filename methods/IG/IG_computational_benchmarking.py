import os
import time
import argparse
import numpy as np
import pandas as pd
import torch

from captum.attr import IntegratedGradients
from common.utils import load_model, load_image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class IGWrapper:
    def __init__(self, model, n_steps=300, method="gausslegendre", internal_batch_size=None):
        self.model = model.eval()
        self.n_steps = int(n_steps)
        self.method = method
        self.internal_batch_size = internal_batch_size
        self.device = next(model.parameters()).device
        self.ig = IntegratedGradients(self.model)

    def attribute(self, inputs: torch.Tensor, target_idx: torch.Tensor, seed=None):
        """
        Returns:
            attr: attribution tensor
            delta: Captum convergence delta
        """
        inputs = inputs.to(self.device)
        target_idx = target_idx.to(self.device)

        baselines = torch.zeros_like(inputs)

        attr, delta = self.ig.attribute(
            inputs=inputs,
            baselines=baselines,
            target=target_idx,
            n_steps=self.n_steps,
            method=self.method,
            internal_batch_size=1,
            return_convergence_delta=True,
        )

        return attr.detach(), delta.detach()


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image timing benchmark for Captum Integrated Gradients.")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--data-dir", type=str, default="./data/ImageNet/")
    parser.add_argument("--subset-size", type=int, default=4,
                        help="Number of images to benchmark. Ignored if --indices is provided.")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start index in the sorted image list.")
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="Explicit indices to benchmark, e.g. --indices 0 3 5 9")
    parser.add_argument("--output-dir", type=str, default="./results/runtime/ig")
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--ig-method", type=str, default="gausslegendre")
    parser.add_argument("--internal-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_image_files(base_path: str):
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Directory not found: {base_path}")

    files = []
    for name in sorted(os.listdir(base_path)):
        full_path = os.path.join(base_path, name)
        ext = os.path.splitext(name)[1].lower()
        if os.path.isfile(full_path) and ext in IMAGE_EXTS:
            files.append(name)

    if not files:
        raise RuntimeError(f"No image files found in: {base_path}")

    return files


def select_indices(num_files: int, subset_size: int, start_index: int, explicit_indices=None):
    if explicit_indices is not None and len(explicit_indices) > 0:
        indices = explicit_indices
    else:
        if subset_size <= 0:
            raise ValueError("--subset-size must be > 0")
        if start_index < 0:
            raise ValueError("--start-index must be >= 0")
        end_index = min(start_index + subset_size, num_files)
        indices = list(range(start_index, end_index))

    if not indices:
        raise ValueError("No indices selected.")

    for idx in indices:
        if idx < 0 or idx >= num_files:
            raise IndexError(f"Selected index {idx} is out of range [0, {num_files - 1}]")

    return indices


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def scalar_or_none(x):
    x_np = to_numpy(x)
    if x_np.size == 1:
        return float(x_np.reshape(-1)[0])
    return None


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    deltas_dir = os.path.join(args.output_dir, "deltas")
    os.makedirs(deltas_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, transformation = load_model(args.model, device=device)
    model.eval()

    attr_method = IGWrapper(
        model=model,
        n_steps=args.n_steps,
        method=args.ig_method,
        internal_batch_size=args.internal_batch_size,
    )

    all_image_files = list_image_files(args.data_dir)
    selected_indices = select_indices(
        num_files=len(all_image_files),
        subset_size=args.subset_size,
        start_index=args.start_index,
        explicit_indices=args.indices,
    )

    selected_paths = [all_image_files[i] for i in selected_indices]

    print(f"Model:               {args.model}")
    print(f"Device:              {device.type.upper()}")
    print(f"Dataset dir:         {args.data_dir}")
    print(f"Total images:        {len(all_image_files)}")
    print(f"Selected indices:    {selected_indices}")
    print(f"Selected files:      {selected_paths}")
    print(f"IG n_steps:          {args.n_steps}")
    print(f"IG method:           {args.ig_method}")
    print(f"Internal batch size: {args.internal_batch_size}")
    print("-" * 70)

    img_tensor_list = [
        load_image(os.path.join(args.data_dir, p), transformation, device=device)
        for p in selected_paths
    ]
    batch = torch.cat(img_tensor_list, dim=0)

    with torch.no_grad():
        logits = model(batch)
        target_idx = logits.argmax(dim=1)

    # --- WARM-UP ON A SINGLE IMAGE ---
    print(f"Executing warm-up for {args.model} on first selected image...")

    dummy_img = batch[0:1].clone()
    dummy_tgt = target_idx[0:1]

    _dummy_attr, _dummy_delta = attr_method.attribute(dummy_img, dummy_tgt, seed=args.seed)

    # --- TIMING ---
    print(f"\nRunning timed attribution sequentially on {len(selected_indices)} image(s)...")
    print("-" * 70)

    results_rows = []
    execution_times = []

    for local_i, global_idx in enumerate(selected_indices):
        single_img = batch[local_i:local_i + 1]   # shape = (1, C, H, W)
        single_tgt = target_idx[local_i:local_i + 1]

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        attr_map, delta = attr_method.attribute(single_img, single_tgt, seed=args.seed)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        img_wall_time = end_time - start_time
        execution_times.append(img_wall_time)

        delta_np = to_numpy(delta)
        delta_value = scalar_or_none(delta_np)

        filename_stem = os.path.splitext(os.path.basename(selected_paths[local_i]))[0]
        delta_file = os.path.join(
            deltas_dir,
            f"{local_i:03d}_global{global_idx:05d}_{filename_stem}_delta.npy"
        )
        np.save(delta_file, delta_np)

        row = {
            "local_order": local_i,
            "global_index": global_idx,
            "filename": selected_paths[local_i],
            "target_idx": int(single_tgt.item()),
            "batch_size_used_for_timing": int(single_img.shape[0]),
            "execution_time_sec": float(img_wall_time),
            "delta_value": delta_value,
            "delta_shape": str(delta_np.shape),
            "delta_file": delta_file,
            "n_steps": int(args.n_steps),
            "ig_method": args.ig_method,
            "internal_batch_size": args.internal_batch_size,
        }
        results_rows.append(row)

        if delta_value is not None:
            delta_msg = f"{delta_value:.8f}"
        else:
            delta_msg = f"non-scalar, shape={delta_np.shape}"

        print(
            f"Image {local_i + 1}/{len(selected_indices)} "
            f"(Global Index {global_idx:3d}, File '{selected_paths[local_i]}', "
            f"Batch Size {single_img.shape[0]}) "
            f"Execution Time: {img_wall_time:.4f} sec | Delta: {delta_msg}"
        )

        del attr_map, delta

    execution_times = np.array(execution_times, dtype=float)

    valid_delta_values = [r["delta_value"] for r in results_rows if r["delta_value"] is not None]

    summary = {
        "model": args.model,
        "device": device.type.upper(),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "num_images": int(len(execution_times)),
        "timing_batch_size": 1,
        "total_loop_time_sec": float(execution_times.sum()),
        "mean_per_image_sec": float(execution_times.mean()),
        "std_per_image_sec": float(execution_times.std()),
        "min_per_image_sec": float(execution_times.min()),
        "max_per_image_sec": float(execution_times.max()),
        "mean_delta_value": float(np.mean(valid_delta_values)) if len(valid_delta_values) > 0 else np.nan,
        "mean_abs_delta_value": float(np.mean(np.abs(valid_delta_values))) if len(valid_delta_values) > 0 else np.nan,
        "data_dir": args.data_dir,
        "n_steps": int(args.n_steps),
        "ig_method": args.ig_method,
        "internal_batch_size": args.internal_batch_size,
        "start_index": args.start_index,
        "subset_size_requested": args.subset_size,
        "explicit_indices": " ".join(map(str, args.indices)) if args.indices else "",
        "deltas_dir": deltas_dir,
    }

    per_image_df = pd.DataFrame(results_rows)
    summary_df = pd.DataFrame([summary])

    per_image_csv = os.path.join(args.output_dir, f"{args.model}_captum_ig_per_image_times.csv")
    summary_csv = os.path.join(args.output_dir, f"{args.model}_captum_ig_summary.csv")

    per_image_df.to_csv(per_image_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("-" * 70)
    print(f"SEQUENTIAL BENCHMARK RESULTS: {args.model.upper()} | CAPTUM IG")
    print("-" * 70)
    print(f"Hardware:          {device.type.upper()}")
    if device.type == "cuda":
        print(f"GPU:               {torch.cuda.get_device_name(0)}")
    print(f"Num images:        {len(execution_times)}")
    print(f"Timing batch size: 1")
    print(f"IG n_steps:        {args.n_steps}")
    print(f"Total Loop Time:   {execution_times.sum():.4f} seconds")
    print(f"Mean Per-Image:    {execution_times.mean():.4f} seconds")
    print(f"Std Deviation:     {execution_times.std():.4f} seconds")
    print(f"Min / Max:         {execution_times.min():.4f} / {execution_times.max():.4f} seconds")
    print("-" * 70)
    print(f"Saved per-image CSV: {per_image_csv}")
    print(f"Saved summary CSV:   {summary_csv}")
    print(f"Saved delta files in: {deltas_dir}")
