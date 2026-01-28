import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _np_trapz_compat(y: np.ndarray, dx: float = 1.0) -> float:
    """
    NumPy 2.x removed np.trapz; np.trapezoid is the replacement.
    This keeps compatibility across versions.
    """
    trap = getattr(np, "trapezoid", None)
    if trap is None:
        trap = np.trapz  # older NumPy
    return float(trap(y, dx=dx))


def _infer_device(model: torch.nn.Module) -> torch.device:
    """
    Robust device inference:
    - If model has parameters, use the first parameter device.
    - Else, fall back to cuda if available, else cpu.
    """
    try:
        p = next(model.parameters())
        return p.device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_3chw(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Normalize tensors:
    - Accept (C,H,W) or (1,C,H,W) and return (C,H,W)
    """
    if x.dim() == 4:
        if x.shape[0] != 1:
            raise ValueError(f"{name} must have batch size 1 if 4D; got shape={tuple(x.shape)}")
        return x.squeeze(0)
    if x.dim() == 3:
        return x
    raise ValueError(f"{name} must be 3D (C,H,W) or 4D (1,C,H,W); got shape={tuple(x.shape)}")


def _normalize_attribution(attr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Accept attribution in:
    - (H,W), (1,H,W), (C,H,W), (1,C,H,W)
    and return a per-pixel importance map of shape (H*W,) computed robustly.
    """
    # Remove batch if present
    if attr.dim() == 4:
        if attr.shape[0] != 1:
            raise ValueError(f"attribution must have batch size 1 if 4D; got shape={tuple(attr.shape)}")
        attr = attr.squeeze(0)

    if attr.dim() == 2:
        # (H,W)
        if attr.shape != x.shape[-2:]:
            raise ValueError(f"Attribution shape {tuple(attr.shape)} does not match image spatial shape {tuple(x.shape[-2:])}")
        attr_map = attr
    elif attr.dim() == 3:
        # Could be (1,H,W) or (C,H,W)
        if attr.shape[0] == 1 and attr.shape[1:] == x.shape[-2:]:
            attr_map = attr[0]
        elif attr.shape[1:] == x.shape[-2:]:
            # (C,H,W): reduce channels robustly
            attr_map = attr.abs().amax(dim=0)
        else:
            raise ValueError(
                f"Attribution shape {tuple(attr.shape)} is not compatible with image shape {tuple(x.shape)}"
            )
    else:
        raise ValueError(f"attribution must be 2D, 3D, or 4D; got shape={tuple(attr.shape)}")

    return attr_map.reshape(-1)


def _make_blur_baseline(x_batch: torch.Tensor, kernel_size: Optional[int] = None) -> torch.Tensor:
    """
    Create a blur baseline using avg_pool2d.
    kernel_size is chosen relative to image size if not provided.
    Ensures odd kernel, >=3, and <= min(H,W) (with padding).
    """
    _, _, h, w = x_batch.shape
    if kernel_size is None:
        # heuristic: ~1/10 of min dimension, odd, >=3
        k = max(3, int(min(h, w) / 10))
    else:
        k = int(kernel_size)

    # enforce odd
    if k % 2 == 0:
        k += 1

    # keep k reasonable (avg_pool with huge kernels can be slow)
    k = min(k, max(3, min(h, w) | 1))  # ensure odd

    pad = k // 2
    return F.avg_pool2d(x_batch, kernel_size=k, stride=1, padding=pad)


class CausalMetricScorer:
    """
    Computes insertion/deletion curves and AUC for a single image.

    Robustness improvements:
    - Handles various attribution shapes
    - Handles NumPy trapz removal (NumPy 2.x)
    - Safer baseline creation
    - Safer device handling
    - Optional progress bar
    """

    def __init__(
        self,
        model: torch.nn.Module,
        steps: int = 50,
        device: Optional[torch.device] = None,
        show_progress: bool = False,
        blur_kernel_size: Optional[int] = None,
    ):
        if steps <= 0:
            raise ValueError(f"steps must be > 0; got {steps}")

        self.model = model.eval()
        self.steps = int(steps)
        self.device = device or _infer_device(model)
        self.show_progress = bool(show_progress)
        self.blur_kernel_size = blur_kernel_size

        # Move model to device if it is on CPU but device is CUDA (or vice versa)
        # This is safe even if already on device.
        self.model.to(self.device)

    @torch.inference_mode()
    def score(
        self,
        x: torch.Tensor,
        attribution: torch.Tensor,
        baseline_mode: str = "blur",
        target_idx: Optional[int] = None,
    ) -> Dict[str, float]:

        x = _ensure_3chw(x, "x").to(self.device)
        x_batch = x.unsqueeze(0)  # (1,C,H,W)

        # Prepare/normalize attribution into flat per-pixel importance
        attr_flat = _normalize_attribution(attribution.to(self.device), x)

        if attr_flat.numel() != x.shape[-2] * x.shape[-1]:
            raise RuntimeError("Internal error: attr_flat size mismatch")

        # Determine target
        logits = self.model(x_batch)
        if logits.dim() != 2 or logits.shape[0] != 1:
            raise ValueError(f"Model output must be shape (1, num_classes); got {tuple(logits.shape)}")

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        if target_idx is None:
            target_idx = int(probs.argmax(dim=1).item())
        else:
            target_idx = int(target_idx)
            if not (0 <= target_idx < num_classes):
                raise ValueError(f"target_idx={target_idx} out of range [0, {num_classes-1}]")

        initial_prob = float(probs[0, target_idx].item())

        # Baseline
        baseline_mode = str(baseline_mode).lower()
        if baseline_mode == "blur":
            baseline = _make_blur_baseline(x_batch, kernel_size=self.blur_kernel_size).squeeze(0)
        elif baseline_mode in ("zero", "zeros", "black"):
            baseline = torch.zeros_like(x)
        else:
            raise ValueError(f"Unknown baseline_mode='{baseline_mode}'. Use 'blur' or 'zero'.")

        # Sort pixels by importance (descending)
        sorted_indices = torch.argsort(attr_flat, descending=True)

        deletion_curve = self._generate_curve(
            x=x,
            baseline=baseline,
            target=target_idx,
            sorted_indices=sorted_indices,
            mode="deletion",
        )

        insertion_curve = self._generate_curve(
            x=x,
            baseline=baseline,
            target=target_idx,
            sorted_indices=sorted_indices,
            mode="insertion",
        )

        dx = 1.0 / self.steps
        del_auc = _np_trapz_compat(deletion_curve, dx=dx)
        ins_auc = _np_trapz_compat(insertion_curve, dx=dx)

        return {
            "deletion_auc": del_auc,
            "insertion_auc": ins_auc,
            "initial_prob": initial_prob,
            "final_ins_prob": float(insertion_curve[-1]),
        }

    @torch.inference_mode()
    def _generate_curve(
        self,
        x: torch.Tensor,
        baseline: torch.Tensor,
        target: int,
        sorted_indices: torch.Tensor,
        mode: str,
    ) -> np.ndarray:

        if mode not in ("insertion", "deletion"):
            raise ValueError(f"mode must be 'insertion' or 'deletion'; got {mode}")

        c, h, w = x.shape
        total_pixels = h * w

        # Initialize canvas and source
        if mode == "insertion":
            canvas = baseline.clone()
            source = x
        else:
            canvas = x.clone()
            source = baseline

        canvas_flat = canvas.view(c, -1)
        source_flat = source.view(c, -1)

        # Step boundaries: ensure monotonic, cover [0,total_pixels]
        # Use torch to avoid dtype surprises and keep on CPU for indexing
        pixel_steps = torch.linspace(0, total_pixels, steps=self.steps + 1, dtype=torch.int64)

        curve = np.empty(self.steps + 1, dtype=np.float64)

        # Step 0
        out0 = F.softmax(self.model(canvas.unsqueeze(0)), dim=1)[0, target].item()
        curve[0] = out0

        it = range(1, self.steps + 1)
        if self.show_progress:
            it = tqdm(it, desc=mode, leave=False)

        for i in it:
            start_idx = int(pixel_steps[i - 1].item())
            end_idx = int(pixel_steps[i].item())

            if end_idx > start_idx:
                idx = sorted_indices[start_idx:end_idx]
                canvas_flat[:, idx] = source_flat[:, idx]

            out = F.softmax(self.model(canvas_flat.view(1, c, h, w)), dim=1)[0, target].item()
            curve[i] = out

        return curve


if __name__ == "__main__":
    # Minimal smoke test
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 10),
    )

    img = torch.randn(3, 224, 224)
    attr = torch.randn(1, 224, 224)

    scorer = CausalMetricScorer(model, steps=20, show_progress=False)
    scores = scorer.score(img, attr, target_idx=5)

    print(f"Insertion AUC: {scores['insertion_auc']:.4f}")
    print(f"Deletion AUC:  {scores['deletion_auc']:.4f}")
