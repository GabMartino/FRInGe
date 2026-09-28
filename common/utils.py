import json
import os
from typing import Iterable, Optional, List

import numpy as np
import torch
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from torchvision.models import resnet18, resnet50, vgg19, inception_v3, VGG19_Weights, Inception_V3_Weights, \
    ResNet50_Weights, ResNet18_Weights, ResNet152_Weights, resnet152, ResNet101_Weights, resnet101, vit_b_16, \
    ViT_B_16_Weights
from torchvision.transforms import transforms, InterpolationMode

def load_model(model_name: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    registry = {
        "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
        "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
        "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V1),
        "resnet152": (resnet152, ResNet152_Weights.IMAGENET1K_V1),
        "vgg19": (vgg19, VGG19_Weights.IMAGENET1K_V1),
        "inception_v3": (inception_v3, Inception_V3_Weights.IMAGENET1K_V1),
        "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
    }

    if model_name not in registry:
        valid = ", ".join(registry.keys())
        raise ValueError(f"Invalid model name '{model_name}'. Valid options: {valid}")

    model_fn, weights = registry[model_name]
    model = model_fn(weights=weights)

    return model.to(device).eval(), weights.transforms()


def normalize_image(image):
    preprocess = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

    return preprocess(image)


def denormalize_image(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    import torchvision.transforms as T

    inv_normalize = T.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    return inv_normalize(image)

def load_image(path, preprocess, device="cuda"):

    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    return x

def append_results(path, results):
    # Load existing data if file exists
    if os.path.exists(path):
        with open(path, "r") as f:
            existing_results = json.load(f)
    else:
        existing_results = {
            "Insertion AUC": [],
            "Deletion AUC": [],
            "AIC AUC": [],
            "SIC AUC": [],
            "Infidelity": []
        }

    for key, val in results.items():
        # Ensure key exists in existing_results
        if key not in existing_results:
            existing_results[key] = []

        # If val is a list/tuple/etc. -> extend
        if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
            existing_results[key].extend(val)
        else:
            # Single scalar -> append
            existing_results[key].append(val)

    # Write updated results
    with open(path, "w") as f:
        json.dump(existing_results, f, indent=4)




def preprocess_attributions(attr):
    #attr = attr.cpu().detach().numpy()
    lower_bound = np.percentile(attr, 80)
    upper_bound = np.percentile(attr, 99)

    attr[attr < lower_bound] = lower_bound
    attr[attr > upper_bound] = upper_bound

    attr = (attr - attr.max())/ (attr.max() - attr.min())

    return attr



def save_attribution(cfg, image_path, attributions, image):

    path = "results"
    path = os.path.join(path, cfg.method_name, cfg.model)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, image_path.split("/")[-1])

    attr = attributions.sum(1, keepdim=True).cpu().squeeze().numpy()
    if attr.ndim == 3 and attr.shape[2] == 1:
        attr = attr[..., 0]  # -> (H,W)

    # Use positive attributions (or np.abs(attr) if you prefer magnitude)
    #attr = np.maximum(attr, 0)
    attr = np.abs(attr)
    # Robust normalize to [0,1] (percentile avoids outliers). Use min-max if you prefer.
    hi = np.percentile(attr, 99.0)
    attr_norm = np.clip(attr / (hi + 1e-12), 0, 1)  # (H,W)

    # --- prep image: (H,W,3) in [0,1] ---
    image = image.cpu()
    image = denormalize_image(image).squeeze().numpy().transpose(1, 2, 0)  # (H,W,3)
    image = np.clip(image, 0, 1)

    # --- highlight only salient regions ---
    mask3 = attr_norm[..., None]  # (H,W,1) -> broadcast to 3 channels
    highlight = image * mask3  # (H,W,3)

    plt.imshow(highlight)
    plt.axis('off')
    plt.savefig(path)

import numpy as np
import torch
from typing import List, Union


def denormalize(
    x: Union[torch.Tensor, np.ndarray],
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """
    Reverse ImageNet normalization.

    Accepts:
      [3, H, W]
      [1, 3, H, W]
      [H, W, 3]

    Returns:
      [H, W, 3] numpy array in [0, 1]
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape {x.shape}")
        x = x[0]

    if x.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got shape {x.shape}")

    # CHW -> HWC
    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    elif x.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, got shape {x.shape}")

    mean = np.asarray(mean, dtype=x.dtype)
    std = np.asarray(std, dtype=x.dtype)

    x = x * std + mean
    return np.clip(x, 0, 1)


def visualize_attributions(images: torch.Tensor,
                           attributions: torch.Tensor,
                           cmap: str = 'jet',  # 'jet' or 'turbo' often provide higher contrast than 'inferno'
                           alpha: float = 0.5,
                           percentile: float = 99,
                           save_path: Optional[str] = None):
    """
    Enhanced visualization using percentile clipping and Gaussian smoothing.
    """
    B = images.shape[0]

    # 1. Aggregation: Summing across channels can cancel out signal if not careful.
    # Using max(abs) is often more robust for 'visibility' than sum(abs).
    attr_map = attributions.detach().abs().max(dim=1)[0]

    # 2. Setup Plot
    fig, axes = plt.subplots(nrows=B, ncols=3, figsize=(15, 5 * B))
    if B == 1: axes = axes[None, :]

    for i in range(B):
        # A. Process Heatmap for Visibility
        attr_np = attr_map[i].cpu().numpy()

        # Robust Scaling: Use percentiles to ignore outlier spikes (hot pixels)
        v_max = np.percentile(attr_np, percentile)
        v_min = attr_np.min()
        attr_np = np.clip((attr_np - v_min) / (v_max - v_min + 1e-8), 0, 1)



        # B. Original Image
        img_viz = denormalize(images[i])
        axes[i, 0].imshow(img_viz)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis('off')

        # C. High-Contrast Heatmap
        # We use a thresholded mask to see the "core" of the attribution
        axes[i, 1].imshow(attr_np, cmap=cmap)
        axes[i, 1].set_title(f"Processed Heatmap ({percentile}th pct)")
        axes[i, 1].axis('off')

        # D. Blended Overlay
        axes[i, 2].imshow(img_viz)
        # Applying a threshold mask to the alpha channel makes the attribution "pop"
        # only where it is actually relevant
        mask = (attr_np > 0.2).astype(float) * alpha
        axes[i, 2].imshow(attr_np, cmap=cmap, alpha=mask)
        axes[i, 2].set_title("Focused Overlay")
        axes[i, 2].axis('off')

    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.show()