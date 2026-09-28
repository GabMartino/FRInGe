"""
Manifold Integrated Gradients (MIG) for the FRInGe benchmark suite.

Ports the geodesic Integrated Gradients method of
Zaher et al., "Manifold Integrated Gradients: Riemannian Geometry for Feature
Attribution" (ICML 2024) so it plugs into this repo's ``MetricsWrapper`` and
produces the *same* metric columns (Insertion/Deletion AUC, MAS, Infidelity,
Sparseness) as every other method here.

Why a pretrained autoencoder instead of training a VAE
------------------------------------------------------
The original method learns a dataset-specific VAE and computes geodesics in its
latent space. Training a faithful VAE on ImageNet is expensive and yields blurry
reconstructions, which would unfairly sink the method (MIG attributes on the
*decoded* image, so recon quality is its ceiling). Instead we use a *pretrained*
image autoencoder -- Stable Diffusion's ``AutoencoderKL`` (KL-f8) -- as the data
manifold. It reconstructs ImageNet-scale images well, is fully differentiable
(required by the decoder-Jacobian geodesic solver), and has a continuous spatial
latent (required for gradient flow along the path).

Normalization boundary
-----------------------
Classifiers here consume ImageNet-normalized tensors; ``AutoencoderKL`` consumes
[-1, 1] pixel tensors. We denormalize -> [0,1] -> [-1,1] to encode, and
decode -> [0,1] -> renormalize before the classifier. All conversions are affine
and differentiable.

Faithfulness note
-----------------
As in the original method, the integration endpoints are the *reconstructions*
``decode(encode(baseline))`` and ``decode(encode(x))``, so the attribution lives
in reconstruction space. With a high-quality AE recon ~= input, so the map aligns
spatially with the original image that the metrics perturb. This is inherent to
MIG, not an artifact of this port.
"""
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

from common.utils import normalize_image, denormalize_image


# ---------------------------------------------------------------------------
# Manifold (pretrained autoencoder)
# ---------------------------------------------------------------------------
def load_manifold_ae(model_id: str = "stabilityai/sd-vae-ft-mse",
                     device: Optional[torch.device] = None):
    """Load a frozen, eval-mode diffusers ``AutoencoderKL`` to act as the manifold.

    First call downloads weights (~335 MB) from the HF hub; on an offline HPC
    node pre-fetch on a login node or set ``HF_HOME`` to a cached location.
    """
    try:
        from diffusers import AutoencoderKL
    except ImportError as e:
        raise ImportError(
            "MIG needs the 'diffusers' package for the pretrained autoencoder "
            "manifold. Install it with: pip install diffusers"
        ) from e

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(model_id).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


class AEManifold:
    """Expose a diffusers ``AutoencoderKL`` as encode/decode over [0, 1] images.

    Pure autoencoding: we use the raw latent mean (no diffusion ``scaling_factor``,
    which is only a convention for the LDM UNet).
    """

    def __init__(self, vae):
        self.vae = vae

    def encode(self, x01: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x01 * 2.0 - 1.0).latent_dist.mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return (self.vae.decode(z).sample + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Geodesic path solver (ported from core/geodesic.py, decoder-energy variant)
# ---------------------------------------------------------------------------
def _compute_etta_d(decode: Callable, zi, zi_minus, zi_plus, dt, img_shape):
    """Discrete geodesic update direction at an interior point (curve energy grad).

    The image-space acceleration (second finite difference of decoded frames) is a
    fixed data vector; MIG pulls it back to latent space via the decoder Jacobian,
    ``etta = -J^T . accel``. Computing the acceleration under ``no_grad`` matches
    that formulation and avoids holding decoder graphs across the iteration loop.
    """
    with torch.no_grad():
        g_minus = decode(zi_minus).reshape(-1)
        g_i = decode(zi).reshape(-1)
        g_plus = decode(zi_plus).reshape(-1)
        finite_diff = ((g_plus - 2.0 * g_i + g_minus) / dt).reshape(img_shape)
    _, jv = torch.autograd.functional.vjp(decode, zi, finite_diff)
    return -jv.reshape_as(zi)


def _path_energy(decode: Callable, z_collection: List[torch.Tensor], dt, img_shape) -> float:
    total = 0.0
    for j in range(1, len(z_collection) - 1):
        e = _compute_etta_d(decode, z_collection[j], z_collection[j - 1],
                            z_collection[j + 1], dt, img_shape)
        total += e.norm().pow(2).item()
    return total


def geodesic_path(decode: Callable, z_collection: List[torch.Tensor], alpha: float,
                  epsilon: float, max_iterations: int, img_shape) -> List[torch.Tensor]:
    """Gradient-descent the interior latents to minimize decoder curve energy.

    Endpoints (indices 0 and -1) are held fixed. With ``max_iterations == 0`` this
    returns the straight-latent path, whose *decoded* trajectory is already curved
    in pixel space (the core MIG effect) -- so the method degrades gracefully.
    """
    T = len(z_collection)
    dt = 1.0 / T
    for _ in range(max_iterations):
        if _path_energy(decode, z_collection, dt, img_shape) <= epsilon:
            break
        for i in range(1, T - 1):
            e = _compute_etta_d(decode, z_collection[i], z_collection[i - 1],
                                z_collection[i + 1], dt, img_shape)
            z_collection[i] = z_collection[i] - alpha * e
    return z_collection


def _interpolate(z0: torch.Tensor, zf: torch.Tensor, n: int) -> List[torch.Tensor]:
    """Linear latent interpolation with exact endpoints (n points, incl. both ends)."""
    return [z0 + (float(i) / (n - 1)) * (zf - z0) for i in range(n)]


# ---------------------------------------------------------------------------
# Integrated Gradients along the (decoded) geodesic path
# ---------------------------------------------------------------------------
def _integrated_gradients_geo(clf: torch.nn.Module,
                              path_imgs_norm: List[torch.Tensor],
                              target: int) -> torch.Tensor:
    """IG along a path: (x_T - x_0) * mean_t grad_x softmax(clf(x_t))[target].

    ``path_imgs_norm`` are classifier-normalized (1,3,H,W) images along the path.
    """
    scaled = torch.cat(path_imgs_norm, dim=0).detach().requires_grad_(True)
    probs = torch.softmax(clf(scaled), dim=1)
    scores = probs[:, target]
    grads = torch.autograd.grad(scores, scaled,
                                grad_outputs=torch.ones_like(scores))[0]
    avg_grad = grads.mean(dim=0, keepdim=True)
    attr = (scaled[-1:].detach() - scaled[:1].detach()) * avg_grad
    return attr


# ---------------------------------------------------------------------------
# Wrapper matching the MetricsWrapper .attribute(inputs, target, seed) contract
# ---------------------------------------------------------------------------
class ManifoldIGWrapper:
    """Manifold IG attribution method.

    Processes one image at a time (each requires its own geodesic optimization),
    mirroring the GIG wrapper, and restacks to the original batch shape.
    """

    def __init__(self, model: torch.nn.Module, ae, num_interpolants: int = 12,
                 geo_alpha: float = 1e-4, geo_epsilon: float = 1e3,
                 geo_max_iterations: int = 3):
        self.clf = model.eval()
        self.device = next(model.parameters()).device
        self.ae = AEManifold(ae)
        self.n = int(num_interpolants)
        self.geo_alpha = float(geo_alpha)
        self.geo_epsilon = float(geo_epsilon)
        self.geo_max_iterations = int(geo_max_iterations)

    def _attribute_single(self, x_norm: torch.Tensor, target: int) -> torch.Tensor:
        # x_norm: (1,3,H,W) classifier-normalized.
        H, W = x_norm.shape[-2:]
        aH, aW = (H // 8) * 8, (W // 8) * 8  # AutoencoderKL (f8) needs multiples of 8

        x01 = denormalize_image(x_norm).clamp(0.0, 1.0)
        x01_ae = x01 if (aH, aW) == (H, W) else F.interpolate(
            x01, size=(aH, aW), mode="bilinear", align_corners=False)
        base01_ae = torch.zeros_like(x01_ae)  # black baseline (IG convention)

        with torch.no_grad():
            z0 = self.ae.encode(base01_ae)
            zf = self.ae.encode(x01_ae)

        z_collection = _interpolate(z0, zf, self.n)
        img_shape = (1, x01_ae.shape[1], aH, aW)
        z_collection = geodesic_path(self.ae.decode, z_collection, self.geo_alpha,
                                     self.geo_epsilon, self.geo_max_iterations, img_shape)

        with torch.no_grad():
            path01 = [self.ae.decode(z).clamp(0.0, 1.0) for z in z_collection]
            if (aH, aW) != (H, W):
                path01 = [F.interpolate(p, size=(H, W), mode="bilinear",
                                        align_corners=False) for p in path01]
            path_norm = [normalize_image(p) for p in path01]

        return _integrated_gradients_geo(self.clf, path_norm, target)

    def attribute(self, inputs: torch.Tensor, target_idx: torch.Tensor,
                  seed: Optional[int] = None):
        inputs = inputs.to(self.device)
        target_idx = target_idx.to(self.device)
        if seed is not None:
            torch.manual_seed(seed)

        attributions = []
        for i in range(inputs.shape[0]):
            attr = self._attribute_single(inputs[i:i + 1], int(target_idx[i].item()))
            attributions.append(attr.detach())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(attributions, dim=0), {}
