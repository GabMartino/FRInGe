# FRInGe: Distribution-Space Integrated Gradients with Fisher–Rao Geometry

**FRInGe** is a gradient-based attribution method that defines both its reference and its interpolation
schedule in the model's predictive distribution space. It replaces a hand-designed input baseline with a
maximum-entropy predictive reference, follows a Fisher–Rao geodesic on the probability simplex, and
realizes that path in input space through a regularized pullback Fisher metric.

<p align="center">
  <img src="assets/predictive_geometry.png" width="100%" alt="FRInGe predictive-space geodesic and its input-space realization">
</p>

<p align="center"><em>
FRInGe prescribes a geodesic from the model prediction to a maximum-entropy reference and tracks its
waypoints through the classifier's pullback geometry.
</em></p>

> **Research-code status.** This repository currently provides the full categorical FRInGe reference
> implementation, diagnostics, evaluation metrics, and an executable notebook. The API and default
> parameters may change while the research release is being finalized.

## Why FRInGe?

Standard Integrated Gradients requires an input-space baseline and usually follows a Euclidean straight
line. Both choices can be problematic: the baseline may not represent missing information, while the path
may cross saturated or poorly conditioned regions. FRInGe changes the construction in five steps:

1. **Predictive reference:** use the uniform categorical distribution as a maximum-entropy endpoint.
2. **Intrinsic schedule:** connect the prediction to that endpoint with a Fisher–Rao geodesic.
3. **Pullback realization:** map each predictive waypoint back to an input update through the model's
   pullback Fisher metric.
4. **Stable updates:** combine damping, spatial regularization, a KL/Fisher trust region, and a Euclidean
   step cap.
5. **Path attribution:** integrate the target-score gradient along the realized input trajectory.

The inner linear system is solved with preconditioned conjugate gradients (PCG), without forming the full
input-space Fisher matrix.

## What the trajectory looks like

<p align="center">
  <img src="assets/trajectory_overview.png" width="100%" alt="FRInGe entropy, intermediate inputs, and accumulated attributions along the trajectory">
</p>

The predictive entropy increases toward the maximum-entropy endpoint while evidence for the target class
is progressively attenuated. The bottom row shows when attribution is accumulated along this trajectory.

## Qualitative comparison

<p align="center">
  <img src="assets/qualitative_comparison.png" width="100%" alt="Qualitative comparison of FRInGe, Integrated Gradients, SmoothGrad, GGIG, and GeoIG">
</p>

The figure includes a strong case, a typical case, and a representative failure case rather than showing
only favorable examples. `MI` and `MD` denote image-level MAS-Insertion and MAS-Deletion, respectively.
Across the manuscript's six ImageNet architectures, FRInGe's clearest and most consistent advantage is on
calibration-oriented MAS metrics; perturbation AUC results are more mixed.

## Installation

Python 3.10 or newer is recommended. A CUDA-capable GPU is strongly recommended because FRInGe performs
repeated Jacobian-vector products and iterative linear solves.

```bash
git clone https://github.com/GabMartino/FRInGe.git
cd FRInGe

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Torchvision downloads pretrained ImageNet weights the first time a model is loaded.

## Quick start

The following example explains the top-1 prediction of a pretrained ResNet-18. These are demonstration
parameters, not universal defaults; stable settings depend on the architecture and input resolution.

```python
import torch

from FisherRaoIG.BiharmonicFisherRaoIntegratedGradients import (
    FisherRaoIntegratedGradients,
    LogConfig,
)
from utils import denormalize_image, load_image, load_model


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_model("resnet18", device=device)
x = load_image("examples/n01580077_jay.JPEG", preprocess, device=device)

with torch.no_grad():
    target = model(x).argmax(dim=1)


def target_score(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return model(inputs).gather(1, targets[:, None]).squeeze(1)


explainer = FisherRaoIntegratedGradients(
    model=model,
    model_forward=target_score,
    target_idx=target,
    use_exact_jvp=True,
)

attributions, diagnostics = explainer.attribute(
    x,
    max_correction_steps=1,
    cg_max_iter=20,
    kl_target=1e-3,
    eta_base=10.0,
    delta_euc=5.0,
    gamma=1e-2,
    alpha_start=1e-3,
    alpha_end=1e-3,
    diag_floor=1e-2,
    clamp_min_diag=1e-3,
    clamp_max_diag=100.0,
    log_cfg=LogConfig(enabled=True, store_viz=False),
    denormalize_image=denormalize_image,
)

print(attributions.shape)
print(sorted(diagnostics))
```

For an end-to-end example with visualizations and metrics, run
[`FRInGe_demo.ipynb`](FRInGe_demo.ipynb). Jupyter is optional and can be installed with:

```bash
python -m pip install jupyterlab
jupyter lab FRInGe_demo.ipynb
```

## Diagnostics

With logging enabled, `attribute` returns a dictionary containing quantities such as:

- predictive entropy and waypoint loss;
- Euclidean, Fisher–Rao, and regularized step lengths;
- relative PCG residuals and iteration counts;
- damping-versus-Fisher energy contributions;
- quadrature completeness residuals;
- optional intermediate inputs and attribution increments.

These diagnostics are intended to make solver failure, poor waypoint tracking, and trust-region clipping
observable rather than silent.

## Evaluation metrics

The `metrics/` package includes:

- blur-based insertion and deletion AUC;
- Magnitude Aligned Scoring (MAS) for insertion and deletion;
- infidelity;
- max sensitivity;
- Gini sparseness.

Metrics and perturbation choices answer different questions. In particular, MAS evaluates whether
attribution magnitude tracks the model's confidence response, whereas insertion/deletion AUC primarily
evaluates the induced feature ranking.

## Repository layout

```text
FisherRaoIG/
  BiharmonicFisherRaoIntegratedGradients.py  # FRInGe solver and logging
  plot_intermediates.py                      # intermediate-path visualizations
  plot_utils.py                              # geometry and attribution plots
metrics/                                     # attribution evaluation metrics
examples/                                    # example ImageNet images
FRInGe_demo.ipynb                            # end-to-end demonstration
utils.py                                     # model and image utilities
```

The repository also contains benchmarking implementations of IG, GuidedIG, SmoothGrad, IG², and
Adversarial IG used during development.

## Reproducibility notes

- `FisherRaoIntegratedGradients` resets its local random generator from the configured seed.
- The returned diagnostic dictionary can be serialized for post-hoc convergence analysis.
- Hyperparameters should be reported together with the model, preprocessing pipeline, target definition,
  and perturbation protocol.
- The current repository is a reference implementation and demo; the complete paper-specific benchmark
  orchestration and architecture-specific configurations are being prepared for release.

## Paper and citation

The accompanying manuscript is titled **"FRInGe: Distribution-Space Integrated Gradients with
Fisher–Rao Geometry."** A public paper link and BibTeX entry will be added when the preprint is released.

If you use the code before then, please link to this repository and record the exact Git commit for
reproducibility.

## Questions and issues

Please use the [GitHub issue tracker](https://github.com/GabMartino/FRInGe/issues) for reproducible bug
reports and questions about the implementation.
