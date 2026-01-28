# FRInGe — Fisher–Rao Integrated Gradients (with Euclidean Trust Region)

This repository contains an experimental implementation of **Fisher–Rao Integrated Gradients** (“FRInGe”): an Integrated Gradients–style attribution method that follows a **geodesic in predictive distribution space** and maps it back to input space via a **(damped) Fisher pullback metric**, solved with **PCG** and stabilized with a **trust-region step rule**.

The code logs rich diagnostics (Euclidean vs Riemannian step sizes, entropy, solver residuals, damping statistics, etc.) and provides plotting utilities to analyze the geometry and stability of the path.

---

## What you get

- **Attributions** via Fisher–Rao geometry in prediction space.
- **Geodesic scheduling** between current prediction and a reference distribution (default: uniform).
- **Damped pullback metric** and **biharmonic regularization** for stability.
- **Preconditioned Conjugate Gradient (PCG)** solver for the inner linear system.
- **Trust-region controls**
  - KL-based step bound (Fisher norm)
  - Euclidean step bound (pixel/feature displacement)

- **Diagnostics & plots**
  - Riemannian vs Euclidean per-step distance
  - entropy evolution
  - solver residual, CG iterations
  - damping/fisher energy split
  - intermediate images and attribution deltas

---

## Repository structure (typical)

> Names may differ depending on your local layout; adjust paths accordingly.

- `FisherRaoIG/`
  - `BiharmonicFisherRaoIntegratedGradientsNoLogs.py` — solver core / attribution loop
  - `plot_utils.py` — plotting style + diagnostics
  - `plot_intermediates.py` — intermediate visualization helpers
- `metrics/`
  - `MetricsWrapper.py` — insertion/deletion AUC, MAS, infidelity, etc.
- `utils.py`
  - `load_model`, `load_image`, `normalize_image`, `denormalize_image`, etc.
- `main.py` (or your script)
  - runnable entry point / experiments

---

## Installation

### 1) Environment
Recommended: Python 3.10+.

Install requirements (example):
```bash
pip install torch torchvision numpy matplotlib tqdm
