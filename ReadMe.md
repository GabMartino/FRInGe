# FRInGe — Fisher–Rao Integrated Gradients (with Euclidean Trust Region)

This repository contains an experimental implementation of **Fisher–Rao Integrated Gradients** (“FRInGe”): an Integrated Gradients–style attribution method that follows a **geodesic in predictive distribution space** and maps it back to input space via a **(damped) Fisher pullback metric**, solved with **PCG** and stabilized with a **trust-region step rule**.

The code logs rich diagnostics (Euclidean vs Riemannian step sizes, entropy, solver residuals, damping statistics, etc.) and provides plotting utilities to analyze the geometry and stability of the path.

## Installation

### 1) Environment
Recommended: Python 3.10+.

Install requirements (example):
```bash
pip install torch torchvision numpy matplotlib tqdm
