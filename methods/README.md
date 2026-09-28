# Method implementations

Each subdirectory is an independent attribution method. The listed entry point
is the implementation imported for the paper experiments. “Project benchmark
implementation” means the code was implemented or adapted in this project from
the cited method description; it does not mean it is an official author release.

| Package | Method | Benchmark entry point | Provenance and qualification |
|---|---|---|---|
| `FisherRaoIG/` | FRInGe / FRInGe-B | `FisherRaoIG_benchmarking.py` | Project implementation. The canonical categorical class dispatches to the binary backend when `binary=true`. |
| `IG/` | Integrated Gradients | `IG_benchmarking.py` | Captum `IntegratedGradients` wrapper. `IntegratedGradients.py` is diagnostic code, not the paper runner. |
| `SmoothGrad/` | SmoothGrad | `SmoothGrad_benchmarking.py` | Project benchmark implementation. |
| `GuidedIG/` | Guided Integrated Gradients | `GuidedIG_benchmarking.py` | Project benchmark implementation of the adaptive-path method. |
| `IG2/` | IG² | `IG2_benchmarking.py` | Project benchmark implementation using an iterative representation-guided path. |
| `AdversarialIG/` | Adversarial Gradient Integration | `AdversarialIG_benchmarking.py` | Project benchmark implementation of AGI. |
| `GGIG/` | Geometrically Guided IG | `GGIG_benchmarking.py` | Project benchmark implementation. |
| `GIG/` | GeoIG | `GIG_benchmarking.py` | Adapter around vendored upstream `geodesic-ig` source at commit `b5614a20201f532d08a4d7304064598ea80b0f25`; see `GIG/geodesic-ig/UPSTREAM.md`. |
| `BlurIG/` | Blur Integrated Gradients | `BlurIG_benchmarking.py` | Project benchmark implementation used in the appendix. |
| `ExpectedGradients/` | Expected Gradients | `ExpectedGradients_benchmarking.py` | Captum-based project implementation. A valid experiment requires an empirical reference pool disjoint from evaluation images. |
| `MIG/` | Manifold Integrated Gradients | `MIG_benchmarking.py` | Project port used in the appendix. It uses a pretrained Stable Diffusion `AutoencoderKL`, not the original method's dataset-specific VAE. |

All runners share `common.utils` for preprocessing and
`evaluation.metrics.MetricsWrapper` for evaluation. This common interface does
not make default computational budgets equivalent. Use
`benchmarks/equal_budget/` when making speed or quality-at-fixed-cost claims.

Run modules from the repository root, for example:

```bash
python -m methods.SmoothGrad.SmoothGrad_benchmarking
```
