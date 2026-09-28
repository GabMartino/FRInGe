# Equal-budget attribution benchmark

This package implements the paper's controlled runtime comparison. The exact
timing, CUDA-memory, aggregation, and plotting rules are encoded in the runner,
analysis module, saved manifests, and scientific contract below.

It compares:

- Integrated Gradients;
- Expected Gradients with a disjoint empirical reference pool;
- SmoothGrad;
- IG with Gaussian smoothing of the final attribution map;
- fast, default, and high-fidelity multiclass FRInGe configurations;
- matched target-vs-rest binary FRInGe configurations.

## Scientific contract

- A seeded ImageNet partition is created once. Reference, warm-up, and
  evaluation images are mutually disjoint.
- Every method/model/candidate uses the same evaluation paths.
- The explained target is the original model top-1 class and the explained
  scalar is its logit.
- Steady-state attribution latency is measured one image at a time.
- CUDA is synchronized before and after attribution.
- Metric time is excluded from attribution time.
- Initialization and optional batched throughput are reported separately.
- Peak allocated/reserved CUDA memory, incremental CUDA memory, and CPU
  maximum RSS are recorded per image.
- Algorithmic budgets are explicit: gradient evaluations, references/noise
  samples, FRInGe waypoints, and FRInGe CG iterations.
- Quality is paired by `image_id`.

The `ig_smoothed_map` control only applies the matched Gaussian kernel to the
final IG attribution. It is deliberately not described as a per-step
regularized path; that stronger control remains a separate experiment.

## Full run

From the repository root:

```bash
python -m benchmarks.equal_budget.EqualBudget_benchmarking
python -m benchmarks.equal_budget.analyse_equal_budget \
  --results-root results/comparisons/EqualBudget
```

The default publication grid uses ResNet-18, ResNet-50, and Inception-v3; ten
warm-up images; 64 held-out evaluation images; 128 disjoint Expected Gradients
references; and 31 candidate configurations. It is intentionally expensive.
Existing completed candidates are reused unless `overwrite=true`.

To run on a scheduler, keep the result root shared and partition models or
methods with Hydra overrides, for example:

```bash
python -m benchmarks.equal_budget.EqualBudget_benchmarking \
  'avail_models=[resnet18]' \
  'only_methods=[ig,expected_gradients]'
```

For the focused IG, full multiclass FRInGe, and target-vs-rest binary FRInGe
comparison, submit the provided 9-task array from the repository root:

```bash
ARRAY_JOB_ID=$(sbatch --parsable cluster/EqualBudget_benchmarking.slurm)
ARRAY_JOB_ID=${ARRAY_JOB_ID%%;*}
sbatch --dependency="afterok:${ARRAY_JOB_ID}" \
  --export="ALL,EQUAL_BUDGET_RUN_ID=${ARRAY_JOB_ID}" \
  cluster/EqualBudget_analyse.slurm
```

It runs one `(model, method)` pair per GPU and writes each task into an isolated
subdirectory under
`results/comparisons/EqualBudgetComparison/<array-job-id>/jobs`. The dependent
CPU job checks that every candidate used the same image split and GPU model, then
selects the candidates nearest measured `1x`, `2x`, `5x`, and `10x` IG-32
latency. If the partition contains heterogeneous GPUs, add the cluster's GPU
constraint or a homogeneous `--nodelist` to the array submission.

To run the analysis manually after the array finishes:

```bash
python -m analyses.scripts.analyse_equal_budget_comparison --run-id <array-job-id>
```

Omit `--run-id` to analyse the latest run. While an array is running, inspect
candidate coverage and generate plots from all currently completed candidates:

```bash
python -m analyses.scripts.analyse_equal_budget_comparison --status-only
```

Without `--status-only`, incomplete runs are analysed using the completed
candidates for models whose IG-32 anchor exists. Such outputs are explicitly
marked as partial in `analysis_manifest.json`. Use `--require-complete` for
publication-final analysis.

Restrict every table and plot to selected models with:

```bash
python -m analyses.scripts.analyse_equal_budget_comparison \
  --models resnet18 resnet50
```

Partial analysis always writes coverage, runtime, memory-versus-time, and
quality-versus-time figures in both PNG and PDF form. Methods are distinguished
by colour and GPU models by marker shape, which makes mixed-hardware exploratory
runs visible rather than silently pooling their timings.

Plot and table labels expose the actual controls: for example,
`FRInGe — full Fisher / KL=0.005 · CG≤10` and
`FRInGe — binary target-vs-rest / KL=0.0001 · inverse≤40`. Legacy candidate
IDs such as `fringe_fast_3` remain only as stable machine-facing directory keys
for already-running and saved experiments.

## Smoke run

```bash
python -m benchmarks.equal_budget.EqualBudget_benchmarking \
  'avail_models=[resnet18]' \
  'only_methods=[ig]' \
  'methods.ig.candidates=[{id:ig_8,n_steps:8}]' \
  split.baseline_pool_size=16 \
  split.warmup_size=1 \
  split.evaluation_size=1 \
  systems.warmup_runs=1 \
  systems.throughput_runs=0 \
  quality.causal_steps=4 \
  quality.mas_step_size=12544 \
  quality.infidelity_perturbations=2 \
  results_root=./results/comparisons/EqualBudgetSmoke
```

## Outputs

Top-level:

```text
results/comparisons/EqualBudget/
├── run_manifest.json
├── resolved_config.yaml
├── split_manifest.json
├── candidate_summary.{csv,parquet}
├── equal_budget_selection.csv
├── equal_memory_selection.csv
├── equal_autograd_selection.csv
├── equal_budget_table.{csv,md}
├── equal_budget_paired_bootstrap.csv
├── full_method_points.csv
├── fringe_gain_retention.csv
├── EQUAL_BUDGET_REPORT.md
├── pareto_*.{png,pdf}
└── memory_vs_mas_insertion_blur.{png,pdf}
```

Each model directory also contains `model_manifest.json`, including reference
pool preprocessing time. The run manifest contains hashes of the benchmark,
FRInGe diagnostic implementation, and default configuration, which provides
source provenance even when the workspace has no usable Git metadata.

Per candidate:

```text
<model>/<method>/<candidate>/
├── candidate_config.json
├── systems.json
├── per_image.{csv,parquet}
├── summary.json
└── attributions/                 # only when save_attributions=true
```

Runtime candidates are selected by minimum absolute log-distance from measured
`1×`, `2×`, `5×`, and `10×` IG-32 median latency. Secondary selection files use
peak allocated CUDA memory and autograd-equivalent evaluations. For FRInGe,
the latter counts direct reverse gradients plus the JVP and VJP in every
recorded Fisher matvec; PCG final-residual, warm-start, and telemetry matvecs
are included. Paired
differences and 95% bootstrap confidence intervals use identical evaluation
images and compare against the selected FRInGe candidate at each budget.

Semantic localization is not emitted because the repository currently has no
localization annotation set. All other roadmap quality fields are present.
