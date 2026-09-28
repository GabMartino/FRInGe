#!/bin/bash
#SBATCH --account=datamining
#SBATCH --job-name=eq_budget
#SBATCH --partition=p_datamining
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=galadriel
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=job_%A_%a_equal_budget.out
#SBATCH --error=job_%A_%a_equal_budget.err
# Three models x three methods, with at most six GPUs in use.
#
# `fringe` is the full multiclass Fisher geometry; `fringe_binary` is the
# target-vs-rest geometry. Each task runs the complete candidate grid for one
# (model, method) pair. Equal-runtime points are selected only after measurement.
#SBATCH --array=0-8%6

set -euo pipefail

module purge
module load python

# This script is intentionally submitted from the repository root. Using
# SLURM_SUBMIT_DIR makes relative paths stable even if Slurm starts the job
# from another directory.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$PROJECT_ROOT"

source "../venvs/fisher_rao_venv/bin/activate"

VENV_PYTHON="$VIRTUAL_ENV/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  echo "Python interpreter not found: $VENV_PYTHON" >&2
  exit 1
fi
if [ ! -f "configs/EqualBudget_config.yaml" ]; then
  echo "Submit this file from the FisherRaoIntegratedGradients repository root." >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

MODELS=(resnet18 resnet50 inception_v3)
METHODS=(ig fringe fringe_binary)
METHOD_COUNT=${#METHODS[@]}
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / METHOD_COUNT))
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID % METHOD_COUNT))
MODEL=${MODELS[$MODEL_INDEX]}
METHOD=${METHODS[$METHOD_INDEX]}

# Every array submission gets a separate run root, and every task writes into
# an isolated subtree below it. Set EQUAL_BUDGET_RUN_ID explicitly only when
# resuming a previous run.
RUN_ID="${EQUAL_BUDGET_RUN_ID:-${SLURM_ARRAY_JOB_ID}}"
COMPARISON_ROOT="./results/comparisons/EqualBudgetComparison/${RUN_ID}"
TASK_RESULTS_ROOT="${COMPARISON_ROOT}/jobs/${MODEL}_${METHOD}"
HYDRA_RUN_DIR="./artifacts/hydra/cluster/equal_budget/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TASK_RESULTS_ROOT" "$HYDRA_RUN_DIR"

echo "=========================================================="
echo "Job:     ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:    $(hostname)"
echo "Model:   $MODEL"
echo "Method:  $METHOD"
echo "Run ID:  $RUN_ID"
echo "Results: $TASK_RESULTS_ROOT"
echo "Python:  $VENV_PYTHON"
echo "Submit:  $PROJECT_ROOT"
echo "=========================================================="
"$VENV_PYTHON" --version
nvidia-smi -L || true

# Validate the step-local GPU binding before starting the expensive benchmark.
# For a valid runtime comparison, constrain the array to homogeneous GPU nodes
# at submission time if p_datamining contains more than one GPU model.
srun --ntasks=1 --ntasks-per-node=1 \
  env CUDA_VISIBLE_DEVICES=0 "$VENV_PYTHON" -c \
  'import os, sys, torch; assert torch.cuda.is_available(), "Allocated GPU is unavailable to PyTorch"; print("Runtime Python:", sys.executable); print("Step CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES")); print("CUDA device:", torch.cuda.get_device_name(0))'

srun --ntasks=1 --ntasks-per-node=1 \
  env CUDA_VISIBLE_DEVICES=0 "$VENV_PYTHON" \
  -m benchmarks.equal_budget.EqualBudget_benchmarking \
    hydra.run.dir="$HYDRA_RUN_DIR" \
    hydra.output_subdir=null \
    "avail_models=[$MODEL]" \
    "only_methods=[$METHOD]" \
    results_root="$TASK_RESULTS_ROOT"

# Submit EqualBudget_analyse.slurm with an afterok dependency on this array.
# That job validates the shared split and GPU model before selecting measured
# 1x, 2x, 5x, and 10x IG-32 runtime matches.
