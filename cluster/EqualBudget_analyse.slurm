#!/bin/bash
#SBATCH --account=datamining
#SBATCH --job-name=eq_budget_analysis
#SBATCH --partition=p_datamining
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=job_%j_equal_budget_analysis.out
#SBATCH --error=job_%j_equal_budget_analysis.err

set -euo pipefail

module purge
module load python

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$PROJECT_ROOT"
source "../venvs/fisher_rao_venv/bin/activate"

VENV_PYTHON="$VIRTUAL_ENV/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  echo "Python interpreter not found: $VENV_PYTHON" >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

RUN_ID="${EQUAL_BUDGET_RUN_ID:?Submit with EQUAL_BUDGET_RUN_ID set to the array job ID}"
COMPARISON_ROOT="./results/comparisons/EqualBudgetComparison"
RUN_ROOT="${COMPARISON_ROOT}/${RUN_ID}"
if [ ! -d "$RUN_ROOT/jobs" ]; then
  echo "Equal-budget result directory not found: $RUN_ROOT/jobs" >&2
  exit 1
fi

"$VENV_PYTHON" -m analyses.scripts.analyse_equal_budget_comparison \
  --results-root "$COMPARISON_ROOT" \
  --run-id "$RUN_ID" \
  --anchor-candidate ig_32 \
  --budget-multiples 1 2 5 10 \
  --reference-method fringe \
  --seed 42 \
  --bootstrap-resamples 10000
