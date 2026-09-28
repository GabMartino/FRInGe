# Analyses

`scripts/` contains the aggregation and figure-generation code used in the
paper workspace. The scripts expect raw per-method outputs under
`results/methods/`; those large files are not committed.

The compact tables already generated from those outputs are versioned under
`results/summaries/`. Run analysis modules from the repository root so package
imports and relative result paths resolve consistently.
