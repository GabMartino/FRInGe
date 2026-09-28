# Benchmark protocols

`equal_budget/` evaluates IG, Expected Gradients, SmoothGrad, full categorical
FRInGe, and target-vs-rest FRInGe at explicit computational candidates. It
records steady-state latency, GPU identity, memory, operation counts, and paired
quality metrics, then selects measured runtime matches to an IG anchor.

This protocol is the appropriate basis for speed or quality-at-fixed-cost
claims. Default method configurations generally have different gradient and
optimization budgets and should not be interpreted as a controlled runtime
comparison.
