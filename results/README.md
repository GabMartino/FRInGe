# Results

`summaries/<model>/summary_table.csv` contains the released aggregate metrics
for the eight-method main benchmark on each of the six ImageNet architectures.
Each row reports the sample count, mean, 95% percentile bootstrap limits, and
formatted interval for MAS insertion, MAS deletion, insertion AUC, and deletion
AUC. The appendix-only BlurIG, Expected Gradients, and MIG implementations are
not rows in these CSVs.

The tables were generated from 1,001 examples per method and architecture with
2,000 bootstrap resamples and seed 42. The implementation is in
`analyses/scripts/analyse_results.py`.

Raw per-image parquet files and attribution tensors are not committed because
of their size. Consequently these tables support inspection of the reported
aggregates but are not, by themselves, a complete independent reproduction
bundle.
