# Local datasets

ImageNet is not distributed with this repository. Place or symlink the
evaluation image directory at `data/ImageNet/`, or override `examples_dir` /
`dataset_dir` in the relevant Hydra configuration.

For Expected Gradients, the reference pool used in a final experiment must be
versioned and disjoint from evaluation images. The equal-budget protocol creates
and records disjoint baseline, warm-up, and evaluation partitions.
