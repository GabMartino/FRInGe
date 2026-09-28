# Local datasets

ImageNet is not distributed with this repository. Place or symlink the
evaluation image directory at `data/ImageNet/`, or override `examples_dir` /
`dataset_dir` in the relevant Hydra configuration.

For Expected Gradients, the reference pool used in a final experiment must be
versioned and disjoint from evaluation images. Set `baselines_dir` to that pool
and enable `require_disjoint_baselines` in
`configs/ExpectedGradients_config.yaml`.
