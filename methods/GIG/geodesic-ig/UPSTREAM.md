# Vendored GeoIG source

The `geodesic/` package in this directory is vendored from:

- repository: <https://github.com/sina-salek/geodesic-ig>
- commit: `b5614a20201f532d08a4d7304064598ea80b0f25`
- license: MIT; retained in [`LICENSE`](LICENSE)
- paper: *Using the Path of Least Resistance to Explain Deep Networks*

The paper benchmark imports `GeodesicIGSVI` through
`methods/GIG/GIG_benchmarking.py`. The adapter is project code; the vendored
`geodesic/` package is third-party source. Generated caches, upstream experiments,
and upstream image assets are not vendored because they are not imported by the
benchmark.
