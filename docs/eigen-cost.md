# Eigendecomposition cost

Phase 2 uses a cyclic Jacobi eigendecomposition rather than a third-party
numerics package. This benchmark records the cost boundary for that choice and
makes the full-covariance versus sep-CMA-ES recommendation measurable.

## Method

The benchmark decomposes the dense symmetric Toeplitz covariance

```text
C[i,j] = 0.9^abs(i-j)
```

at the three dimensions required by `PLAN.md`. This is positive definite and
dense, and unlike an identity or diagonal matrix it exercises the Jacobi
rotations rather than measuring only setup and convergence detection. Matrix
construction is outside the timed region.

Run the practical sizes with calibrated samples and the capacity case once:

```sh
go test -run '^$' -bench '^BenchmarkEigenDecomposition$/n=(56|200)$' -benchtime=1s -count=3
go test -run '^$' -bench '^BenchmarkEigenDecomposition$/n=1000$' -benchtime=1x -count=1
```

The one-iteration n=1000 setting is deliberate because that case takes more
than a minute on the reference host. It is a capacity measurement, not a
statistically calibrated microbenchmark.

## Reference result

Measured 2026-08-25 on Linux/amd64 with Go 1.26.0:

- AMD Ryzen 5 4600H, 6 cores / 12 logical CPUs
- Linux 7.0.0-30-generic

| Dimension | Time per decomposition |   Bytes/op | Allocations/op | Sample policy                  |
| --------: | ---------------------: | ---------: | -------------: | :----------------------------- |
|        56 |               3.646 ms |     87,416 |             11 | median of 3 calibrated samples |
|       200 |               224.2 ms |  1,003,064 |             11 | median of 3 calibrated samples |
|      1000 |                77.82 s | 24,109,112 |             11 | one iteration                  |

The allocation count is constant because each matrix is backed by one
contiguous data slice; the byte growth is the expected O(n²). Runtime is O(n³),
with the n=1000 case additionally crossing cache boundaries.

## Interpretation

At the motivating n=56, decomposition is 3.65 ms. The same host's recorded
MayFlyCircleFit 512×512/K100 renderer measurements are 18.53 ms with one render
thread and 6.257 ms with twelve, so decomposition is about one fifth of one
serial render or three fifths of one parallel render. It is not literally
free, but CMA-ES performs λ objective evaluations per generation and refreshes
the eigenbasis lazily, so this cost is amortized over many renders.

At n=200 the decomposition is already about a quarter second. At n=1000,
77.82 seconds per refresh makes dense covariance operationally inappropriate;
`NewHighDimensionalConfig` and separable covariance are appropriate when axes
are independent. `NewBlockDiagonalConfig` provides the middle ground for
structured problems whose correlations stay inside small parameter groups; see
`blockdiag-cost.md` for the n=14,000 measurement.

Renderer comparison values come from the same-host table in
`../MayFlyCircleFit/docs/cpu-performance-history.md`; remeasure both workloads
before drawing conclusions on another machine.
