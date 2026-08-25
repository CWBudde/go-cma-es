# Block-diagonal covariance cost

Phase 8 targets the MayFlyCircleFit joint stage: 14,000 parameters arranged as 2,000
nearly independent circles with seven parameters each. A full 14,000×14,000 covariance
is inappropriate for that structure; 2,000 independent 7×7 covariances retain the
within-circle correlations.

## Complexity

For problem dimension `n`, maximum block width `k`, and `n/k` equally sized blocks:

| Operation                           |           Work or storage |                 n = 14,000, k = 7 |
| ----------------------------------- | ------------------------: | --------------------------------: |
| Covariance and eigensystem matrices |                  `O(n*k)` |  98,000 entries per matrix family |
| Transform or whiten one sample      |                  `O(n*k)` |         98,000 multiply-add terms |
| Rank-one covariance contribution    |                  `O(n*k)` | 98,000 upper/lower-triangle terms |
| Decompose every block               | `O((n/k)*k^3) = O(n*k^2)` |         686,000 cubic-scale terms |

The original plan called sampling `O(n*k^2)`. That counted `n` blocks of width `k`; there
are only `n/k` blocks, so `(n/k)*k^2 = n*k`. The decomposition retains the `O(n*k^2)`
bound because each block costs `O(k^3)`.

The persistent covariance and current eigensystem contain 196,000 floating-point matrix
entries together, about 1.50 MiB of numeric payload. Go slice metadata and the lazily
prepared next eigensystem add overhead while preserving `O(n*k)` growth. By comparison,
one dense 14,000×14,000 matrix alone contains 196 million entries, about 1.46 GiB.

## Measurement

Measured on 2026-08-25 with Go 1.26 on an AMD Ryzen 5 4600H (12 logical CPUs):

```text
go test -run '^$' -bench '^BenchmarkBlockDiagonalN14000K7$' -benchtime=10x -benchmem -count=3

BenchmarkBlockDiagonalN14000K7-12  10  96704969 ns/op  28435234 B/op  32252 allocs/op
BenchmarkBlockDiagonalN14000K7-12  10  89087012 ns/op  28434700 B/op  32251 allocs/op
BenchmarkBlockDiagonalN14000K7-12  10  91842803 ns/op  28434718 B/op  32251 allocs/op
```

The median is 91.84 ms for one complete generation with the default 32 candidates:
sampling, objective evaluation, active covariance adaptation, and decomposition of all
2,000 blocks. The allocation figure covers the entire generation, including candidate
position, step, and normal vectors; it is not the persistent covariance footprint.

The benchmark is intentionally reproducible rather than a one-off timing harness. Run it
again with:

```sh
go test -run '^$' -bench '^BenchmarkBlockDiagonalN14000K7$' -benchmem
```
