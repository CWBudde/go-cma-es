# Full versus separable covariance at n = 200

Phase 6 adds `BenchmarkCovarianceModesN200`, which runs one generation of the same
separable sphere objective through each covariance representation. The objective is
deliberately cheap so the result measures optimizer overhead rather than hiding it.

Measured on 2026-08-25 with Go 1.26 on an AMD Ryzen 5 4600H (`-count=5`):

| Mode      | Time per generation | Bytes per run | Allocations |
| --------- | ------------------: | ------------: | ----------: |
| full      |       7.58–11.24 ms |       920,468 |         160 |
| separable |      0.246–0.279 ms |       203,649 |         128 |

This is a 27–46× wall-time improvement in this benchmark. More importantly, the
separable strategy state has no dense covariance or eigenvector matrix: its persistent
covariance storage is two length-n slices, and it never calls the eigendecomposition.
The asymptotic difference therefore remains `O(n)` versus `O(n²)` even when allocator
and harness costs vary between machines.

Reproduce with:

```sh
go test -run '^$' -bench '^BenchmarkCovarianceModesN200$' -benchmem -count=5 ./...
```
