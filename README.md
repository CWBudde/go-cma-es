# go-cma-es

A dependency-free Go implementation of the **Covariance Matrix Adaptation Evolution
Strategy** (CMA-ES), following Hansen's tutorial (arXiv:1604.00772).

Third in a family with [Mayfly](https://github.com/cwbudde/mayfly) and
[Dragonfly](https://github.com/CWBudde/dragonfly), and deliberately unlike both of them.

> **Status: under construction.** Phase 0 (scaffolding) is complete; the algorithm is not
> implemented yet. [`PLAN.md`](PLAN.md) is the source of truth for progress.

## Why

Swarm metaheuristics — particle swarm, mayfly, dragonfly, and the rest of the zoo —
share one structural weakness: an **isotropic, externally scheduled step size and no
learned metric**. On a problem whose parameters differ in curvature, a step correct along
one axis is wrong along every other, and the population collapses onto a point long
before its budget is spent.

That is not a hypothesis. The project this library was written for measured it directly:
population spread fell below 10% of its initial value by iteration 11–16, in the last
786,432 offspring not one beat both of its parents, and the run's best cost was frozen for
its final two thirds. The collapse was flat in population size from 64 to 4096, flat in
offspring count, flat in damping, and flat across seven algorithm variants.

CMA-ES answers both halves of that:

- **It learns a metric.** The sampling distribution is a full multivariate normal whose
  covariance adapts toward the inverse Hessian. On an ellipsoid of condition number
  `c`, an isotropic search needs `O(c)` evaluations; CMA-ES needs `O(log c)`.
- **It knows when it has converged.** Cumulative step-size adaptation, plus stopping
  criteria read off the distribution's own state, end a converged run rather than
  spending the remaining budget at zero velocity.

## Planned features

| Feature                     | Purpose                                                          |
| --------------------------- | ---------------------------------------------------------------- |
| Full-covariance CMA-ES      | the standard strategy, `n` up to a few hundred                   |
| Active CMA                  | negative rank-µ weights; the modern reference default            |
| sep-CMA-ES                  | diagonal covariance, `O(n)` per sample, for large `n`            |
| Block-diagonal CMA-ES       | one small block per parameter group; matches structured problems |
| IPOP / BIPOP restarts       | population-doubling restart strategies for multimodal problems   |
| Deterministic parallel eval | bit-identical to a serial run of the same seed                   |
| WebAssembly demo            | watch the covariance ellipse align with the valley               |

## Install

```sh
go get github.com/CWBudde/go-cma-es
```

The module path ends in `go-cma-es`; the package is named `cmaes`, because a Go package
name cannot contain a hyphen.

```go
import "github.com/CWBudde/go-cma-es"
```

## Usage

Not yet available — the public API lands in PLAN.md Phase 1. It will follow the sibling
libraries' shape:

```go
config := cmaes.NewDefaultConfig()
config.ObjectiveFunc = cmaes.Rosenbrock
config.ProblemSize = 10
config.LowerBound, config.UpperBound = -5, 5

result, err := cmaes.Optimize(config)
```

## Development

```sh
just setup-deps   # install the pinned formatters and linters
just check        # format, tidy, lint, test
just ci           # the full gate as CI runs it
```

See [`AGENTS.md`](AGENTS.md) for the working contract and [`PLAN.md`](PLAN.md) for the
roadmap.

## References

- Hansen, N. (2016). _The CMA Evolution Strategy: A Tutorial._ arXiv:1604.00772.
- Hansen, N. & Ostermeier, A. (2001). _Completely Derandomized Self-Adaptation in
  Evolution Strategies._ Evolutionary Computation 9(2), 159–195.
- Jastrebski, G. & Arnold, D. (2006). _Improving Evolution Strategies through Active
  Covariance Matrix Adaptation._ CEC 2006.
- Ros, R. & Hansen, N. (2008). _A Simple Modification in CMA-ES Achieving Linear Time and
  Space Complexity._ PPSN X.
- Auger, A. & Hansen, N. (2005). _A Restart CMA Evolution Strategy With Increasing
  Population Size._ CEC 2005.

## License

MIT — see [LICENSE](LICENSE).
