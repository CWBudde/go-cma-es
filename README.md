# go-cma-es

A dependency-free Go implementation of the **Covariance Matrix Adaptation Evolution
Strategy** (CMA-ES), following Hansen's tutorial (arXiv:1604.00772).

Third in a family with [Mayfly](https://github.com/cwbudde/mayfly) and
[Dragonfly](https://github.com/CWBudde/dragonfly), and deliberately unlike both of them.

> **Status: under construction.** Phases 0–10, 12 and 13 provide active full-covariance,
> separable, and block-diagonal CMA-ES, box-boundary handling, nonlinear constraints,
> convergence criteria, lifecycle observers, IPOP/BIPOP restarts, and a WebAssembly
> showcase.
> [`PLAN.md`](PLAN.md) is the source of truth for progress.

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

## Features and roadmap

| Feature                     | Status | Purpose                                                          |
| --------------------------- | ------ | ---------------------------------------------------------------- |
| Full-covariance CMA-ES      | ready  | the standard strategy, `n` up to a few hundred                   |
| Active CMA                  | ready  | guarded negative rank-µ weights; the modern reference default    |
| sep-CMA-ES                  | ready  | diagonal covariance, `O(n)` per sample, for large `n`            |
| Block-diagonal CMA-ES       | ready  | one small block per parameter group; matches structured problems |
| IPOP / BIPOP restarts       | ready  | shared-budget restart strategies for multimodal problems         |
| Deterministic parallel eval | ready  | bit-identical to a serial run of the same seed                   |
| WebAssembly demo            | ready  | watch the covariance ellipse align with the valley               |
| Benchmark function suite    | ready  | the same objectives as Mayfly and Dragonfly, for comparable runs |

## Live demo

Open the [Covariance Lab](https://cwbudde.github.io/go-cma-es/) to watch the actual Go
library, compiled to WebAssembly, optimize ten 2-D landscapes. Its headline view replays
the sampled population, updated mean, best trail, and the 2σ ellipse reconstructed from
`DistributionSnapshot`.

The companion pages put CMA-ES beside a fixed isotropic search on a rotated
condition-10⁶ ellipsoid, align best cost with σ and covariance-condition histories, and
map the public IPOP population-doubling schedule across Rastrigin basins. The restart page
renders the per-run records returned by `OptimizeWithRestarts`.

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

Configurations are dimension-aware because Hansen's population and learning parameters
depend on the problem size:

```go
config := cmaes.NewDefaultConfig(10)
config.ObjectiveFunc = func(x []float64) float64 {
	var cost float64
	for _, value := range x {
		cost += value * value
	}
	return cost
}
config.LowerBound, config.UpperBound = -5, 5

result, err := cmaes.Optimize(config)
if err != nil {
	log.Fatal(err)
}
fmt.Printf("best cost: %g\n", result.GlobalBest.Cost)
```

Active covariance adaptation is enabled by default. It uses the worst-ranked samples to
reduce variance in unproductive directions, with the published weight-mass and
Mahalanobis-length guards that preserve positive-definiteness. Set `config.ActiveCMA =
false` to reproduce the passive update.

For high-dimensional problems whose important scaling is coordinate-aligned, use
`NewSeparableConfig` (or set `CovarianceMode = CovarianceSeparable`). It retains only the
covariance diagonal, requires no eigendecomposition, and takes `O(n)` time and storage
per sample. It cannot learn rotated correlations; use full covariance when those matter.
The reproducible n=200 benchmark is `BenchmarkCovarianceModesN200`.

For high-dimensional problems with correlations confined to small parameter groups, use
block-diagonal covariance. Consecutive groups need only a block size:

```go
config := cmaes.NewBlockDiagonalConfig(14_000, 7)
```

For non-contiguous groups, set an exact partition of all coordinate indices. Explicit
groups take precedence over `BlockSize`:

```go
config := cmaes.NewBlockDiagonalConfig(9, 0)
config.BlockGroups = [][]int{{0, 3, 6}, {1, 4, 7}, {2, 5, 8}}
```

Block mode uses `O(n*k)` matrix storage and sampling work for maximum block size `k`,
while decomposing all blocks costs `O(n*k^2)`. `BlockSize == n` is bit-identical to full
CMA-ES and `BlockSize == 1` is bit-identical to sep-CMA-ES. A distribution observer
receives the sparse eigensystems in `DistributionSnapshot.Blocks`; its dense
`Eigenvectors` field is nil in a genuine block run. See
[`docs/blockdiag-cost.md`](docs/blockdiag-cost.md) for the n=14,000, k=7 measurement.

The default `BoundaryPenalty` method uses Hansen's smooth linear/quadratic transformation
to evaluate every candidate inside the box while retaining its latent Gaussian step for
covariance adaptation. A scale-adaptive penalty discourages remote periodic copies.
`BoundaryClamp` and `BoundaryReflect` are also available, but both feed repaired samples
back into adaptation. In particular, repeated clamping creates duplicate samples on a
face and can bias the learned covariance.

This differs deliberately from Dragonfly's default `wrap`. Wrapping is a useful
exploration rule for a swarm moving in a toroidal search space, but its discontinuity at
opposite faces is a poor default for a strategy whose central job is to learn a local
Gaussian metric.

Nonlinear inequalities use `g(x) <= 0`; equalities are satisfied within
`EqualityTolerance`. Deb's feasibility rules are the factor-free default:

```go
config.Constraints = &cmaes.ConstraintConfig{
	Inequalities: []cmaes.ConstraintFunction{
		func(x []float64) float64 { return x[0] + x[1] - 1 },
	},
	Equalities: []cmaes.ConstraintFunction{
		func(x []float64) float64 { return x[2] - x[3] },
	},
	EqualityTolerance: 1e-6,
}
```

Set `Handling` to `ConstraintHandlingPenalty`, choose `PenaltyLinear` or
`PenaltyQuadratic`, and provide a positive `PenaltyFactor` to rank by penalized cost
instead.

Hansen's distribution-derived stopping criteria are enabled by default: TolX, TolFun,
TolXUp, covariance condition number, and the no-effect-axis and no-effect-coordinate
checks. A target and stagnation window are opt-in:

```go
target := 1e-10
config.Convergence.TargetCost = &target
config.Convergence.MinImprovement = 1e-12
config.Convergence.StagnationIterations = 50
config.Convergence.MinIterations = 10
```

Set an individual numeric tolerance to zero, or a no-effect flag to false, to disable
that criterion. Set `config.Convergence = nil` to run only to the iteration or evaluation
cap. Every completed iteration adds aligned entries to `ConvergenceCurve`,
`SigmaHistory`, and `ConditionNumberHistory`.

Run options provide seeding, synchronous observation, and structured logging without
mutating the reusable configuration:

```go
result, err := cmaes.OptimizeContext(
	ctx,
	config,
	cmaes.WithInitialMean(previous.Position, 0.2),
	cmaes.WithProgressObserver(func(progress cmaes.Progress) {
		fmt.Printf("%d: cost=%g\n", progress.Iteration, progress.Best.Cost)
	}),
	cmaes.WithDistributionObserver(func(snapshot cmaes.DistributionSnapshot) {
		fmt.Printf("sigma=%g condition=%g\n", snapshot.Sigma, snapshot.ConditionNumber)
	}),
)
```

Population and distribution observers are opt-in because they deep-copy the data they
expose. Cancellation after a run starts returns its best-so-far with
`TerminationCancelled`; a context canceled before startup still returns
`context.Canceled` and no result.

For multimodal objectives, set a positive global evaluation budget and select IPOP or
BIPOP. IPOP doubles λ after each completed run, except where the shared budget is
nearly spent and the final run's λ is capped at what remains. BIPOP spends from
whichever of its large and small regimes has used fewer evaluations; small runs
randomize their population, initial sigma, and capped budget. Both strategies use
fresh deterministic means inside the configured box and return an auditable record
for every run:

```go
config.MaxEvaluations = 100_000
result, err := cmaes.OptimizeWithRestarts(config, cmaes.RestartBIPOP)
if err != nil {
	log.Fatal(err)
}
for _, run := range result.Restarts {
	fmt.Printf("%s λ=%d evaluations=%d best=%g\n",
		run.Regime, run.Lambda, run.Evaluations, run.Best.Cost)
}
```

A target-cost termination ends the overall schedule early. Other convergence and run-cap
terminations start a new run while the shared budget has evaluations left.
`OptimizeWithRestartsContext` provides cancellation and accepts the same lifecycle run
options; observers receive counts local to each run, and initialization options apply to
the first run only.

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
- Hansen, N. (2009). _Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function
  Testbed._ GECCO Companion 2009.

## License

MIT — see [LICENSE](LICENSE).
