# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because a released version pins a search trajectory, **any change to the update rules is
a breaking change for reproducibility** even when the API is untouched: a given seed will
no longer reproduce the run an earlier version described. Such a change must bump
`Version` in `version.go` and say so explicitly here, so a consumer's checkpoint guard can
refuse to resume across it.

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-08-26

The first release. It carries everything built in Phases 0-10, 12 and 13: full-covariance,
separable and block-diagonal CMA-ES, active negative weights, IPOP and BIPOP restarts,
Hansen's boundary handling and stopping criteria, the WebAssembly demo, and the shared
benchmark function suite. `PLAN.md` had scoped block-diagonal covariance to a later
`v0.2.0`; since nothing shipped before now, it lands here instead.

### Added

- Repository scaffolding: module `github.com/CWBudde/go-cma-es`, package `cmaes`, the
  pinned formatter/linter toolchain, CI gates, and `PLAN.md` (Phase 0).
- `Version`, for a consumer's checkpoint and resume guard.
- Public CMA-ES types, dimension-aware configuration presets, Hansen parameter
  derivation, validation, and strict JSON configuration persistence (Phase 1).
- A dependency-free cyclic Jacobi eigendecomposition, covariance matrix helpers,
  numerical reference tests, and recorded scaling benchmarks (Phase 2).
- The passive full-covariance CMA-ES strategy: sampling, weighted recombination,
  cumulative step-size adaptation, evolution paths, rank-one and rank-µ covariance
  updates, lazy eigendecomposition, deterministic parallel evaluation, and the
  `Optimize` and `OptimizeContext` entry points (Phase 3).
- Clamp, repeated-reflection, and Hansen linear/quadratic transformation boundary
  handling, with the transformation plus Hansen's adaptive per-coordinate `BoundPenalty`
  as the default; nonlinear inequality and equality constraints with Deb feasibility or
  linear/quadratic penalty ranking (Phase 4).
- Hansen's TolX, TolFun, TolXUp, covariance-condition and no-effect stopping criteria;
  target and stagnation termination; run-scoped initial population and mean seeding;
  deep-copy progress, population and distribution observers; structured logging;
  convergence, sigma and condition-number histories; and best-so-far cancellation
  results (Phase 5).
- Active CMA with guarded negative rank-µ weights and Mahalanobis normalization, enabled
  by default; separable CMA-ES with diagonal-only `O(n)` sampling and covariance updates,
  the Ros-Hansen learning-rate correction, full/separable dispatch, and comparative
  performance and objective tests (Phase 6).
- IPOP and BIPOP restart schedules under one shared evaluation budget; deterministic
  random restart means and per-run seeds; large/small BIPOP budget accounting; and
  `RestartResult` records containing each run's population, allocation, best solution,
  and termination reason (Phase 7).
- Block-diagonal covariance with consecutive `BlockSize` partitions and optional
  non-contiguous `BlockGroups`; independently decomposed small matrices, sparse lifecycle
  snapshots, learning-rate scaling by maximum block width, bit-exact full/separable
  endpoints, and the recorded n=14,000, k=7 cost (Phase 8).
- A framework-free WebAssembly showcase with four contour-shaded objectives, replayable
  population and 2σ covariance views, a same-seed/same-budget isotropic comparison,
  aligned cost/σ/condition telemetry, public-API IPOP population doubling, reproducible
  build tooling, and GitHub Pages deployment (Phase 12).
- `functions.go`: the benchmark objective suite shared verbatim with Mayfly and
  Dragonfly -- Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel, Levy, Zakharov,
  Michalewicz, DixonPrice, BentCigar, Discus, Weierstrass, HappyCat and
  ExpandedSchafferF6 -- plus the new n-dimensional `Himmelblau`. The WebAssembly demo now
  consumes the library functions instead of local copies (Phase 13).
- Six further WebAssembly demo landscapes built on the library suite -- Sphere, Ackley,
  Schwefel, Michalewicz, Zakharov and the expanded Schaffer F6 -- each with a tuned start
  point and σ₀. `landscapeSpec` now records the expected optimum value and a tolerance,
  so a landscape whose optimum is not exactly zero can still be asserted.

- Per-dimension `Config.LowerBounds` and `Config.UpperBounds`. The scalar `LowerBound` and
  `UpperBound` still work and broadcast to every coordinate when the slices are nil.
  Boundary handling, the linear/quadratic shoulder widths, and validation are now
  resolved one coordinate at a time.
- `Result.IterationBestHistory`, the best cost within each generation's population.
  Unlike `ConvergenceCurve`, which is the running global best and therefore monotone, it
  can rise, which is what makes an oscillation visible next to the sigma curve.
- `Result` and `Best` now carry snake_case JSON tags, matching the configuration types.
- Saved configuration files carry a `format_version`. A file without one is read as the
  current version, so existing files still load.

### Changed

All of the following alter the search trajectory relative to an untagged pre-release
revision. Because `0.1.0` is the first release, there is no earlier `Version` to compare a
seed against: a consumer's checkpoint guard has nothing to refuse a resume across until
the next release changes the update rules.

- Covariance adaptation learns from the **sampled** step rather than the boundary-repaired
  one, for the rank-µ terms and the active negative weights alike, in both the full and
  the separable mode. Mean recombination and both evolution paths still follow the
  repaired position. Affects `clamp` and `reflect` runs only; `penalty` never repaired a
  genotype and is bit-identical to before.
- The default boundary penalty is Hansen's adaptive `BoundPenalty`: per-coordinate weights
  scaled by each coordinate's variance `σ²·C_ii`, adapted across generations, with the
  out-of-bounds-mean term. It replaces a single interquartile-range-scaled scalar whose
  degenerate fallback was a bare `1.0` in raw objective units.
- `TolX` is relative to the initial sigma, matching Hansen's `1e-12·σ⁽⁰⁾`, and the default
  moved from `1e-11` to `1e-12`. It was previously absolute, so a problem started at a
  small `InitialSigma` terminated on iteration 1.
- `TolFun` uses the best-of-generation value over its window together with the current
  generation's spread, rather than the full spread of every generation in the window.
- `TolXUp` measures growth against the initial distribution extent `σ⁽⁰⁾·max(D⁽⁰⁾)`
  instead of assuming `D⁽⁰⁾ = I`.
- `Validate` now rejects an `InitialMean` outside the bounds, a positive `MaxEvaluations`
  smaller than `Lambda`, and a `ConstraintConfig` that is configured but carries no
  constraint functions. All three were previously accepted.
- `IsFeasible` treats a negative aggregate violation as feasible and `NaN` as infeasible.

### Fixed

- `Optimize` no longer writes the generated `*rand.Rand` back into the caller's `Config`.
  A seeded configuration could previously be optimized only once — the second call failed
  validation with "seed and Rand are mutually exclusive" — which would have broken
  `OptimizeWithRestarts` and any consumer that reuses a configuration.
- Active CMA no longer diverges on bound-active problems under `clamp` and `reflect`.
  Negative rank-µ weights land on the worst candidates, which are the repaired ones, so
  reflection folded their direction back toward the interior and the strategy subtracted
  variance along the direction of the optimum. Measured on a bound-active problem,
  `reflect` failed every one of 30 seeds under the shipped defaults.
- A cancelled context no longer produces a different `Result` depending on
  `EnableParallel`. A generation in which every candidate was evaluated is now completed
  identically in both modes.
- `DistributionSnapshot` and `ConditionNumberHistory` describe a single generation.
  Eigenvectors and eigenvalues were taken from the lazily refreshed decomposition while
  the mean and sigma were post-update, so the reported condition number was a step
  function that only moved when the decomposition happened to refresh.
- A non-positive eigenvalue is repaired to a small positive value rather than floored to
  exactly zero. Zero left that axis with a step size of zero, killing the dimension for
  the rest of the run and tripping the condition-number criterion immediately. A
  genuinely ill-conditioned positive-definite covariance keeps its condition number and
  can still trip `ConditionCov`.
- A Jacobi eigendecomposition that fails to converge panics instead of silently returning
  a half-diagonalized matrix.
- `BetterConstrainedCandidate` is a strict weak ordering for every input. A `NaN`
  violation was incomparable with everything, making equality non-transitive and the sort
  order arbitrary.
- Observer and logger panics are contained and reported rather than destroying the run.
- Constraint functions lost across a JSON round-trip now fail loudly at `Validate` instead
  of yielding a silently unconstrained run.
- The per-iteration log event moved from info to debug level; a failed run emits a
  terminal event, so every logged start has exactly one logged end.
- The coverage gate no longer conflates "no statements to cover" with "0% covered";
  `go tool cover -func` reports 0.0% for both.

[Unreleased]: https://github.com/CWBudde/go-cma-es/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CWBudde/go-cma-es/releases/tag/v0.1.0
