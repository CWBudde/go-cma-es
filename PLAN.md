# CMA-ES (Go) — Implementation Plan

Module: `github.com/CWBudde/go-cma-es`
Package: `cmaes` (flat, at the repository root)
Status: **Phases 0–10, 12 and 13 complete.** Full, separable, and block-diagonal CMA-ES,
active covariance adaptation, boundary handling, nonlinear constraints, convergence
criteria, lifecycle observers, IPOP/BIPOP restarts, the WebAssembly showcase, and the
shared benchmark function suite are implemented. Released as `v0.1.0`.

This document is the roadmap and the single source of truth for progress. It is organised
the same way as the sibling [Mayfly](https://github.com/cwbudde/mayfly) and
[Dragonfly](https://github.com/CWBudde/dragonfly) projects' `PLAN.md`: numbered phases,
`- [ ]` / `- [x]` task checkboxes, and a bolded `**Rationale**:` paragraph after each
subsection explaining _why_ the task exists. Update the checkboxes as work lands.

---

## 0. Project overview

### Why this library exists

Every optimizer the consuming project (MayFlyCircleFit) has tried belongs to one family.
Mayfly with seven variants, Dragonfly, and quasi-random initialization are all
position/velocity updates with an **isotropic, externally scheduled step size and no
learned metric**. That project's `docs/restart-vs-budget-report.md` measured the
consequence directly:

- population spread falls below 10% of its initial value by iteration 11–16;
- in the final 786,432 offspring, not one beats both parents;
- a 2048-iteration run is frozen from about iteration 640 — roughly 1.4 million
  evaluations producing no change at all.

The collapse is **flat in population size (64→4096), flat in offspring count, flat in
damping, and flat across every variant**. Competing conventions and a deceptive landscape
were both tested as explanations and rejected. Cold restarts recovered about 157 cost
points precisely by discarding the collapsed state.

CMA-ES supplies the two capabilities none of those contenders has:

1. **A learned metric.** Rank-one and rank-µ covariance adaptation move the sampling
   distribution toward the inverse Hessian, so parameters of different curvature are not
   forced to share one step size.
2. **Step-size control that detects its own stagnation.** Cumulative step-size adaptation
   plus criteria read off the distribution's state (TolX, TolFun, condition number,
   no-effect axis) terminate a converged run instead of continuing it.

### Design principles (inherited from Mayfly and Dragonfly — do not diverge without a note here)

1. **Standard library only.** The single direct dependency is `github.com/cucumber/godog`,
   and it is test-only. No numeric or utility third-party packages. **This is why the
   symmetric eigendecomposition is written here** (Phase 2) rather than taken from gonum.
2. **Flat root package.** Source and tests are siblings at the repo root. No `internal/`,
   no `pkg/`, no `cmd/`.
3. **Configuration-driven.** One flat `Config` struct with snake_case JSON tags, field
   order chosen for `fieldalignment` (pointers/interfaces → strings → float64 → int →
   bool). Factory functions (`NewDefaultConfig`, `NewSeparableConfig`, …) layer presets on
   top.
4. **Explicit RNG threading.** Every stochastic helper takes `rng *rand.Rand` as its last
   parameter. `Config.Rand` is the injection point; when nil, `OptimizeContext` creates a
   run-local generator, never writes to `Config`, and records the seed in `Result.Seed`.
   `Config` is read-only for the duration of a run, so the same configuration can be
   optimized twice — which `OptimizeWithRestarts` (Phase 7) depends on.
5. **Deterministic parallelism.** All RNG draws happen on the calling goroutine during a
   `prepare*` phase; worker goroutines only evaluate the objective. A seeded run must
   produce bit-identical results with `EnableParallel` on or off.
6. **White-box tests.** Test files live in `package cmaes` and may exercise unexported
   helpers. `example_test.go` is the sole black-box file.
7. **Reference fidelity first, ergonomics second.** Where Hansen's tutorial and
   `purecmaes.m` disagree, implement the tutorial and expose the alternative behind a
   config field.

**Rationale**: The three libraries are meant to read as a family. A user who knows
`dragonfly.Optimize(config)` should be able to guess `cmaes.Optimize(config)` without
opening the docs, and a contributor moving between the repos should not have to relearn
the conventions.

### Scope

| Release  | Contents                                                                                                 |
| -------- | -------------------------------------------------------------------------------------------------------- |
| `v0.1.0` | full-covariance CMA-ES, active (negative) weights, sep-CMA-ES, IPOP and BIPOP, block-diagonal covariance |
| `v0.2.0` | fix: `ActiveCMA` was silently inert wherever the rank-mu rate reached its ceiling                        |

Block-diagonal covariance was planned for `v0.2.0`, but nothing was tagged before it
landed, so the first release carries it too.

---

## 1. The algorithm — specification to implement against

Primary source: Hansen, N. (2016). _"The CMA Evolution Strategy: A Tutorial."_
arXiv:[1604.00772](https://arxiv.org/abs/1604.00772) — the normative pseudocode.

Supporting sources:

- Hansen, N. & Ostermeier, A. (2001). _Completely Derandomized Self-Adaptation in
  Evolution Strategies._ Evolutionary Computation 9(2), 159–195.
- Jastrebski, G. & Arnold, D. (2006). _Improving Evolution Strategies through Active
  Covariance Matrix Adaptation._ CEC 2006 — negative weights.
- Ros, R. & Hansen, N. (2008). _A Simple Modification in CMA-ES Achieving Linear Time and
  Space Complexity._ PPSN X — sep-CMA-ES.
- Auger, A. & Hansen, N. (2005). _A Restart CMA Evolution Strategy With Increasing
  Population Size._ CEC 2005 — IPOP.
- Hansen, N. (2009). _Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function
  Testbed._ — BIPOP.

Reference implementations to cross-check against: `purecmaes.m` and `cma.py`.

### 1.1 One generation

For a distribution with mean `m`, step size `σ`, and covariance `C = B·D²·Bᵀ`:

```
sample        x_k = m + σ·B·D·z_k,          z_k ~ N(0, I),   k = 1..λ
select        sort by f(x_k); keep the µ best
recombine     m' = Σ w_i · x_{i:λ}
step path     p_σ ← (1-c_σ)·p_σ + sqrt(c_σ(2-c_σ)µ_eff)·C^(-1/2)·(m'-m)/σ
step size     σ  ← σ · exp( (c_σ/d_σ)·( ||p_σ|| / E||N(0,I)|| - 1 ) )
cov path      p_c ← (1-c_c)·p_c + h_σ·sqrt(c_c(2-c_c)µ_eff)·(m'-m)/σ
covariance    C  ← (1-c_1-c_µ)·C + c_1·(p_c p_cᵀ + δ(h_σ)·C) + c_µ·Σ w_i·y_i y_iᵀ
```

Three details that are easy to get wrong and must each be covered by a unit test:

- The **Heaviside correction `h_σ`** gates the `p_c` update and adds the `δ(h_σ)·C`
  compensation term to the rank-one update. Omitting it degrades performance in a way
  that looks like ordinary noise on easy functions and only shows up on hard ones.
- `C^(-1/2)` in the `p_σ` update is `B·D^(-1)·Bᵀ`, **not** `C^(-1)` and not an
  element-wise reciprocal.
- The **eigendecomposition is lazy.** Recomputing every generation is correct but wasteful;
  the published trigger is `evaluations - eigenEval > λ / (10·n·(c_1 + c_µ))`. Getting the
  staleness bound wrong is a silent accuracy bug, not a crash.

---

## Phase 0 — Repository scaffolding ✅

- [x] `git init`; `go mod init github.com/CWBudde/go-cma-es`, `go 1.23.3`
- [x] Copy and adapt from Dragonfly: `.gitignore`, `.golangci.toml`, `treefmt.toml`,
      `justfile` (pinned tool versions kept in sync), `LICENSE`
- [x] `justfile`: drop the Dragonfly-specific study recipes; make `check-examples`
      discover nested modules instead of hard-coding them, and make `check-wasm-demo`
      skip cleanly until Phase 12 creates the demo
- [x] `AGENTS.md` + `CLAUDE.md` pointer file
- [x] `README.md`, `CHANGELOG.md` with an `Unreleased` section
- [x] `PLAN.md` — this document
- [x] `.github/workflows/` — `test.yml`, `security.yml`, `release.yml`
- [x] `doc.go` with the package comment
- [x] `version.go` + `version_test.go` — `Version`, for a consumer's resume guard
- [x] Make the coverage gate distinguish "no statements yet" from "0% covered" — both
      `just check-coverage` and the CI step, which reported the same 0.0% for each
- [x] `go build ./... && go vet ./... && go test ./...` green

Deferred out of this phase, deliberately:

- [x] `.github/workflows/wasm-demo-pages.yml` — lands in Phase 12 with the demo it
      publishes. Committing it now would add a workflow that fails on every push.

**Rationale**: A contributor moving between the three repos should not have to relearn
conventions, and every later phase should land into working CI rather than build it. The
`version.go`/`version_test.go` pair is not filler: the consuming project pins optimizer
versions in its checkpoints and refuses to resume across a version that changed the search
trajectory, so the constant has to exist before anything depends on it. The coverage-gate
fix was forced by this phase rather than chosen: a scaffold has no executable statements,
`go tool cover` reports that as 0.0%, and the 80% floor would therefore have failed every
build until Phase 1 landed.

---

## Phase 1 — Types, config, and the public surface

- [x] `types.go`: `ObjectiveFunction`, `ConstraintFunction`, `Best`, `Result`,
      `TerminationReason` (`maximum_iterations`, `maximum_evaluations`, `target_cost`,
      `stagnation`, `tol_x`, `tol_fun`, `condition_number`, `no_effect_axis`,
      `no_effect_coord`)
- [x] `types.go`: `Config` with `ObjectiveFunc`, `Rand`, `Seed`, `Convergence`,
      `Constraints`, `BoundaryMethod`, `CovarianceMode`, `ProblemSize`, `LowerBound`,
      `UpperBound`, `InitialMean`, `InitialSigma`, `Lambda`, `Mu`, `MaxIterations`,
      `MaxEvaluations`, `MaxWorkers`, `ActiveCMA`, `EnableParallel`
- [x] `config.go`: `NewDefaultConfig` with Hansen's defaults — `λ = 4 + floor(3·ln n)`,
      `µ = floor(λ/2)`, log-decreasing weights, `µ_eff`, `c_σ`, `d_σ`, `c_c`, `c_1`, `c_µ`
- [x] `config.go`: `NewSeparableConfig`, `NewHighDimensionalConfig`,
      `NewFastConvergenceConfig`
- [x] `config.go`: `Validate()` — reject non-positive `InitialSigma`, `Lambda < 2`,
      `Mu > Lambda`, mismatched `InitialMean` length, non-finite bounds
- [x] `config_loader.go` + tests: JSON round-trip
- [x] Table-driven test of the derived parameters against published values for
      n ∈ {2, 10, 56, 100}

**Rationale**: The parameter formulas are where CMA-ES implementations most often go
quietly wrong. Isolating them in one file with a table of published values makes a
transcription error a test failure, rather than a mysteriously poor fit discovered six
phases later.

---

## Phase 2 — Linear algebra (standard library only)

- [x] `eigen.go`: cyclic **Jacobi eigenvalue algorithm** for real symmetric matrices,
      returning eigenvalues and orthonormal eigenvectors
- [x] `eigen.go`: enforce symmetry (`C := (C + Cᵀ)/2`) and floor negative eigenvalues
      before the square root — the two guards every production CMA-ES carries
- [x] `eigen_test.go`: reconstruction `B·diag(D)·Bᵀ ≈ C` to 1e-12; orthonormality
      `BᵀB ≈ I`; known-answer cases (diagonal, 2×2 rotation, Hilbert 6×6); an
      ill-conditioned input; the `n = 1` edge case
- [x] `matrix.go`: symmetric rank-one update, matrix-vector product, condition number
- [x] Benchmark the decomposition at n = 56, 200, 1000; record it in `docs/eigen-cost.md`

**Rationale**: This is the only genuinely non-obvious numerics in the library, and it is
where the standard-library-only constraint bites. Building it standalone and provably
correct means every convergence bug found later is an algorithm bug rather than a
linear-algebra bug. The benchmark also settles the recurring "is CMA-ES too slow?"
objection with a number: measured against one 512×512 render, the decomposition at n = 56
should be free, and the n = 1000 figure locates the point where sep-CMA becomes mandatory.

---

## Phase 3 — Core CMA-ES

- [x] `cmaes.go`: state — mean `m`, step size `σ`, covariance `C`, `B`, `D`, evolution
      paths `p_σ` and `p_c`, `eigenEval` counter
- [x] Sampling `x_k = m + σ·B·D·z_k`
- [x] Selection and weighted recombination of the µ best
- [x] `p_σ` update and cumulative step-size adaptation
- [x] `p_c` update with the Heaviside `h_σ` correction
- [x] Rank-one and rank-µ covariance update
- [x] Lazy eigendecomposition on the published staleness trigger
- [x] `Optimize(config) (*Result, error)` and `OptimizeContext(ctx, config, ...RunOption)`
- [x] `cmaes_test.go`: sphere below 1e-10; **ellipsoid at condition 1e6**; Rosenbrock
      n = 10 below 1e-6; identical seeds are bit-identical; `Result.FuncEvalCount` matches
      a counting objective

**Rationale**: The ellipsoid test is the whole thesis of this library in one assertion. An
isotropic optimizer needs O(cond) evaluations on it and CMA-ES needs O(log cond); if that
test does not pass cleanly, the covariance adaptation is wrong and nothing downstream is
worth measuring.

---

## Phase 4 — Boundary handling and constraints

- [x] `boundary.go`: `BoundaryMethod` — `clamp`, `reflect`, and Hansen's
      **transformation + penalty**
- [x] Default to the penalty method; document why it differs from Dragonfly's `wrap`
- [x] `constraints.go`: `ConstraintConfig` with Deb feasibility rules and penalty methods,
      matching the sibling API shape field for field
- [x] Tests: a bound-active optimum is found under all three methods; the penalty method
      does not distort σ on an unconstrained problem

**Rationale**: The API must feel like Dragonfly's so a consumer's adapter can be a near
copy of its `dragonfly_adapter.go`, but the _default_ must be what is correct for this
algorithm. Naive clamping biases the covariance estimate: repeatedly pinning samples to a
bound is exactly how a CMA-ES run reports a healthy σ while going nowhere.

---

## Phase 5 — Convergence and lifecycle

- [x] `convergence.go`: `ConvergenceConfig` (`TargetCost`, `MinImprovement`,
      `StagnationIterations`, `MinIterations`) — the sibling shape
- [x] CMA-ES's own criteria: `TolX`, `TolFun`, `TolXUp`, `ConditionCov`, `NoEffectAxis`,
      `NoEffectCoord`, each mapping to a distinct `TerminationReason`
- [x] `lifecycle.go`: `Progress`, `ProgressObserver`, `PopulationSnapshot`,
      `PopulationObserver`, `Logger`, `RunOption`, `WithInitialPopulation`,
      `WithProgressObserver`, `WithPopulationObserver`, `WithLogger`
- [x] `WithInitialMean(m, sigma)` — the seeding hook a consumer's `Initial` candidate needs
- [x] `DistributionSnapshot` + `WithDistributionObserver`: mean, σ, eigenvalues `D`,
      eigenvectors `B`, condition number, per iteration, as deep copies and opt-in
- [x] `monitoring.go`: `ConvergenceCurve`, plus σ and condition-number history
- [x] Cancellation after at least one evaluation returns the best-so-far with a cancelled
      termination reason; a context already cancelled on entry returns an error, because
      nothing has been computed yet
- [x] Tests: every criterion fires on a constructed case and reports its own reason

**Rationale**: The consuming project's reports were only possible because runs were
observable. Recording σ and the condition number per iteration means the _next_
premature-convergence investigation reads its answer off a curve instead of re-instrumenting
the optimizer. `DistributionSnapshot` is also what Phase 12 draws its ellipse from, which
is why it is a first-class observer rather than a debug hook.

---

## Phase 6 — sep-CMA-ES and active CMA

- [x] `separable.go`: diagonal-only covariance, `O(n)` per sample, no eigendecomposition
- [x] `CovarianceMode` (`full`, `separable`) dispatch, with the Ros & Hansen learning-rate
      correction `c_µ ← c_µ · (n + 2)/3`
- [x] `active.go`: negative rank-µ weights with the published guards — correct weight sums,
      positive-definiteness preserved
- [x] `ActiveCMA` config field, defaulting **on**
- [x] Tests: sep-CMA beats full CMA per unit _time_ on a separable n = 200 problem; active
      CMA is no worse than passive across sphere, ellipsoid and Rosenbrock

**Rationale**: sep-CMA is the scalability escape hatch. A 2000-circle joint stage is
n = 14000, where a full 14000×14000 covariance is not merely slow but does not fit in
memory. Phase 8's block-diagonal mode suits the consumer's problem better, but sep-CMA is
the standard, published, testable stepping stone to it.

---

## Phase 7 — IPOP and BIPOP restarts ✅

- [x] `restart.go`: `OptimizeWithRestarts` — on any convergence criterion, restart with
      `λ ← 2λ` until the budget is exhausted; return the best
- [x] BIPOP: interleave large-population (IPOP) runs with small-population runs on random
      small budgets, advancing whichever regime has consumed less budget
- [x] `RestartResult` with per-restart records: λ, evaluations, best cost, termination
      reason
- [x] Tests: Rastrigin n = 10 — restarts find the global optimum where a single run does
      not; the evaluation budget is respected exactly

**Rationale**: The consumer measured that restarts are worth about 157 cost points and
that restart _length_ barely matters — everything from 512 down to 64 iterations was
statistically indistinguishable. IPOP answers the question that ladder left open: it
decides when to restart from the algorithm's own state rather than from a fixed arm. BIPOP
covers both the many-short and few-long regimes the ladder had to choose between.

---

## Phase 8 — Block-diagonal covariance (v0.2.0) ✅

- [x] `blockdiag.go`: `CovarianceMode = "block"` with a configurable `BlockSize`, one k×k
      covariance per block, each decomposed independently
- [x] Record measured cost at n = 14000, k = 7: `O(n·k)` memory and per-sample work,
      `O((n/k)·k³) = O(n·k²)` per full decomposition. The earlier `O(n·k²)` per-sample
      estimate double-counted the number of blocks; each of the `n/k` blocks performs a
      k×k matrix-vector product. See `docs/blockdiag-cost.md`.
- [x] Optional `BlockGroups [][]int` for non-contiguous groupings
- [x] Tests: recovers a block-structured ellipsoid that sep-CMA cannot; degenerates exactly
      to full CMA at `BlockSize == n` and to sep-CMA at `BlockSize == 1`

**Rationale**: The consumer's cost is near-additive per circle and its conditioning lies
_within_ each circle's seven parameters rather than across circles, so block-diagonal is
the shape of the true Hessian at a fraction of full CMA's cost. It is also the least
standard piece here, which is why it lands last and why the two degeneracy tests matter:
they give it a reference to be correct against.

---

## Phase 9 — MayFlyCircleFit adapter ✅

Lands in the consuming repository as its PLAN.md Phase 19, on a topic branch and through a
pull request — its `main` is protected. See ../MayFlyCircleFit

- [x] `internal/opt/cmaes_adapter.go`, modelled on `dragonfly_adapter.go`: `CMAESAdapter`,
      `NewCMAES(maxIters, popSize, seed, ...CMAESOption)`, `WithCMAESLogger`,
      `WithCMAESEarlyStop`, `WithCMAESParallelEvaluation`
- [x] Implement `Optimizer`, `LifecycleOptimizer`, `IterationBudgetOptimizer`; honour
      `RunOptions.Initial` (→ `WithInitialMean`), `AdditionalSeeds`, `SeedOffset`,
      `ResumeCount`, `Observer`, `ProgressMapper`, `EpochObserver`
- [x] Map `Problem.Repair` and `Problem.Inequalities` onto the Phase 4 surface
- [x] Pin the module version in `internal/opt/version.go` and extend `resume_guard.go` so a
      checkpoint written under one version is refused by another
- [x] Satisfy `optimizer_contract_test.go` and `parallel_evaluation_test.go` unchanged; add
      `TestCMAESParallelEvaluationMatchesSerial`
- [x] **Decide and document** whether the consumer's `WithRestarts` wraps the adapter or
      the adapter uses this library's IPOP internally

**Rationale**: `opt.Optimizer` is a proven seam — Dragonfly went through it without
touching the pipelines — so the work is contained to one new file plus registration, and
the existing contract tests do most of the verification. The restart question needs an
explicit answer because `WithRestarts` and IPOP are two implementations of one idea, and
shipping both silently would be worse than shipping either.

---

## Phase 10 — Configuration, CLI, server, schedule

- [x] Add `cmaes` to the consumer's optimizer enum; refuse MayFly-only knobs (`variant`,
      `qmcInit`, `NC`, `DanceDamp`) under it, as `dragonfly` already is
- [x] Add the CMA-ES knobs (`initialSigma`, `covarianceMode`, `activeCMA`,
      `restartStrategy`) with defaults and limits
- [x] `cmd/run --optimizer cmaes`; job payload; schedule document `base.optimizer`
- [x] Web UI engine selector; regenerate and commit the templ output
- [x] Polishing stays MayFly-only for now — state that explicitly in the consumer's
      behavior invariants
- [x] Update the consumer's support matrix, known limitations, and `AGENTS.md`

**Rationale**: The engine-selection plumbing already exists end to end for `dragonfly`.
This phase follows an established path rather than building one, which is exactly why it
should be a separate, boring, reviewable step.

---

## Phase 11 — Measurement

**No figure currently recorded in the consumer's `docs/` is a valid baseline.** All of them
were taken under MayFly v0.6.0 or earlier and the pin is now v0.7.0, which changed results
for every variant. The baseline is re-measured here, first.

- [ ] Re-establish the baseline on the current pin, under the conditions in the consumer's
      `docs/restart-vs-budget-report.md`
- [ ] Twelve paired blocks, disjoint seed pools, shared seed prefix within a block; paired
      t-tests, df = 11
- [ ] **Evaluation-matched, not iteration-matched** — λ differs from `popSize`, and the
      consumer's AOBLMOA report shows how a per-iteration cost difference invalidates an
      iteration-matched comparison
- [ ] Arms: Mayfly standard single-run; Mayfly standard `r16`; CMA-ES single-run; CMA-ES
      IPOP; sep-CMA-ES IPOP
- [ ] Record σ and condition-number trajectories and compare them against the Mayfly spread
      collapse table, so the mechanism claim is measured rather than asserted
- [ ] Write the report in the consumer's house style: what was run, conditions, a result
      table with t and blocks-won, and an explicit "what this does not say" section
- [x] Preserve the operator-stopped preliminary subset in the consumer: three completed
      jobs and one interrupted IPOP job from the first paired block, with raw mechanism
      trajectories and a descriptive report that performs no inferential statistics

**Rationale**: That project's docs are unusually honest about what a measurement does and
does not establish, and that discipline is why its negative results stay reusable. A
CMA-ES report that did not meet the same bar would be worth less than no report. The first
campaign was intentionally stopped on 2026-08-25 once its several-day runtime was clear;
the preliminary report lives at `../MayFlyCircleFit/docs/cmaes-preliminary-report.md` and
does not satisfy any of the six twelve-block requirements above.

---

## Phase 12 — WebAssembly demo ✅

Follows the sibling pattern exactly: a nested `go.mod`, `main.go` behind
`//go:build js && wasm` with a `main_stub.go` for the native build, a `bridge.go`
marshalling layer, plain JS and canvas — no framework, no bundler.
**Depends only on Phases 1–5.**

- [x] `examples/wasm-demo/go.mod` with a `replace` onto the parent module;
      `scripts/build-wasm-demo.sh`; `boot.js` loading `wasm_exec.js`
- [x] `landscape.go` + `render.js`: contour-shaded 2-D landscapes — Rosenbrock,
      conditioned ellipsoid, Rastrigin, Himmelblau, Sphere, Ackley, Schwefel,
      Michalewicz, Zakharov, expanded Schaffer F6. Each spec carries its own optimum
      value and tolerance, because Michalewicz's optimum is negative and Schwefel's
      published minimizer and constant are both rounded
- [x] `index.html` — the headline view: the sampled population, the mean, and the **2σ
      covariance ellipse** drawn from `DistributionSnapshot`. Play/pause/step, seed box,
      λ and σ₀ controls
- [x] `compare.html` — CMA-ES against a fixed isotropic step, same seed, same budget, side
      by side on the condition-1e6 ellipsoid
- [x] `charts.html` — σ, condition number and best cost on one time axis
- [x] `restart.html` — IPOP: λ doubling across restarts, each restart's basin marked
- [x] `favicon.svg`, `style.css`, `README.md` matching the sibling demo's structure
- [x] `.github/workflows/wasm-demo-pages.yml` publishing to GitHub Pages
- [x] `just check-wasm-demo` already gates both build paths; drop its
      does-not-exist-yet skip once the directory lands

**Rationale**: The swarm demos can only show dots moving. CMA-ES can show _why_ it works —
an ellipse rotating to align with the valley floor and stretching along it is the learned
metric, made visible. That is both the best available README asset and a real debugging
tool: a covariance update with a sign error looks wrong on screen within a dozen
iterations, long before a benchmark table would catch it. It is worth pulling forward to
immediately after Phase 5 for exactly that reason.

---

## Phase 13 — Benchmark function suite ✅

- [x] `functions.go`: the sibling suite copied verbatim from Mayfly and Dragonfly — Sphere,
      Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel, Levy, Zakharov, Michalewicz,
      DixonPrice, BentCigar, Discus, Weierstrass, HappyCat, ExpandedSchafferF6 — with only
      the package clause and the header's sibling reference changed, so the three files
      stay diffable
- [x] `Himmelblau`, new in all three suites: four equal global minima in two dimensions,
      extended to n by summing over disjoint coordinate pairs, with an odd dimension's
      unpaired coordinate contributing its square
- [x] Single-objective only — the Deferred section rules out MO-CMA-ES, so Dragonfly's
      ZDT/Schaffer block is deliberately not ported
- [x] `functions_test.go`: the sibling tests, including the suite-wide
      `TestBenchmarkFunctionsEmptyInput` convention check, plus `TestHimmelblau`
- [x] Delete the duplicated `sphere` and `rastrigin` test helpers in favour of the library
      functions; the seeded tests keep their existing thresholds unchanged
- [x] `examples/wasm-demo/landscape.go` consumes `cmaes.Rosenbrock`, `cmaes.Rastrigin` and
      `cmaes.Himmelblau`; only the rotated condition-10⁶ ellipsoid stays local, because the
      sibling suite has no counterpart to it

**Rationale**: The two siblings ship one identical suite, and a CMA-ES result is only
directly comparable with a Mayfly or Dragonfly result if it was scored by the same
function — a reordered summation or a different Rastrigin constant is enough to make two
tables disagree for reasons that have nothing to do with the optimizers. Copying the file
rather than rewriting it is the point: the diff between the three is the guarantee. The
demo's local copies were a second source of the same drift, one dimension deep, and
folding them back removes it.

---

## Phase order

Phases 0 → 5 are strictly sequential. After Phase 5 the graph opens up:

```
0 → 1 → 2 → 3 → 4 → 5 ─┬─→ 6 → 7 → 8            (library depth)
                       ├─→ 12 → 13               (web demo, benchmark suite)
                       └─→ 9 → 10 → 11           (integration and measurement)
```

---

## Deferred

Recorded honestly rather than silently dropped.

- **Multi-objective CMA-ES (MO-CMA-ES).** Both siblings carry a multi-objective variant.
  This library does not plan one, because the consuming problem is single-objective.
  Revisit only on a concrete use case.
- **Gherkin/`features/` integration suite.** Both siblings have one and the `justfile`
  keeps the `test-integration` recipe, but no feature files are planned before Phase 3.
- **Surrogate-assisted CMA-ES (lq-CMA-ES).** A natural fit for an expensive objective like
  image rendering, and a plausible v0.3.0. Out of scope until the plain strategy has been
  measured, so that any surrogate gain is attributable.
