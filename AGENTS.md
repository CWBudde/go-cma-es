# Repository Guidelines

The contract for working in this repository, for contributors and coding agents alike.
`CLAUDE.md` is a pointer to this file; `AGENTS.md` is the one to edit.

## Project Overview

Go implementation of the **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)**,
following Hansen's tutorial (arXiv:1604.00772). A dependency-free metaheuristic
optimization library, and the third in a family with
[Mayfly](https://github.com/cwbudde/mayfly) and
[Dragonfly](https://github.com/CWBudde/dragonfly).

| Variant          | Problem class                             | Entry point                    |
| ---------------- | ----------------------------------------- | ------------------------------ |
| **CMA-ES**       | Single-objective, continuous, n ≲ few 100 | `Optimize` / `OptimizeContext` |
| **sep-CMA-ES**   | Same, high dimension, separable-ish       | `CovarianceMode = "separable"` |
| **block-CMA-ES** | Same, block-structured parameters         | `CovarianceMode = "block"`     |
| **IPOP / BIPOP** | Multimodal, restart strategies            | `OptimizeWithRestarts`         |

**Current status**: nothing is released. **Phases 0–9 and 12 are complete**, including
block-diagonal covariance, IPOP/BIPOP restarts, and the WebAssembly showcase.

**PLAN.md is the single source of truth for progress.** Before starting work, read
PLAN.md and check which boxes are ticked. Do not infer status from this file or from
package documentation.

**Module**: `github.com/CWBudde/go-cma-es`, package `cmaes`, flat at the repo root
(no `internal/`, no `pkg/`, no `cmd/`). Go 1.23.3. The module path ends in `go-cma-es`
while the package is named `cmaes`, because a Go package name cannot contain a hyphen.
Import it and refer to it as `cmaes`.

## Why this library exists

The consumer that motivated it — MayFlyCircleFit — measured that every swarm optimizer
it tried collapses its population diversity within 10–20 iterations regardless of
population size, offspring count, damping, or variant, and then spends the rest of its
budget frozen. See that project's `docs/restart-vs-budget-report.md`.

CMA-ES addresses both halves of that failure directly:

1. **It learns a metric.** Covariance adaptation means an isotropic step is not imposed
   on parameters with wildly different curvature.
2. **It detects its own stagnation.** Cumulative step-size adaptation plus the TolX /
   TolFun / condition-number criteria terminate a converged run instead of continuing it.

Keep that framing in mind when weighing a change: a modification that improves a
benchmark number while weakening either property is going the wrong way.

## Build & Development Commands

```bash
just build           # go build ./...
just test-quick      # go test -short ./...   (narrow loop while developing)
just test            # full suite with an HTML coverage report
just test-race       # go test -race -short
just check-coverage  # enforce the 80% floor rather than merely reporting it
just fmt             # treefmt
just lint            # golangci-lint
just check           # format + tidy + lint + test
just ci              # the full gate, as CI runs it
```

Formatters and linters are version-pinned in the `justfile` and installed by
`just setup-deps`. Run `just check-tools` if formatting passes locally and fails in CI —
treefmt runs with `--allow-missing-formatter`, so a missing formatter is silent.

## Architecture & Core Concepts

Planned file layout, in dependency order. A file is listed here once PLAN.md's
corresponding phase is complete.

| File               | Contents                                                          |
| ------------------ | ----------------------------------------------------------------- |
| `doc.go`           | package documentation                                             |
| `version.go`       | `Version`, for a consumer's checkpoint/resume guard               |
| `types.go`         | `Config`, `Result`, `Best`, `TerminationReason`                   |
| `config.go`        | `NewDefaultConfig` and preset factories, Hansen's parameter set   |
| `config_loader.go` | versioned JSON persistence for `Config`                           |
| `eigen.go`         | cyclic Jacobi eigendecomposition for real symmetric matrices      |
| `matrix.go`        | dense matrix helpers used by covariance adaptation                |
| `cmaes.go`         | the strategy itself: sampling, recombination, CSA, C update       |
| `boundary.go`      | box handling; `constraints.go` for Deb rules and penalties        |
| `convergence.go`   | the stopping criteria; `lifecycle.go` for observers and options   |
| `monitoring.go`    | structured lifecycle logging only; the histories live on `Result` |
| `separable.go`     | sep-CMA-ES; `active.go` for negative rank-µ weights               |
| `blockdiag.go`     | block-diagonal covariance                                         |
| `restart.go`       | IPOP and BIPOP                                                    |

**No third-party numerics.** The eigendecomposition is written here, in `eigen.go`,
rather than pulled from gonum. That is a deliberate constraint inherited from the sibling
libraries, and it is why `eigen.go` gets its own phase and its own reconstruction and
orthonormality tests.

## Style

Idiomatic Go, gofumpt formatting. Short math identifiers (`m`, `sigma`, `pc`, `psigma`,
`cmu`) mirror the paper and are exempted from `varnamelen` in `.golangci.toml` — prefer
the paper's symbol over a descriptive rename, and put the expansion in the doc comment.

`Config` is one flat struct with snake_case JSON tags, ordered for `fieldalignment`:
pointers and interfaces, then strings, then float64, then int, then bool.

Every stochastic helper takes `rng *rand.Rand` as its **last** parameter. `Config.Rand`
is the injection point; when it is nil, `OptimizeContext` creates a **run-local**
generator, never writes to `Config`, and records the seed in `Result.Seed` /
`Result.SeedKnown`.

`Config` is treated as read-only for the duration of a run. That is what makes the same
configuration safe to optimize repeatedly, which `OptimizeWithRestarts` and the Phase 9
consumer adapter both depend on.

`omitempty` goes on pointer fields only, where nil means "unset" and is meaningfully
distinct from the zero value. Value fields are always written, so a saved configuration
records every setting explicitly.

## Tests

White-box: test files are `package cmaes` and may exercise unexported helpers.
`example_test.go` is the only black-box file. Table-driven cases for anything with a
published reference value.

Two invariants every change must preserve:

- **Determinism.** A given seed reproduces a run exactly. Any change to the update rules
  breaks this by design — that is what `Version` exists to record.
- **Parallel equivalence.** A seeded run is bit-identical with `EnableParallel` on or
  off. All RNG draws happen on the calling goroutine during a prepare phase; workers
  only evaluate the objective.

Coverage floor is 80%, enforced by `just check-coverage` and by CI.

## Commits and pull requests

Conventional Commits with a scope where it helps (`feat(eigen): cyclic Jacobi solver`,
`fix(cmaes): apply the h-sigma correction to the p_c update`). Keep commits focused. A
pull request should name the PLAN.md task it advances, and tick that box in the same
change.

Never describe a check as passing unless its command was actually run for the revision
being discussed.
