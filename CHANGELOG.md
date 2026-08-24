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

### Fixed

- The coverage gate no longer conflates "no statements to cover" with "0% covered";
  `go tool cover -func` reports 0.0% for both.

[Unreleased]: https://github.com/CWBudde/go-cma-es/commits/main
