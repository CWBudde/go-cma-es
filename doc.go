// Package cmaes implements the Covariance Matrix Adaptation Evolution Strategy
// (CMA-ES) for continuous minimization.
//
// CMA-ES differs from the swarm metaheuristics in the sibling Mayfly and
// Dragonfly libraries in the one respect that matters on ill-conditioned
// problems: it learns a metric. The default sampling distribution is a full
// multivariate normal whose covariance adapts toward the inverse Hessian, so a
// step that is correct along one axis is not forced to be equally large along
// every other. Active covariance adaptation is enabled by default. Separable
// mode learns coordinate scaling with linear time and storage when correlations
// are unnecessary.
// A swarm with an isotropic, externally scheduled step size needs O(cond)
// evaluations on a conditioned ellipsoid where CMA-ES needs O(log cond).
//
// It also controls its own step size by cumulative step-size adaptation. TolX,
// TolFun, TolXUp, condition-number, and no-effect criteria stop a converged or
// numerically exhausted distribution without spending the rest of its budget.
//
// The implementation follows Hansen, N. (2016), "The CMA Evolution Strategy: A
// Tutorial", arXiv:1604.00772, and cross-checks against the reference
// purecmaes.m. Where the tutorial and the reference implementation disagree, the
// tutorial is implemented and the alternative is exposed through Config.
//
// # Status
//
// This package is under construction. Active full-covariance and separable
// strategies, box-boundary handling, nonlinear constraints, convergence
// criteria, lifecycle observers, and the WebAssembly showcase are implemented;
// PLAN.md in the repository root is the single source of truth for later
// variant work.
//
// # Conventions
//
// The package is dependency-free: the only direct requirement is
// github.com/cucumber/godog, and it is test-only. Every stochastic helper takes
// its *math/rand.Rand as the final parameter, Config.Rand is the injection
// point, and a seeded run is bit-identical whether or not parallel evaluation is
// enabled. These conventions are shared with the sibling Mayfly and Dragonfly
// libraries so that code reads the same across all three.
package cmaes
