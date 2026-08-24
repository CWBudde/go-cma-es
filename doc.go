// Package cmaes implements the Covariance Matrix Adaptation Evolution Strategy
// (CMA-ES) for bounded continuous minimization.
//
// CMA-ES differs from the swarm metaheuristics in the sibling Mayfly and
// Dragonfly libraries in the one respect that matters on ill-conditioned
// problems: it learns a metric. The sampling distribution is a full multivariate
// normal whose covariance adapts toward the inverse Hessian, so a step that is
// correct along one axis is not forced to be equally large along every other.
// A swarm with an isotropic, externally scheduled step size needs O(cond)
// evaluations on a conditioned ellipsoid where CMA-ES needs O(log cond).
//
// It also controls its own step size by cumulative step-size adaptation, and
// terminates on a set of criteria derived from the distribution's own state
// (TolX, TolFun, condition number, no-effect axes). A run therefore reports that
// it has converged instead of continuing to spend evaluations at zero velocity.
//
// The implementation follows Hansen, N. (2016), "The CMA Evolution Strategy: A
// Tutorial", arXiv:1604.00772, and cross-checks against the reference
// purecmaes.m. Where the tutorial and the reference implementation disagree, the
// tutorial is implemented and the alternative is exposed through Config.
//
// # Status
//
// This package is under construction. PLAN.md in the repository root is the
// single source of truth for what is implemented; do not infer status from this
// comment.
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
