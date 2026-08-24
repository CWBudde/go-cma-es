// Core public types for CMA-ES.

package cmaes

import "math/rand"

// ObjectiveFunction evaluates a position and returns the cost to minimize.
// The position is read-only for the duration of the call.
type ObjectiveFunction func([]float64) float64

// ConstraintFunction evaluates a constraint at a position. Inequality
// constraints are satisfied at values less than or equal to zero. Equality
// constraints are satisfied within ConstraintConfig.EqualityTolerance.
type ConstraintFunction func([]float64) float64

// BoundaryMethod names the rule used to handle samples outside the box.
type BoundaryMethod string

const (
	// BoundaryPenalty applies Hansen's boundary transformation and penalty.
	BoundaryPenalty BoundaryMethod = "penalty"
	// BoundaryClamp pins an out-of-range coordinate to the nearest bound.
	BoundaryClamp BoundaryMethod = "clamp"
	// BoundaryReflect mirrors an out-of-range coordinate back into the box.
	BoundaryReflect BoundaryMethod = "reflect"
)

// CovarianceMode selects the covariance representation used by the strategy.
type CovarianceMode string

const (
	// CovarianceFull learns a dense covariance matrix.
	CovarianceFull CovarianceMode = "full"
	// CovarianceSeparable learns only the diagonal covariance entries.
	CovarianceSeparable CovarianceMode = "separable"
)

// ConstraintHandlingMethod selects how constrained candidates are ranked.
type ConstraintHandlingMethod string

const (
	// ConstraintHandlingFeasibility applies Deb's feasibility rules.
	ConstraintHandlingFeasibility ConstraintHandlingMethod = "feasibility"
	// ConstraintHandlingPenalty ranks candidates by their penalized cost.
	ConstraintHandlingPenalty ConstraintHandlingMethod = "penalty"
)

// PenaltyMethod selects how aggregate constraint violation is penalized.
type PenaltyMethod string

const (
	// PenaltyLinear adds factor times violation to the objective cost.
	PenaltyLinear PenaltyMethod = "linear"
	// PenaltyQuadratic adds factor times squared violation to the objective cost.
	PenaltyQuadratic PenaltyMethod = "quadratic"
)

// ConstraintConfig configures optional nonlinear constraints. Function fields
// cannot be serialized and must be restored after loading a configuration.
type ConstraintConfig struct {
	Handling          ConstraintHandlingMethod `json:"handling,omitempty"`
	PenaltyMethod     PenaltyMethod            `json:"penalty_method,omitempty"`
	Inequalities      []ConstraintFunction     `json:"-"`
	Equalities        []ConstraintFunction     `json:"-"`
	PenaltyFactor     float64                  `json:"penalty_factor,omitempty"`
	EqualityTolerance float64                  `json:"equality_tolerance,omitempty"`
}

// ConvergenceConfig controls optional target-cost and stagnation termination.
// The distribution-derived CMA-ES criteria are implemented in Phase 5.
type ConvergenceConfig struct {
	// TargetCost stops once the best cost is at most this value. Nil disables
	// the target, while a pointer to zero represents an enabled zero target.
	TargetCost *float64 `json:"target_cost,omitempty"`

	// MinImprovement is the absolute improvement needed to reset stagnation.
	MinImprovement float64 `json:"min_improvement"`

	// StagnationIterations is the number of non-improving iterations allowed.
	// Zero disables stagnation detection.
	StagnationIterations int `json:"stagnation_iterations"`

	// MinIterations delays convergence checks until this many iterations have
	// completed. Zero begins checking at the first iteration boundary.
	MinIterations int `json:"min_iterations"`
}

// TerminationReason describes why an optimization run ended.
type TerminationReason string

const (
	// TerminationMaxIterations means the configured iteration cap was reached.
	TerminationMaxIterations TerminationReason = "maximum_iterations"
	// TerminationMaxEvaluations means the configured evaluation cap was reached.
	TerminationMaxEvaluations TerminationReason = "maximum_evaluations"
	// TerminationTargetCost means the configured target cost was reached.
	TerminationTargetCost TerminationReason = "target_cost"
	// TerminationStagnation means the best cost stopped improving sufficiently.
	TerminationStagnation TerminationReason = "stagnation"
	// TerminationTolX means the distribution step fell below TolX.
	TerminationTolX TerminationReason = "tol_x"
	// TerminationTolFun means the recent cost range fell below TolFun.
	TerminationTolFun TerminationReason = "tol_fun"
	// TerminationConditionNumber means the covariance became ill-conditioned.
	TerminationConditionNumber TerminationReason = "condition_number"
	// TerminationNoEffectAxis means movement along a principal axis had no effect.
	TerminationNoEffectAxis TerminationReason = "no_effect_axis"
	// TerminationNoEffectCoord means movement in a coordinate had no effect.
	TerminationNoEffectCoord TerminationReason = "no_effect_coord"
)

// Best is the best position found and its objective and constraint values.
type Best struct {
	Position            []float64
	Cost                float64
	ConstraintViolation float64
}

// Config holds the configuration for a CMA-ES run.
//
// ObjectiveFunc, LowerBound, and UpperBound must be supplied by the caller.
// Every other field has a usable value from one of the constructors.
// ObjectiveFunc and configured constraint callbacks may be called concurrently
// when EnableParallel is true and therefore must be safe for concurrent use.
type Config struct {
	ObjectiveFunc ObjectiveFunction `json:"-"`
	Rand          *rand.Rand        `json:"-"`
	// Seed requests a reproducible run-local random stream. It is mutually
	// exclusive with Rand, whose original seed cannot be recovered.
	Seed        *int64             `json:"seed,omitempty"`
	Convergence *ConvergenceConfig `json:"convergence,omitempty"`
	Constraints *ConstraintConfig  `json:"constraints,omitempty"`

	BoundaryMethod BoundaryMethod `json:"boundary_method"`
	CovarianceMode CovarianceMode `json:"covariance_mode"`
	InitialMean    []float64      `json:"initial_mean"`

	LowerBound   float64 `json:"lower_bound"`
	UpperBound   float64 `json:"upper_bound"`
	InitialSigma float64 `json:"initial_sigma"`

	ProblemSize    int `json:"problem_size"`
	Lambda         int `json:"lambda"`
	Mu             int `json:"mu"`
	MaxIterations  int `json:"max_iterations"`
	MaxEvaluations int `json:"max_evaluations"`
	MaxWorkers     int `json:"max_workers"`

	ActiveCMA      bool `json:"active_cma"`
	EnableParallel bool `json:"enable_parallel"`
}

// Result holds the outcome and accounting information for an optimization.
type Result struct {
	TerminationReason TerminationReason
	GlobalBest        Best
	FuncEvalCount     int
	IterationCount    int
	Seed              int64
	SeedKnown         bool
}
