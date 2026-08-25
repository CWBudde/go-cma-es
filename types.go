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
	// BoundaryPenalty evaluates the objective at Hansen's linear/quadratic
	// transformation of the sample while adapting the distribution from the
	// untransformed one, and adds an adaptive per-coordinate penalty to the
	// selection ranking. The penalty weights are scaled by each coordinate's
	// variance and grow while the distribution mean stays out of bounds; they
	// affect ranking only, never a reported cost or position.
	//
	// This is the default because it is the only method that leaves the
	// sampled step untouched. Repairing a genotype biases the covariance
	// estimate, which is the failure mode this library exists to avoid.
	BoundaryPenalty BoundaryMethod = "penalty"
	// BoundaryClamp pins an out-of-range coordinate to the nearest bound.
	// The repaired position is what the objective sees and what the mean
	// recombines, but covariance adaptation still learns from the step that
	// was sampled, so repeatedly pinning samples to a bound does not bias the
	// learned metric.
	BoundaryClamp BoundaryMethod = "clamp"
	// BoundaryReflect mirrors an out-of-range coordinate back into the box,
	// folding repeatedly for a position more than one box width outside. It
	// shares BoundaryClamp's split between the repaired position and the
	// sampled step.
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
// cannot be serialized and must be restored after loading a configuration;
// Validate rejects a configured ConstraintConfig that carries no functions so
// that a round-tripped configuration cannot silently drop its constraints.
//
// The JSON tags follow the package-wide rule stated on Config: only pointer
// fields carry omitempty, so a saved configuration is a complete record.
type ConstraintConfig struct {
	// Handling selects the ranking rule. An empty value means feasibility.
	Handling ConstraintHandlingMethod `json:"handling"`

	// PenaltyMethod selects how the aggregate violation enters the cost. It is
	// used only with penalty handling and is ignored (not rejected) when
	// Handling is feasibility, including the empty value that means feasibility.
	PenaltyMethod PenaltyMethod `json:"penalty_method"`

	Inequalities []ConstraintFunction `json:"-"`
	Equalities   []ConstraintFunction `json:"-"`

	// PenaltyFactor scales the violation term. It must be positive with penalty
	// handling and is ignored with feasibility handling.
	PenaltyFactor float64 `json:"penalty_factor"`

	// EqualityTolerance is the absolute slack within which an equality
	// constraint counts as satisfied.
	EqualityTolerance float64 `json:"equality_tolerance"`
}

// ConvergenceConfig controls early termination. A zero numeric tolerance and
// a false no-effect flag disable the corresponding criterion.
type ConvergenceConfig struct {
	// TargetCost stops once the best cost is at most this value. Nil disables
	// the target, while a pointer to zero represents an enabled zero target.
	TargetCost *float64 `json:"target_cost,omitempty"`

	// TolX stops when the distribution has shrunk below this multiple of the
	// run's initial sigma, matching Hansen's TolX = 1e-12 * sigma^(0). The
	// criterion compares sigma * max(max(D), max_i |p_c,i|) against
	// TolX * sigma^(0): it uses the largest distribution axis max(D) and the
	// covariance evolution path, not the per-coordinate sqrt(C_ii).
	TolX float64 `json:"tol_x"`

	// TolFun stops when the range of recent population scores falls below this
	// absolute tolerance.
	TolFun float64 `json:"tol_fun"`

	// TolXUp stops when sigma * max(D) grows beyond this multiple of its own
	// initial value sigma^(0) * max(D^(0)), which signals a diverging run.
	TolXUp float64 `json:"tol_x_up"`

	// ConditionCov stops when the covariance matrix's spectral condition
	// number exceeds this value.
	ConditionCov float64 `json:"condition_cov"`

	// MinImprovement is the absolute improvement needed to reset stagnation.
	MinImprovement float64 `json:"min_improvement"`

	// StagnationIterations is the number of non-improving iterations allowed.
	// Zero disables stagnation detection.
	StagnationIterations int `json:"stagnation_iterations"`

	// MinIterations delays convergence checks until this many iterations have
	// completed. Zero begins checking at the first iteration boundary.
	MinIterations int `json:"min_iterations"`

	// NoEffectAxis enables detection of principal-axis steps that no longer
	// change the floating-point mean.
	NoEffectAxis bool `json:"no_effect_axis"`

	// NoEffectCoord enables detection of coordinate steps that no longer
	// change the floating-point mean.
	NoEffectCoord bool `json:"no_effect_coord"`
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
	// TerminationTolXUp means the distribution step grew beyond TolXUp.
	TerminationTolXUp TerminationReason = "tol_x_up"
	// TerminationConditionNumber means the covariance became ill-conditioned.
	TerminationConditionNumber TerminationReason = "condition_number"
	// TerminationNoEffectAxis means movement along a principal axis had no effect.
	TerminationNoEffectAxis TerminationReason = "no_effect_axis"
	// TerminationNoEffectCoord means movement in a coordinate had no effect.
	TerminationNoEffectCoord TerminationReason = "no_effect_coord"
	// TerminationCancelled means the context canceled an in-progress run.
	TerminationCancelled TerminationReason = "cancelled" //nolint:misspell // Public value follows PLAN.md.
)

// Best is the best position found and its objective and constraint values.
type Best struct {
	Position            []float64 `json:"position"`
	Cost                float64   `json:"cost"`
	ConstraintViolation float64   `json:"constraint_violation"`
}

// Config holds the configuration for a CMA-ES run.
//
// ObjectiveFunc and the search box must be supplied by the caller. Every other
// field has a usable value from one of the constructors. ObjectiveFunc and
// configured constraint callbacks may be called concurrently when
// EnableParallel is true and therefore must be safe for concurrent use.
//
// JSON tags are snake_case throughout the package, and omitempty is used on
// pointer fields only, where nil means "unset" and is distinguishable from the
// zero value. Every other field is always written, so a saved configuration is
// a complete record of the run rather than a diff against the defaults.
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

	// LowerBounds and UpperBounds give each coordinate its own search interval.
	// A nil slice means the corresponding scalar bound broadcasts to every
	// coordinate; a non-nil slice must have length ProblemSize and takes
	// precedence over the scalar for that side. The two sides are independent,
	// so a per-dimension lower bound may be combined with a scalar upper bound.
	// Use coordinateBounds rather than reading these fields directly.
	LowerBounds []float64 `json:"lower_bounds"`
	UpperBounds []float64 `json:"upper_bounds"`

	// LowerBound and UpperBound describe a hypercube search box. They are
	// ignored for a side that supplies a per-dimension slice.
	LowerBound   float64 `json:"lower_bound"`
	UpperBound   float64 `json:"upper_bound"`
	InitialSigma float64 `json:"initial_sigma"`

	ProblemSize   int `json:"problem_size"`
	Lambda        int `json:"lambda"`
	Mu            int `json:"mu"`
	MaxIterations int `json:"max_iterations"`

	// MaxEvaluations caps objective evaluations. Zero means no cap; a positive
	// value must be at least Lambda, since a run cannot complete a generation
	// on a smaller budget.
	MaxEvaluations int `json:"max_evaluations"`

	// MaxWorkers caps the goroutines used when EnableParallel is true. Zero
	// means runtime.NumCPU(); it is what the constructors record explicitly.
	MaxWorkers int `json:"max_workers"`

	ActiveCMA      bool `json:"active_cma"`
	EnableParallel bool `json:"enable_parallel"`
}

// Result holds the outcome and accounting information for an optimization.
//
// Every history slice contains one entry per completed iteration, so entries at
// the same index describe the same iteration.
type Result struct {
	// ConvergenceCurve records the running global best cost, which is monotone
	// non-increasing by construction. Use IterationBestHistory to see how the
	// population itself fluctuates.
	ConvergenceCurve []float64 `json:"convergence_curve"`

	// IterationBestHistory records the best cost within the population sampled
	// in each iteration. Unlike ConvergenceCurve it is not monotone: it rises
	// again whenever a generation is worse than the incumbent, which is what
	// makes fitness oscillation visible next to SigmaHistory.
	IterationBestHistory []float64 `json:"iteration_best_history"`

	SigmaHistory           []float64 `json:"sigma_history"`
	ConditionNumberHistory []float64 `json:"condition_number_history"`

	TerminationReason TerminationReason `json:"termination_reason"`
	GlobalBest        Best              `json:"global_best"`
	FuncEvalCount     int               `json:"func_eval_count"`
	IterationCount    int               `json:"iteration_count"`
	Seed              int64             `json:"seed"`

	// SeedKnown is false when the caller injected its own Rand, whose seed
	// cannot be recovered; Seed is then meaningless.
	SeedKnown bool `json:"seed_known"`
}
