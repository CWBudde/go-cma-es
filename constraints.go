package cmaes

import "math"

// ConstraintEvaluation describes the aggregate constraint state of a
// position. Violation is non-negative; a zero violation is feasible.
type ConstraintEvaluation struct {
	Violation float64
	Feasible  bool
}

// CandidateEvaluation contains the raw objective and aggregate constraint
// violation used to compare constrained candidates. ConstraintViolation is
// expected to be non-negative; BetterConstrainedCandidate sanitizes any other
// value rather than producing an inconsistent order.
type CandidateEvaluation struct {
	Cost                float64
	ConstraintViolation float64
}

// EvaluateConstraints evaluates and aggregates all configured constraints.
// Inequalities contribute max(0, g(x)); equalities contribute
// max(0, |h(x)|-tolerance). Nil or non-finite callbacks produce an infinite
// violation so an unusable candidate loses every comparison.
func EvaluateConstraints(position []float64, config *ConstraintConfig) ConstraintEvaluation {
	if config == nil {
		return ConstraintEvaluation{Violation: 0, Feasible: true}
	}

	violation := 0.0

	for _, constraint := range config.Inequalities {
		if constraint == nil {
			return ConstraintEvaluation{Violation: math.Inf(1), Feasible: false}
		}

		value := constraint(position)
		if !isFinite(value) {
			return ConstraintEvaluation{Violation: math.Inf(1), Feasible: false}
		}

		violation += max(0, value)
	}

	for _, constraint := range config.Equalities {
		if constraint == nil {
			return ConstraintEvaluation{Violation: math.Inf(1), Feasible: false}
		}

		value := constraint(position)
		if !isFinite(value) {
			return ConstraintEvaluation{Violation: math.Inf(1), Feasible: false}
		}

		violation += max(0, math.Abs(value)-config.EqualityTolerance)
	}

	if !isFinite(violation) {
		return ConstraintEvaluation{Violation: math.Inf(1), Feasible: false}
	}

	return ConstraintEvaluation{Violation: violation, Feasible: violation == 0}
}

// IsFeasible reports whether an aggregate constraint violation describes a
// feasible position. Aggregation clamps every term with max(0, ...), so a
// violation produced by EvaluateConstraints is never negative; a negative value
// supplied by a caller is treated as no violation and therefore feasible. NaN
// is infeasible, since an unusable candidate must lose every comparison.
func IsFeasible(violation float64) bool {
	return !math.IsNaN(violation) && violation <= 0
}

// PenalizedCost applies a linear or quadratic penalty to a raw objective cost.
// PenaltyLinear adds factor*violation; every other method — PenaltyQuadratic,
// the empty string and any unrecognized value alike — adds
// factor*violation*violation. Configurations are screened by validation, so an
// unrecognized method reaches this function only through a direct call.
func PenalizedCost(cost, violation, factor float64, method PenaltyMethod) float64 {
	if method == PenaltyLinear {
		return cost + factor*violation
	}

	return cost + factor*violation*violation
}

// BetterConstrainedCandidate reports whether candidate is preferred over
// incumbent. A nil config uses ordinary minimization. The default constrained
// policy applies Deb's feasibility rules: a feasible candidate beats an
// infeasible one, two infeasible candidates compare by violation, and equally
// feasible or equally violating candidates compare by cost.
//
// Penalty handling ranks solely by penalized cost and only falls back to Deb's
// rules when the two penalized scores are exactly equal as float64, which in
// practice is rare. It is therefore no safety net: with a large enough cost
// range an infeasible candidate outranks a feasible one, which is inherent to
// penalty methods and is what choosing ConstraintHandlingPenalty asks for.
// Pick the penalty factor accordingly, or use the default feasibility rules.
//
// Violations are sanitized before comparison — NaN becomes +Inf and negative
// values become zero — so the result is a strict weak ordering for every input
// and is safe to hand to sort.SliceStable. A NaN cost sorts after every real
// cost, and two NaN costs compare as equivalent.
func BetterConstrainedCandidate(
	candidateEvaluation, incumbentEvaluation CandidateEvaluation,
	config *ConstraintConfig,
) bool {
	candidateEvaluation.ConstraintViolation = sanitizedViolation(
		candidateEvaluation.ConstraintViolation,
	)
	incumbentEvaluation.ConstraintViolation = sanitizedViolation(
		incumbentEvaluation.ConstraintViolation,
	)

	if config == nil {
		return lessCost(candidateEvaluation.Cost, incumbentEvaluation.Cost)
	}

	if effectiveConstraintHandling(config) == ConstraintHandlingPenalty {
		candidateScore := PenalizedCost(
			candidateEvaluation.Cost,
			candidateEvaluation.ConstraintViolation,
			config.PenaltyFactor,
			effectivePenaltyMethod(config),
		)
		incumbentScore := PenalizedCost(
			incumbentEvaluation.Cost,
			incumbentEvaluation.ConstraintViolation,
			config.PenaltyFactor,
			effectivePenaltyMethod(config),
		)

		if candidateScore != incumbentScore {
			return lessCost(candidateScore, incumbentScore)
		}
	}

	candidateFeasible := IsFeasible(candidateEvaluation.ConstraintViolation)
	incumbentFeasible := IsFeasible(incumbentEvaluation.ConstraintViolation)

	if candidateFeasible != incumbentFeasible {
		return candidateFeasible
	}

	if !candidateFeasible &&
		candidateEvaluation.ConstraintViolation != incumbentEvaluation.ConstraintViolation {
		return candidateEvaluation.ConstraintViolation < incumbentEvaluation.ConstraintViolation
	}

	return lessCost(candidateEvaluation.Cost, incumbentEvaluation.Cost)
}

func effectiveConstraintHandling(config *ConstraintConfig) ConstraintHandlingMethod {
	if config != nil && config.Handling == ConstraintHandlingPenalty {
		return ConstraintHandlingPenalty
	}

	return ConstraintHandlingFeasibility
}

func effectivePenaltyMethod(config *ConstraintConfig) PenaltyMethod {
	if config != nil && config.PenaltyMethod == PenaltyLinear {
		return PenaltyLinear
	}

	return PenaltyQuadratic
}

// sanitizedViolation maps an arbitrary caller-supplied violation onto the
// non-negative range EvaluateConstraints produces: NaN becomes +Inf so an
// unusable candidate loses every comparison, and a negative value becomes zero
// so it is ranked as feasible. Without this, NaN would be incomparable with
// everything and the resulting order would not be a strict weak ordering.
func sanitizedViolation(violation float64) float64 {
	if math.IsNaN(violation) {
		return math.Inf(1)
	}

	return max(0, violation)
}

// lessCost orders costs with NaN as the largest value, so that NaN costs form a
// single equivalence class at the end of the order rather than being
// incomparable.
func lessCost(left, right float64) bool {
	if math.IsNaN(left) {
		return false
	}

	return math.IsNaN(right) || left < right
}
