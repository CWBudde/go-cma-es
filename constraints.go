package cmaes

import "math"

// ConstraintEvaluation describes the aggregate constraint state of a
// position. A zero violation is feasible.
type ConstraintEvaluation struct {
	Violation float64
	Feasible  bool
}

// CandidateEvaluation contains the raw objective and aggregate constraint
// violation used to compare constrained candidates.
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
		return ConstraintEvaluation{Feasible: true}
	}

	violation := 0.0

	for _, constraint := range config.Inequalities {
		if constraint == nil {
			return ConstraintEvaluation{Violation: math.Inf(1)}
		}

		value := constraint(position)
		if !isFinite(value) {
			return ConstraintEvaluation{Violation: math.Inf(1)}
		}

		violation += max(0, value)
	}

	for _, constraint := range config.Equalities {
		if constraint == nil {
			return ConstraintEvaluation{Violation: math.Inf(1)}
		}

		value := constraint(position)
		if !isFinite(value) {
			return ConstraintEvaluation{Violation: math.Inf(1)}
		}

		violation += max(0, math.Abs(value)-config.EqualityTolerance)
	}

	if !isFinite(violation) {
		return ConstraintEvaluation{Violation: math.Inf(1)}
	}

	return ConstraintEvaluation{Violation: violation, Feasible: violation == 0}
}

// IsFeasible reports whether an aggregate constraint violation is zero.
func IsFeasible(violation float64) bool {
	return violation == 0
}

// PenalizedCost applies a linear or quadratic penalty to a raw objective cost.
// An empty method defaults to quadratic.
func PenalizedCost(cost, violation, factor float64, method PenaltyMethod) float64 {
	if method == PenaltyLinear {
		return cost + factor*violation
	}

	return cost + factor*violation*violation
}

// BetterConstrainedCandidate reports whether candidate is preferred over
// incumbent. A nil config uses ordinary minimization. The default constrained
// policy applies Deb's feasibility rules; penalty handling ranks by penalized
// cost and uses Deb's rules to break exact ties.
func BetterConstrainedCandidate(
	candidateEvaluation, incumbentEvaluation CandidateEvaluation,
	config *ConstraintConfig,
) bool {
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

func lessCost(left, right float64) bool {
	if math.IsNaN(left) {
		return false
	}

	return math.IsNaN(right) || left < right
}
