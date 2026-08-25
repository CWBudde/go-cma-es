package cmaes

import "math"

type generationScoreRange struct {
	minimum float64
	maximum float64
}

// convergenceTracker owns the stateful TolFun and stagnation windows. It is
// observed exactly once after each completed distribution update.
type convergenceTracker struct {
	config             *ConvergenceConfig
	constraints        *ConstraintConfig
	recentScores       []generationScoreRange
	referenceBest      CandidateEvaluation
	initialSigma       float64
	stagnantIterations int
	historyLimit       int
}

func newConvergenceTracker(
	config *ConvergenceConfig,
	constraints *ConstraintConfig,
	initialSigma float64,
	dimension, lambda int,
) *convergenceTracker {
	historyLimit := 10
	if lambda > 0 {
		historyLimit += int(math.Ceil(30 * float64(dimension) / float64(lambda)))
	}

	return &convergenceTracker{
		config:        config,
		constraints:   constraints,
		referenceBest: CandidateEvaluation{Cost: math.Inf(1), ConstraintViolation: math.Inf(1)},
		initialSigma:  initialSigma,
		historyLimit:  historyLimit,
	}
}

func (tracker *convergenceTracker) observe(
	iteration int,
	best Best,
	state *strategyState,
	population []candidate,
) (TerminationReason, bool) {
	if tracker.config == nil {
		return "", false
	}

	tracker.observeImprovement(best)
	tracker.observeScores(population)

	if iteration < max(1, tracker.config.MinIterations) {
		return "", false
	}

	if tracker.targetReached(best) {
		return TerminationTargetCost, true
	}

	if tolXReached(state, tracker.config.TolX) {
		return TerminationTolX, true
	}

	if tracker.tolFunReached() {
		return TerminationTolFun, true
	}

	if tolXUpReached(state, tracker.initialSigma, tracker.config.TolXUp) {
		return TerminationTolXUp, true
	}

	if tracker.config.ConditionCov > 0 &&
		covarianceConditionNumber(state.d) > tracker.config.ConditionCov {
		return TerminationConditionNumber, true
	}

	if tracker.config.NoEffectAxis && noEffectAxis(state, iteration) {
		return TerminationNoEffectAxis, true
	}

	if tracker.config.NoEffectCoord && noEffectCoord(state) {
		return TerminationNoEffectCoord, true
	}

	if tracker.config.StagnationIterations > 0 &&
		tracker.stagnantIterations >= tracker.config.StagnationIterations {
		return TerminationStagnation, true
	}

	return "", false
}

func (tracker *convergenceTracker) observeImprovement(best Best) {
	current := CandidateEvaluation{
		Cost:                best.Cost,
		ConstraintViolation: best.ConstraintViolation,
	}

	if tracker.significantlyImproved(current) {
		tracker.referenceBest = current
		tracker.stagnantIterations = 0

		return
	}

	tracker.stagnantIterations++
}

func (tracker *convergenceTracker) significantlyImproved(current CandidateEvaluation) bool {
	if !BetterConstrainedCandidate(current, tracker.referenceBest, tracker.constraints) {
		return false
	}

	if tracker.constraints != nil &&
		effectiveConstraintHandling(tracker.constraints) == ConstraintHandlingPenalty {
		currentScore := PenalizedCost(
			current.Cost,
			current.ConstraintViolation,
			tracker.constraints.PenaltyFactor,
			effectivePenaltyMethod(tracker.constraints),
		)
		referenceScore := PenalizedCost(
			tracker.referenceBest.Cost,
			tracker.referenceBest.ConstraintViolation,
			tracker.constraints.PenaltyFactor,
			effectivePenaltyMethod(tracker.constraints),
		)

		return referenceScore-currentScore > tracker.config.MinImprovement
	}

	currentFeasible := IsFeasible(current.ConstraintViolation)

	referenceFeasible := IsFeasible(tracker.referenceBest.ConstraintViolation)
	if currentFeasible != referenceFeasible {
		return currentFeasible
	}

	if currentFeasible {
		return tracker.referenceBest.Cost-current.Cost > tracker.config.MinImprovement
	}

	return tracker.referenceBest.ConstraintViolation-current.ConstraintViolation >
		tracker.config.MinImprovement
}

func (tracker *convergenceTracker) targetReached(best Best) bool {
	return tracker.config.TargetCost != nil && IsFeasible(best.ConstraintViolation) &&
		best.Cost <= *tracker.config.TargetCost
}

func (tracker *convergenceTracker) observeScores(population []candidate) {
	if tracker.config.TolFun <= 0 || len(population) == 0 {
		return
	}

	current := generationScoreRange{minimum: math.Inf(1), maximum: math.Inf(-1)}

	for _, candidate := range population {
		score := convergenceScore(candidate, tracker.constraints)
		if !isFinite(score) {
			tracker.recentScores = nil

			return
		}

		current.minimum = min(current.minimum, score)
		current.maximum = max(current.maximum, score)
	}

	tracker.recentScores = append(tracker.recentScores, current)
	if len(tracker.recentScores) > tracker.historyLimit {
		copy(tracker.recentScores, tracker.recentScores[len(tracker.recentScores)-tracker.historyLimit:])
		tracker.recentScores = tracker.recentScores[:tracker.historyLimit]
	}
}

func convergenceScore(current candidate, constraints *ConstraintConfig) float64 {
	cost := current.cost + current.boundaryPenalty
	if constraints == nil {
		return cost
	}

	if effectiveConstraintHandling(constraints) == ConstraintHandlingPenalty {
		return PenalizedCost(
			cost,
			current.constraintViolation,
			constraints.PenaltyFactor,
			effectivePenaltyMethod(constraints),
		)
	}

	if IsFeasible(current.constraintViolation) {
		return cost
	}

	return current.constraintViolation
}

func (tracker *convergenceTracker) tolFunReached() bool {
	if tracker.config.TolFun <= 0 || len(tracker.recentScores) < tracker.historyLimit {
		return false
	}

	minimum := math.Inf(1)
	maximum := math.Inf(-1)

	for _, scores := range tracker.recentScores {
		minimum = min(minimum, scores.minimum)
		maximum = max(maximum, scores.maximum)
	}

	return maximum-minimum <= tracker.config.TolFun
}

func tolXReached(state *strategyState, tolerance float64) bool {
	if tolerance <= 0 {
		return false
	}

	maximum := 0.0
	for _, value := range state.pc {
		maximum = max(maximum, math.Abs(value))
	}

	for _, value := range state.d {
		maximum = max(maximum, value)
	}

	return state.sigma*maximum <= tolerance
}

func tolXUpReached(state *strategyState, initialSigma, tolerance float64) bool {
	if tolerance <= 0 {
		return false
	}

	maximum := 0.0
	for _, value := range state.d {
		maximum = max(maximum, value)
	}

	return state.sigma*maximum >= initialSigma*tolerance
}

func covarianceConditionNumber(axisScales []float64) float64 {
	eigenvalues := make([]float64, len(axisScales))
	for index, scale := range axisScales {
		eigenvalues[index] = scale * scale
	}

	return conditionNumber(eigenvalues)
}

func noEffectAxis(state *strategyState, iteration int) bool {
	if len(state.m) == 0 {
		return false
	}

	axis := (iteration - 1) % len(state.m)
	for coordinate, mean := range state.m {
		step := 0.1 * state.sigma * state.d[axis] * state.b[coordinate][axis]
		if mean+step != mean {
			return false
		}
	}

	return true
}

func noEffectCoord(state *strategyState) bool {
	for coordinate, mean := range state.m {
		step := 0.2 * state.sigma * math.Sqrt(max(0, state.c[coordinate][coordinate]))
		if mean+step == mean {
			return true
		}
	}

	return false
}
