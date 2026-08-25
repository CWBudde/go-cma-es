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
	recentBest         []float64
	referenceBest      CandidateEvaluation
	currentScores      generationScoreRange
	initialSigma       float64
	initialAxisScale   float64
	stagnantIterations int
	historyLimit       int
	hasCurrentScores   bool
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

	tracker.captureInitialAxisScale(state)
	tracker.observeImprovement(best)
	tracker.observeScores(population)

	if iteration < max(1, tracker.config.MinIterations) {
		return "", false
	}

	if tracker.targetReached(best) {
		return TerminationTargetCost, true
	}

	if tolXReached(state, tracker.initialSigma, tracker.config.TolX) {
		return TerminationTolX, true
	}

	if tracker.tolFunReached() {
		return TerminationTolFun, true
	}

	if tolXUpReached(state, tracker.initialSigma, tracker.initialAxisScale,
		tracker.config.TolXUp) {
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

// captureInitialAxisScale records max(D) once, from the first state the
// tracker sees. TolXUp is published as "sigma * max(D) grew by more than
// TolXUp over its initial value" (Hansen, tutorial App. B); comparing against
// sigma0 alone is only the same statement when D starts at the identity. The
// tracker is built before any state exists, so the reference is taken at the
// first observation -- one distribution update in -- which keeps the criterion
// meaningful for a warm start or a variant seeding a non-identity C.
func (tracker *convergenceTracker) captureInitialAxisScale(state *strategyState) {
	if tracker.initialAxisScale > 0 || state == nil {
		return
	}

	scale := maxAxisScale(state.d)
	if scale > 0 && isFinite(scale) {
		tracker.initialAxisScale = scale
	}
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

// observeScores feeds Hansen's TolFun window. The criterion (tutorial App. B,
// cma.py "tolfun") ranges over the best value of each of the last
// historyLimit generations, unioned with every value of the *current*
// generation -- so the window keeps a best-of-generation series, not each
// generation's full spread.
//
// A non-finite score is skipped rather than allowed to reset the window: one
// NaN evaluation late in a run must not cost the whole accumulated history. A
// generation in which no candidate scored finitely is skipped whole, leaving
// both the series and the previous generation's spread untouched.
func (tracker *convergenceTracker) observeScores(population []candidate) {
	if tracker.config.TolFun <= 0 || len(population) == 0 {
		return
	}

	current := generationScoreRange{minimum: math.Inf(1), maximum: math.Inf(-1)}
	finite := false

	for _, candidate := range population {
		score := convergenceScore(candidate, tracker.constraints)
		if !isFinite(score) {
			continue
		}

		finite = true
		current.minimum = min(current.minimum, score)
		current.maximum = max(current.maximum, score)
	}

	if !finite {
		return
	}

	tracker.currentScores = current
	tracker.hasCurrentScores = true

	tracker.recentBest = append(tracker.recentBest, current.minimum)
	if len(tracker.recentBest) > tracker.historyLimit {
		copy(tracker.recentBest, tracker.recentBest[len(tracker.recentBest)-tracker.historyLimit:])
		tracker.recentBest = tracker.recentBest[:tracker.historyLimit]
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

// tolFunReached reports whether the range of the best-of-generation series
// over the last historyLimit generations, unioned with the current
// generation's full spread, has fallen to TolFun or below.
func (tracker *convergenceTracker) tolFunReached() bool {
	if tracker.config.TolFun <= 0 || !tracker.hasCurrentScores ||
		len(tracker.recentBest) < tracker.historyLimit {
		return false
	}

	minimum := tracker.currentScores.minimum
	maximum := tracker.currentScores.maximum

	for _, score := range tracker.recentBest {
		minimum = min(minimum, score)
		maximum = max(maximum, score)
	}

	return maximum-minimum <= tracker.config.TolFun
}

// tolXReached reports whether the distribution has shrunk below TolX, which
// Hansen defines relative to the initial step size (default 1e-12 * sigma^(0)).
// Following purecma it uses max(D) rather than the per-coordinate sqrt(C_ii),
// which is the conservative simplification: max(D) >= max_i sqrt(C_ii).
func tolXReached(state *strategyState, initialSigma, tolerance float64) bool {
	if tolerance <= 0 {
		return false
	}

	maximum := 0.0
	for _, value := range state.pc {
		maximum = max(maximum, math.Abs(value))
	}

	maximum = max(maximum, maxAxisScale(state.d))

	return state.sigma*maximum <= initialSigma*tolerance
}

// tolXUpReached reports whether sigma * max(D) has grown by more than TolXUp
// over its initial value. initialAxisScale is max(D^(0)); a non-positive value
// means no usable initial state was seen and the identity is assumed.
func tolXUpReached(state *strategyState, initialSigma, initialAxisScale, tolerance float64) bool {
	if tolerance <= 0 {
		return false
	}

	reference := initialAxisScale
	if reference <= 0 {
		reference = 1
	}

	return state.sigma*maxAxisScale(state.d) >= initialSigma*reference*tolerance
}

func maxAxisScale(axisScales []float64) float64 {
	maximum := 0.0
	for _, value := range axisScales {
		maximum = max(maximum, value)
	}

	return maximum
}

// covarianceConditionNumber returns cond(C) from the axis scales in D. D holds
// standard deviations, so the condition number is max(d)^2 / min(d)^2. The
// ratio is squared rather than the extremes, which avoids both an intermediate
// slice and an overflow on a wide spectrum.
func covarianceConditionNumber(axisScales []float64) float64 {
	if len(axisScales) == 0 {
		return 0
	}

	minimum := math.Inf(1)
	maximum := 0.0

	for _, scale := range axisScales {
		if !isFinite(scale) || scale < 0 {
			return math.Inf(1)
		}

		minimum = math.Min(minimum, scale)
		maximum = math.Max(maximum, scale)
	}

	if minimum == 0 {
		return math.Inf(1)
	}

	ratio := maximum / minimum

	return ratio * ratio
}

func noEffectAxis(state *strategyState, iteration int) bool {
	if len(state.m) == 0 {
		return false
	}

	axis := (iteration - 1) % len(state.m)
	switch state.mode {
	case CovarianceSeparable:
		step := 0.1 * state.sigma * state.d[axis]

		return state.m[axis]+step == state.m[axis]
	case CovarianceBlock:
		return !blockAxisStepHasEffect(state, axis)
	}

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
		variance := state.diagonalVariance(coordinate)
		step := 0.2 * state.sigma * math.Sqrt(max(0, variance))

		if mean+step == mean {
			return true
		}
	}

	return false
}

func (state *strategyState) diagonalVariance(coordinate int) float64 {
	switch state.mode {
	case CovarianceSeparable:
		return state.diagonal[coordinate]
	case CovarianceBlock:
		block := state.coordinateBlock[coordinate]
		local := state.coordinateLocal[coordinate]

		return state.blockC[block][local][local]
	default:
		if state.c == nil {
			return 1
		}

		return state.c[coordinate][coordinate]
	}
}
