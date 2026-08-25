package cmaes

import (
	"math"
	"sort"
)

// applyBoundaryHandling maps sampled genotypes onto positions accepted by the
// objective. Clamp and reflect repair the genotype itself and recompute the
// step that reaches it, while candidate.sampledY keeps the step as it was
// drawn: recombination follows the repaired position because that is where the
// objective was evaluated, and covariance adaptation follows the sampled step
// because that is what the distribution actually produced. BoundaryPenalty
// instead keeps the genotype untouched and stores a smoothly transformed
// phenotype for evaluation, recording how far outside the box the genotype
// itself lies.
//
// Every branch resolves the box one coordinate at a time through
// coordinateBounds, because Hansen's transformation is defined per coordinate:
// its shoulder widths al_i and au_i are derived from that coordinate's own
// bounds and are unrelated to any other coordinate's interval.
func applyBoundaryHandling(population []candidate, state *strategyState, config *Config) {
	for index := range population {
		current := &population[index]

		switch config.BoundaryMethod {
		case BoundaryClamp:
			clampToBounds(current.x, config)
			recomputeStep(current, state)
		case BoundaryReflect:
			reflectIntoBounds(current.x, config)
			recomputeStep(current, state)
		case BoundaryPenalty:
			current.evaluationX = make([]float64, len(current.x))

			for coordinate, value := range current.x {
				lower, upper := coordinateBounds(config, coordinate)
				current.evaluationX[coordinate] = transformBounded(value, lower, upper)

				deviation := outOfBoxDeviation(value, lower, upper)
				current.boundaryDistance += deviation * deviation
			}
		}
	}
}

// outOfBoxDeviation returns the displacement Hansen's BoundPenalty squares: how
// far a genotype lies outside the domain its repair maps faithfully onto the
// box, signed, and zero inside it.
//
// For cma.py's BoundPenalty that repair is a clip and the domain is the box
// itself. Here the repair is the linear/quadratic transformation, which is
// injective and onto the box on [lb - al, ub + au] and only there: beyond that
// interval it reflects and then wraps, so the genotype stops describing the
// phenotype it is evaluated at and the distribution would be adapting towards a
// remote periodic copy. That interval is therefore the domain to measure from.
//
// Measuring from the box itself instead would bias every bound-active optimum
// inwards, because the transformation reaches a bound exactly at the end of the
// shoulder, ub + au, which is a point the penalty must leave free of charge.
func outOfBoxDeviation(value, lower, upper float64) float64 {
	lowerMargin, upperMargin := shoulderMargins(lower, upper)

	return value - min(upper+upperMargin, max(lower-lowerMargin, value))
}

func clampToBounds(position []float64, config *Config) {
	for index, value := range position {
		lower, upper := coordinateBounds(config, index)
		position[index] = min(upper, max(lower, value))
	}
}

func reflectIntoBounds(position []float64, config *Config) {
	for index, value := range position {
		lower, upper := coordinateBounds(config, index)
		width := coordinateBoxWidth(config, index)
		period := 2 * width

		offset := math.Mod(value-lower, period)
		if offset < 0 {
			offset += period
		}

		if offset <= width {
			position[index] = lower + offset
		} else {
			position[index] = upper - (offset - width)
		}
	}
}

// transformBounded implements Hansen's smooth linear/quadratic box
// transformation for one coordinate (cma.py's
// BoxConstraintsLinQuadTransformation). A value inside the box away from its
// shoulders is returned unchanged; a value outside is folded back by a periodic
// wrap and up to two reflections, and the two shoulders of width al and au join
// the reflection to the identity with a quadratic piece, so the composite map is
// continuously differentiable everywhere.
func transformBounded(value, lower, upper float64) float64 {
	width := upper - lower
	lowerMargin, upperMargin := shoulderMargins(lower, upper)
	principal := value
	lowerPeriodStart := lower - 2*lowerMargin - width/2
	upperPeriodEnd := upper + 2*upperMargin + width/2

	if principal < lowerPeriodStart || principal > upperPeriodEnd {
		period := 2 * (width + lowerMargin + upperMargin)
		principal -= period * math.Floor((principal-lowerPeriodStart)/period)
	}

	if principal > upper+upperMargin {
		principal -= 2 * (principal - upper - upperMargin)
	}

	if principal < lower-lowerMargin {
		principal += 2 * (lower - lowerMargin - principal)
	}

	transformed := principal
	if transformed < lower+lowerMargin {
		delta := transformed - (lower - lowerMargin)
		transformed = lower + delta*delta/(4*lowerMargin)
	} else if transformed > upper-upperMargin {
		delta := transformed - (upper + upperMargin)
		transformed = upper - delta*delta/(4*upperMargin)
	}

	return transformed
}

// shoulderMargins returns Hansen's al and au, the widths of the quadratic
// shoulders inside the lower and the upper bound. cma.py uses
// min((ub-lb)/2, (1+|b|)/20) for each bound: a fifth of a tenth of the bound's
// own magnitude, offset so that a bound at the origin still gets a shoulder,
// and never wider than half the box. The offset form is the usual
// absolute-plus-relative tolerance idiom and is affine in |b|, which matters
// here because a max(1, |b|) variant would put a kink at the arbitrary
// magnitude |b| = 1 and make the shoulder width depend on the units the search
// space happens to be expressed in.
func shoulderMargins(lower, upper float64) (float64, float64) {
	halfWidth := (upper - lower) / 2

	return min(halfWidth, (1+math.Abs(lower))/20), min(halfWidth, (1+math.Abs(upper))/20)
}

// recomputeStep rewrites the step that leads to the repaired position. It
// allocates rather than writing in place, because current.sampledY aliases the
// step as it was drawn and must survive the repair.
func recomputeStep(current *candidate, state *strategyState) {
	repaired := make([]float64, len(current.y))
	for coordinate := range repaired {
		repaired[coordinate] = (current.x[coordinate] - state.m[coordinate]) / state.sigma
	}

	current.y = repaired
}

// adaptationStep returns the step covariance adaptation learns from: the one
// the distribution sampled, not the one a clamp or a reflection shortened or
// folded back into the box. Rank-mu and the active negative weights both use
// it, so a repaired candidate neither contributes a truncated positive step nor
// has variance subtracted along a direction it never actually explored.
//
// The mean recombination deliberately does not use it, because the mean has to
// move to the positions the objective was evaluated at, and the two evolution
// paths have to keep describing that same displacement.
func (current *candidate) adaptationStep() []float64 {
	if current.sampledY != nil {
		return current.sampledY
	}

	return current.y
}

// boundaryPenaltyState is Hansen's adaptive boundary penalty (cma.py's
// BoundPenalty), which needs to persist across generations and therefore lives
// on the run rather than being recomputed per generation.
//
// The penalty added to a candidate's cost for ranking purposes is
//
//	sum_i gamma_i * outOfBoxDeviation(x_i)^2 / n
//
// where gamma is a per-coordinate weight in objective units per squared
// search-space unit. gamma is seeded from the generation's interquartile
// fitness range divided by the mean coordinate variance sigma^2 * C_ii, which
// is what keeps the penalty commensurate with the fitness differences the
// strategy can actually resolve as the distribution contracts, and what makes
// the whole scheme invariant to positive scaling of the objective. gamma then
// grows multiplicatively, with damping, for as long as the distribution mean
// itself stays more than three standard deviations out of bounds.
type boundaryPenaltyState struct {
	// gamma is the per-coordinate penalty weight.
	gamma []float64
	// history holds recent interquartile fitness ranges, each normalized by
	// the mean coordinate variance of the generation that produced it. gamma
	// is seeded from its median, so a single freak generation cannot set the
	// penalty scale on its own.
	history      []float64
	muEff        float64
	historyLimit int
	initialized  bool
}

// newBoundaryPenaltyState prepares the penalty weights for one run. The history
// limit is Hansen's 20 + 3n/lambda generations.
func newBoundaryPenaltyState(config *Config) *boundaryPenaltyState {
	return &boundaryPenaltyState{
		gamma:        make([]float64, config.ProblemSize),
		muEff:        deriveStrategyParameters(config).muEff,
		historyLimit: 20 + 3*config.ProblemSize/max(1, config.Lambda),
	}
}

// assign adapts the penalty weights to the generation that has just been
// evaluated and then writes each candidate's penalty. The penalty affects
// selection ranking only: it is never folded into a reported cost, and the
// reported position stays the feasible point the objective was evaluated at.
func (penalty *boundaryPenaltyState) assign(
	population []candidate,
	state *strategyState,
	config *Config,
) {
	if config.BoundaryMethod != BoundaryPenalty || len(population) == 0 {
		return
	}

	penalty.update(population, state, config)

	dimension := float64(len(state.m))

	for index := range population {
		current := &population[index]

		var weighted float64

		for coordinate, value := range current.x {
			lower, upper := coordinateBounds(config, coordinate)
			deviation := outOfBoxDeviation(value, lower, upper)
			weighted += penalty.gamma[coordinate] * deviation * deviation
		}

		current.boundaryPenalty = weighted / dimension
	}
}

// update performs cma.py's BoundPenalty.update for one generation: it records
// the normalized fitness spread, seeds gamma the first time the distribution
// mean itself goes out of bounds, and then increases the weight of every
// coordinate whose mean is more than three standard deviations out.
func (penalty *boundaryPenaltyState) update(
	population []candidate,
	state *strategyState,
	config *Config,
) {
	variances := coordinateVariances(state)
	penalty.observeSpread(population, meanValue(variances))

	// A scale for the penalty can only be read off an observed spread of the
	// objective. A flat, a single-candidate or an all-infinite generation
	// offers none, so gamma keeps whatever it learned earlier - and, before it
	// has learned anything, stays zero. Both are invariant under positive
	// scaling of the objective, which a constant in raw objective units would
	// not be.
	deltaFitness := medianValue(penalty.history)
	if deltaFitness <= 0 {
		return
	}

	deviation := meanBoundaryDeviation(state, config, variances)

	if !penalty.initialized {
		if !anyNonZero(deviation) {
			return
		}

		for coordinate := range penalty.gamma {
			penalty.gamma[coordinate] = 2 * deltaFitness
		}

		penalty.initialized = true
	}

	dimension := float64(len(state.m))
	damping := min(1, penalty.muEff/(10*dimension))
	threshold := 3 * max(1, math.Sqrt(dimension)/penalty.muEff)

	for coordinate, value := range deviation {
		excess := math.Abs(value) - threshold
		if excess <= 0 {
			continue
		}

		penalty.gamma[coordinate] *= math.Pow(math.Exp(math.Tanh(excess/3)/2), damping)
	}
}

// observeSpread appends this generation's interquartile fitness range, divided
// by the mean coordinate variance, to the history. Dividing by the variance is
// what lets a spread measured while the distribution was wide still describe
// the penalty once it has contracted, because both shrink together.
func (penalty *boundaryPenaltyState) observeSpread(population []candidate, meanVariance float64) {
	if meanVariance <= 0 || !isFinite(meanVariance) {
		return
	}

	costs := make([]float64, 0, len(population))

	for _, current := range population {
		if isFinite(current.cost) {
			costs = append(costs, current.cost)
		}
	}

	if len(costs) < 2 {
		return
	}

	sort.Float64s(costs)

	quartileBase := len(costs) + 1
	lowerIndex := min(len(costs)-1, quartileBase/4)
	upperIndex := min(len(costs)-1, 3*quartileBase/4)

	spread := (costs[upperIndex] - costs[lowerIndex]) / meanVariance
	if !isFinite(spread) || spread <= 0 {
		return
	}

	penalty.history = append(penalty.history, spread)
	if len(penalty.history) > penalty.historyLimit {
		penalty.history = penalty.history[1:]
	}
}

// coordinateVariances returns sigma^2 * C_ii, the variance the distribution
// currently assigns to each coordinate.
func coordinateVariances(state *strategyState) []float64 {
	variances := make([]float64, len(state.m))

	for coordinate := range variances {
		diagonal := 1.0

		switch {
		case state.mode == CovarianceSeparable:
			diagonal = state.diagonal[coordinate]
		case state.c != nil:
			diagonal = state.c[coordinate][coordinate]
		}

		variances[coordinate] = state.sigma * state.sigma * diagonal
	}

	return variances
}

// meanBoundaryDeviation returns how far the distribution mean lies out of
// bounds in each coordinate, measured in that coordinate's standard deviations.
// It is the term that makes the penalty respond to the mean itself drifting out
// rather than only to individual samples doing so.
func meanBoundaryDeviation(state *strategyState, config *Config, variances []float64) []float64 {
	deviation := make([]float64, len(state.m))

	for coordinate, value := range state.m {
		if variances[coordinate] <= 0 {
			continue
		}

		lower, upper := coordinateBounds(config, coordinate)
		deviation[coordinate] = outOfBoxDeviation(value, lower, upper) /
			math.Sqrt(variances[coordinate])
	}

	return deviation
}

func anyNonZero(values []float64) bool {
	for _, value := range values {
		if value != 0 {
			return true
		}
	}

	return false
}

func meanValue(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	var sum float64
	for _, value := range values {
		sum += value
	}

	return sum / float64(len(values))
}

func medianValue(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)

	middle := len(sorted) / 2
	if len(sorted)%2 == 1 {
		return sorted[middle]
	}

	return (sorted[middle-1] + sorted[middle]) / 2
}

func (current *candidate) evaluatedPosition() []float64 {
	if current.evaluationX != nil {
		return current.evaluationX
	}

	return current.x
}
