package cmaes

import (
	"math"
	"sort"
)

// applyBoundaryHandling maps sampled genotypes onto positions accepted by the
// objective. Clamp and reflect repair the genotype itself, so covariance
// adaptation observes the repaired step. BoundaryPenalty instead keeps the
// genotype untouched and stores a smoothly transformed phenotype for
// evaluation.
func applyBoundaryHandling(population []candidate, state *strategyState, config *Config) {
	for index := range population {
		current := &population[index]

		switch config.BoundaryMethod {
		case BoundaryClamp:
			clampToBounds(current.x, config.LowerBound, config.UpperBound)
			recomputeStep(current, state)
		case BoundaryReflect:
			reflectIntoBounds(current.x, config.LowerBound, config.UpperBound)
			recomputeStep(current, state)
		case BoundaryPenalty:
			current.evaluationX = make([]float64, len(current.x))

			for coordinate, value := range current.x {
				transformed, reference := transformBounded(
					value,
					config.LowerBound,
					config.UpperBound,
				)
				current.evaluationX[coordinate] = transformed

				distance := (value - reference) / (config.UpperBound - config.LowerBound)
				current.boundaryDistance += distance * distance
			}
		}
	}
}

func clampToBounds(position []float64, lower, upper float64) {
	for index, value := range position {
		position[index] = min(upper, max(lower, value))
	}
}

func reflectIntoBounds(position []float64, lower, upper float64) {
	width := upper - lower
	period := 2 * width

	for index, value := range position {
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
// transformation. It returns both the feasible phenotype and the equivalent
// genotype in the principal domain. The latter is used to penalize remote
// periodic copies without penalizing the quadratic shoulders around a bound.
func transformBounded(value, lower, upper float64) (float64, float64) {
	width := upper - lower
	lowerMargin := min(width/2, max(1, math.Abs(lower))/20)
	upperMargin := min(width/2, max(1, math.Abs(upper))/20)
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

	return transformed, principal
}

func recomputeStep(current *candidate, state *strategyState) {
	for coordinate := range current.y {
		current.y[coordinate] = (current.x[coordinate] - state.m[coordinate]) / state.sigma
	}
}

// assignBoundaryPenalties converts the dimensionless squared distance from a
// remote periodic copy into objective units. The generation's interquartile
// cost range makes the penalty invariant to positive scaling of the objective.
func assignBoundaryPenalties(population []candidate, method BoundaryMethod) {
	if method != BoundaryPenalty {
		return
	}

	finiteCosts := make([]float64, 0, len(population))
	for _, current := range population {
		if isFinite(current.cost) {
			finiteCosts = append(finiteCosts, current.cost)
		}
	}

	sort.Float64s(finiteCosts)

	penaltyScale := 1.0

	if len(finiteCosts) > 1 {
		quartileBase := len(finiteCosts) + 1
		lowerIndex := min(len(finiteCosts)-1, quartileBase/4)
		upperIndex := min(len(finiteCosts)-1, 3*quartileBase/4)

		spread := finiteCosts[upperIndex] - finiteCosts[lowerIndex]
		if isFinite(spread) && spread > 0 {
			penaltyScale = spread
		}
	}

	for index := range population {
		population[index].boundaryPenalty = penaltyScale * population[index].boundaryDistance
	}
}

func (current *candidate) evaluatedPosition() []float64 {
	if current.evaluationX != nil {
		return current.evaluationX
	}

	return current.x
}
