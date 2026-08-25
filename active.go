package cmaes

import "math"

// deriveNegativeWeights returns the active-CMA weights for population ranks
// Mu+1 through Lambda. Their magnitude is limited by Hansen's zero-decay,
// variance-effective-mass, and positive-definiteness guards.
func deriveNegativeWeights(
	config *Config,
	positiveWeights []float64,
	muEff, c1, cmu, weightBase float64,
) []float64 {
	count := config.Lambda - config.Mu
	if count <= 0 || cmu <= 0 {
		return nil
	}

	weights := make([]float64, count)
	sum := 0.0
	squareSum := 0.0

	for index := range weights {
		rank := config.Mu + index + 1

		weight := weightBase - math.Log(float64(rank))
		if weight > 0 {
			weight = 0
		}

		weights[index] = weight
		sum -= weight
		squareSum += weight * weight
	}

	if sum == 0 || squareSum == 0 {
		return nil
	}

	muEffMinus := sum * sum / squareSum
	negativeMass := math.Min(
		1+c1/cmu,
		1+2*muEffMinus/(muEff+2),
	)
	positiveDefiniteMass := (1 - c1 - cmu*sumFloats(positiveWeights)) /
		(float64(config.ProblemSize) * cmu)
	negativeMass = math.Min(negativeMass, positiveDefiniteMass)

	for index := range weights {
		weights[index] *= negativeMass / sum
	}

	return weights
}

func sumFloats(values []float64) float64 {
	var sum float64
	for _, value := range values {
		sum += value
	}

	return sum
}

func covarianceDecay(parameters strategyParameters, hSigma bool) float64 {
	correction := 0.0
	if !hSigma {
		correction = parameters.cc * (2 - parameters.cc)
	}

	weightSum := sumFloats(parameters.weights) + sumFloats(parameters.negativeWeights)

	return 1 - parameters.c1 + parameters.c1*correction - parameters.cmu*weightSum
}

// activeUpdateVector limits a negatively weighted step to squared
// Mahalanobis length n. This makes the negative-weight mass guard sufficient
// to preserve positive-definiteness.
func activeUpdateVector(state *strategyState, step []float64) []float64 {
	whitened := inverseCovarianceSquareRootProduct(state, step)

	squaredNorm := 0.0
	for _, value := range whitened {
		squaredNorm += value * value
	}

	if squaredNorm == 0 {
		return append([]float64(nil), step...)
	}

	scale := math.Sqrt(float64(len(step)) / squaredNorm)
	normalized := make([]float64, len(step))

	for index := range normalized {
		normalized[index] = scale * step[index]
	}

	return normalized
}
