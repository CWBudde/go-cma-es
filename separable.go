package cmaes

import "math"

func newSeparableStrategyState(config *Config) *strategyState {
	diagonal := make([]float64, config.ProblemSize)
	d := make([]float64, config.ProblemSize)

	for index := range diagonal {
		diagonal[index] = 1
		d[index] = 1
	}

	return &strategyState{
		m:        append([]float64(nil), config.InitialMean...),
		psigma:   make([]float64, config.ProblemSize),
		pc:       make([]float64, config.ProblemSize),
		diagonal: diagonal,
		d:        d,
		sigma:    config.InitialSigma,
		mode:     CovarianceSeparable,
	}
}

func transformSeparableNormal(d, z []float64) []float64 {
	transformed := make([]float64, len(z))
	for index := range transformed {
		transformed[index] = d[index] * z[index]
	}

	return transformed
}

func inverseSeparableSquareRootProduct(d, step []float64) []float64 {
	product := make([]float64, len(step))
	for index := range product {
		if d[index] > 0 {
			product[index] = step[index] / d[index]
		}
	}

	return product
}

func updateSeparableCovariance(
	state *strategyState,
	population []candidate,
	hSigma bool,
	parameters strategyParameters,
) {
	decay := covarianceDecay(parameters, hSigma)

	negativeSteps := make([][]float64, len(parameters.negativeWeights))
	for index := range negativeSteps {
		populationIndex := len(parameters.weights) + index
		negativeSteps[index] = activeUpdateVector(state, population[populationIndex].y)
	}

	for coordinate := range state.diagonal {
		value := decay*state.diagonal[coordinate] +
			parameters.c1*state.pc[coordinate]*state.pc[coordinate]

		for index, weight := range parameters.weights {
			step := population[index].y[coordinate]
			value += parameters.cmu * weight * step * step
		}

		for index, weight := range parameters.negativeWeights {
			step := negativeSteps[index][coordinate]
			value += parameters.cmu * weight * step * step
		}

		state.diagonal[coordinate] = math.Max(math.SmallestNonzeroFloat64, value)
		state.d[coordinate] = math.Sqrt(state.diagonal[coordinate])
	}
}

func separableEigenvectors(size int) [][]float64 {
	return identityMatrix(size)
}
