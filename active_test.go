package cmaes

import (
	"math"
	"testing"
)

func TestActiveWeightsUsePublishedMassGuards(t *testing.T) {
	config := NewDefaultConfig(10)
	parameters := deriveStrategyParameters(config)

	positiveMass := sumFloats(parameters.weights)
	negativeMass := -sumFloats(parameters.negativeWeights)

	if math.Abs(positiveMass-1) > 1e-14 {
		t.Fatalf("positive weight mass = %.17g, want 1", positiveMass)
	}

	if len(parameters.negativeWeights) != config.Lambda-config.Mu {
		t.Fatalf("negative weight count = %d, want %d",
			len(parameters.negativeWeights), config.Lambda-config.Mu)
	}

	for index, weight := range parameters.negativeWeights {
		if weight > 0 {
			t.Errorf("negativeWeights[%d] = %g, want <= 0", index, weight)
		}
	}

	muEffMinus := effectiveSelectionMass(parameters.negativeWeights)
	wantMass := math.Min(
		1+parameters.c1/parameters.cmu,
		1+2*muEffMinus/(parameters.muEff+2),
	)
	wantMass = math.Min(
		wantMass,
		(1-parameters.c1-parameters.cmu)/
			(float64(config.ProblemSize)*parameters.cmu),
	)

	if math.Abs(negativeMass-wantMass) > 1e-14 {
		t.Errorf("negative weight mass = %.17g, want %.17g", negativeMass, wantMass)
	}
}

func TestPassiveCMAHasNoNegativeWeights(t *testing.T) {
	config := NewDefaultConfig(10)
	config.ActiveCMA = false

	parameters := deriveStrategyParameters(config)
	if parameters.negativeWeights != nil {
		t.Errorf("passive negative weights = %v, want nil", parameters.negativeWeights)
	}
}

func TestActiveUpdatePreservesPositiveDefiniteness(t *testing.T) {
	config := NewDefaultConfig(8)
	parameters := deriveStrategyParameters(config)
	state := newStrategyState(config)
	population := make([]candidate, config.Lambda)

	for index := range population {
		population[index].y = make([]float64, config.ProblemSize)
		if index >= config.Mu {
			// Align every rejected step with the same axis, the worst case for
			// subtracting variance in a single direction.
			population[index].y[0] = 1e6 * float64(index+1)
		}
	}

	updateStrategyCovariance(state, population, true, parameters)

	eigenvalues, _ := symmetricEigendecomposition(state.c)
	for index, eigenvalue := range eigenvalues {
		if eigenvalue <= 0 || !isFinite(eigenvalue) {
			t.Errorf("eigenvalue[%d] = %g, want finite and positive", index, eigenvalue)
		}
	}
}

func TestActiveUpdateVectorHasDimensionMahalanobisNorm(t *testing.T) {
	config := NewDefaultConfig(3)
	state := newStrategyState(config)
	state.d = []float64{2, 3, 4}
	step := transformNormal(state.b, state.d, []float64{6, -2, 1})

	normalized := activeUpdateVector(state, step)
	whitened := inverseCovarianceSquareRootProduct(state, normalized)

	var squaredNorm float64
	for _, value := range whitened {
		squaredNorm += value * value
	}

	if math.Abs(squaredNorm-float64(config.ProblemSize)) > 1e-14 {
		t.Errorf("squared Mahalanobis norm = %.17g, want %d", squaredNorm, config.ProblemSize)
	}
}

func effectiveSelectionMass(weights []float64) float64 {
	sum := sumFloats(weights)

	var squareSum float64
	for _, weight := range weights {
		squareSum += weight * weight
	}

	return sum * sum / squareSum
}

func TestActiveCMAIsNoWorseThanPassive(t *testing.T) {
	const condition = 1e6

	tests := []struct {
		name       string
		objective  ObjectiveFunction
		iterations int
	}{
		{name: "sphere", objective: sphere, iterations: 300},
		{
			name: "ellipsoid",
			objective: func(position []float64) float64 {
				var cost float64

				for index, value := range position {
					exponent := float64(index) / float64(len(position)-1)
					cost += math.Pow(condition, exponent) * value * value
				}

				return cost
			},
			iterations: 1000,
		},
		{
			name: "rosenbrock",
			objective: func(position []float64) float64 {
				var cost float64

				for index := range len(position) - 1 {
					delta := position[index+1] - position[index]*position[index]
					cost += 100*delta*delta + (1-position[index])*(1-position[index])
				}

				return cost
			},
			iterations: 2000,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			active := optimizationConfig(10, 610, test.objective)
			active.InitialMean = filledVector(10, 2)
			active.InitialSigma = 0.7
			active.MaxIterations = test.iterations
			active.Convergence = nil

			passive := optimizationConfig(10, 610, test.objective)
			passive.InitialMean = filledVector(10, 2)
			passive.InitialSigma = 0.7
			passive.MaxIterations = test.iterations
			passive.Convergence = nil
			passive.ActiveCMA = false

			activeResult, activeErr := Optimize(active)
			if activeErr != nil {
				t.Fatalf("active Optimize: %v", activeErr)
			}

			passiveResult, passiveErr := Optimize(passive)
			if passiveErr != nil {
				t.Fatalf("passive Optimize: %v", passiveErr)
			}

			t.Logf("active=%g passive=%g", activeResult.GlobalBest.Cost, passiveResult.GlobalBest.Cost)
			// Differences below 1e-18 are numerical floor noise on the sphere,
			// not a meaningful optimization regression.
			noWorseLimit := math.Max(passiveResult.GlobalBest.Cost, 1e-18)
			if activeResult.GlobalBest.Cost > noWorseLimit {
				t.Errorf("active cost = %g, worse than passive cost %g",
					activeResult.GlobalBest.Cost, passiveResult.GlobalBest.Cost)
			}
		})
	}
}
