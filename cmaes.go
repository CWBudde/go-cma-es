package cmaes

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"time"
)

// RunOption customizes one optimization run. Phase 5 adds the lifecycle
// option constructors; the opaque type is defined now because it is part of
// the OptimizeContext signature.
type RunOption struct {
	apply func(*runOptions) error
}

type runOptions struct{}

// strategyState is the mutable full-covariance CMA-ES distribution.
// Eigenvectors are stored as the columns of b and d contains their standard
// deviations, so c = b * diag(d^2) * b^T.
type strategyState struct {
	m         []float64
	psigma    []float64
	pc        []float64
	c         [][]float64
	b         [][]float64
	d         []float64
	sigma     float64
	eigenEval int
}

type candidate struct {
	x    []float64
	y    []float64
	z    []float64
	cost float64
}

// Optimize runs full-covariance CMA-ES with a background context.
func Optimize(config *Config) (*Result, error) {
	return OptimizeContext(context.Background(), config)
}

// OptimizeContext runs full-covariance CMA-ES, honoring context cancellation.
// Random samples are prepared on the calling goroutine before any objective
// evaluations begin, making seeded serial and parallel runs bit-identical.
func OptimizeContext(ctx context.Context, config *Config, options ...RunOption) (*Result, error) {
	if ctx == nil {
		return nil, errors.New("context must not be nil")
	}

	ctxErr := ctx.Err()
	if ctxErr != nil {
		return nil, ctxErr
	}

	optionsErr := resolveRunOptions(options)
	if optionsErr != nil {
		return nil, optionsErr
	}

	validationErr := config.Validate()
	if validationErr != nil {
		return nil, validationErr
	}

	if config.CovarianceMode != CovarianceFull {
		return nil, fmt.Errorf("covariance mode %q is not implemented yet", config.CovarianceMode)
	}

	if config.Constraints != nil {
		return nil, errors.New("constraints are not implemented yet")
	}

	if config.Convergence != nil {
		return nil, errors.New("convergence criteria are not implemented yet")
	}

	rng, seed, seedKnown := resolveRandomSource(config)
	parameters := deriveStrategyParameters(config)
	state := newStrategyState(config)
	best := Best{Cost: math.Inf(1)}
	evaluations := 0
	iterations := 0
	reason := TerminationMaxIterations

	for generation := range config.MaxIterations {
		ctxErr = ctx.Err()
		if ctxErr != nil {
			return nil, ctxErr
		}

		populationSize := generationPopulationSize(config, evaluations)
		if populationSize == 0 {
			reason = TerminationMaxEvaluations

			break
		}

		refreshEigensystemIfStale(state, evaluations, config.Lambda, parameters)
		population := samplePopulation(state, populationSize, rng)

		evaluationErr := evaluatePopulation(ctx, config, population)
		if evaluationErr != nil {
			return nil, evaluationErr
		}

		evaluations += len(population)
		updateBest(&best, population)

		if len(population) < config.Mu {
			reason = TerminationMaxEvaluations

			break
		}

		sortPopulation(population)
		updateDistribution(state, population, parameters, generation)
		iterations = generation + 1

		if config.MaxEvaluations > 0 && evaluations >= config.MaxEvaluations {
			reason = TerminationMaxEvaluations

			break
		}
	}

	return &Result{
		TerminationReason: reason,
		GlobalBest:        best,
		FuncEvalCount:     evaluations,
		IterationCount:    iterations,
		Seed:              seed,
		SeedKnown:         seedKnown,
	}, nil
}

func resolveRunOptions(options []RunOption) error {
	var resolved runOptions

	for index, option := range options {
		if option.apply == nil {
			return fmt.Errorf("run option %d is invalid", index)
		}

		applyErr := option.apply(&resolved)
		if applyErr != nil {
			return fmt.Errorf("apply run option %d: %w", index, applyErr)
		}
	}

	return nil
}

func resolveRandomSource(config *Config) (*rand.Rand, int64, bool) {
	if config.Rand != nil {
		return config.Rand, 0, false
	}

	seed := time.Now().UnixNano()
	if config.Seed != nil {
		seed = *config.Seed
	}

	config.Rand = rand.New(rand.NewSource(seed))

	return config.Rand, seed, true
}

func newStrategyState(config *Config) *strategyState {
	d := make([]float64, config.ProblemSize)
	for index := range d {
		d[index] = 1
	}

	return &strategyState{
		m:      append([]float64(nil), config.InitialMean...),
		psigma: make([]float64, config.ProblemSize),
		pc:     make([]float64, config.ProblemSize),
		c:      identityMatrix(config.ProblemSize),
		b:      identityMatrix(config.ProblemSize),
		d:      d,
		sigma:  config.InitialSigma,
	}
}

func generationPopulationSize(config *Config, evaluations int) int {
	if config.MaxEvaluations == 0 {
		return config.Lambda
	}

	return min(config.Lambda, config.MaxEvaluations-evaluations)
}

func samplePopulation(state *strategyState, populationSize int, rng *rand.Rand) []candidate {
	population := make([]candidate, populationSize)

	for index := range population {
		z := make([]float64, len(state.m))
		for coordinate := range z {
			z[coordinate] = rng.NormFloat64()
		}

		y := transformNormal(state.b, state.d, z)

		x := make([]float64, len(state.m))
		for coordinate := range x {
			x[coordinate] = state.m[coordinate] + state.sigma*y[coordinate]
		}

		population[index] = candidate{x: x, y: y, z: z}
	}

	return population
}

func transformNormal(b [][]float64, d, z []float64) []float64 {
	scaled := make([]float64, len(z))
	for index := range scaled {
		scaled[index] = d[index] * z[index]
	}

	return matrixVectorProduct(b, scaled)
}

func evaluatePopulation(ctx context.Context, config *Config, population []candidate) error {
	if !config.EnableParallel || len(population) < 2 {
		for index := range population {
			ctxErr := ctx.Err()
			if ctxErr != nil {
				return ctxErr
			}

			population[index].cost = config.ObjectiveFunc(population[index].x)
		}

		return nil
	}

	workerCount := config.MaxWorkers
	if workerCount == 0 {
		workerCount = runtime.NumCPU()
	}

	workerCount = min(workerCount, len(population))
	jobs := make(chan int)

	var workers sync.WaitGroup
	workers.Add(workerCount)

	for range workerCount {
		go func() {
			defer workers.Done()

			for index := range jobs {
				if ctx.Err() != nil {
					continue
				}

				population[index].cost = config.ObjectiveFunc(population[index].x)
			}
		}()
	}

	for index := range population {
		select {
		case jobs <- index:
		case <-ctx.Done():
			close(jobs)
			workers.Wait()

			return ctx.Err()
		}
	}

	close(jobs)
	workers.Wait()

	return ctx.Err()
}

func updateBest(best *Best, population []candidate) {
	for _, current := range population {
		if math.IsNaN(current.cost) || current.cost >= best.Cost {
			continue
		}

		best.Cost = current.cost
		best.Position = append(best.Position[:0], current.x...)
	}
}

func sortPopulation(population []candidate) {
	sort.SliceStable(population, func(left, right int) bool {
		leftCost := population[left].cost
		rightCost := population[right].cost

		if math.IsNaN(leftCost) {
			return false
		}

		return math.IsNaN(rightCost) || leftCost < rightCost
	})
}

func updateDistribution(
	state *strategyState,
	population []candidate,
	parameters strategyParameters,
	generation int,
) {
	weightedX, weightedY := recombine(population, parameters.weights)
	inverseStep := inverseSquareRootProduct(state.b, state.d, weightedY)
	updateStepPath(state.psigma, inverseStep, parameters)
	hSigma := covariancePathIsActive(state.psigma, generation, len(state.m), parameters.cSigma)
	updateCovariancePath(state.pc, weightedY, hSigma, parameters)
	updateCovariance(state.c, state.pc, population, hSigma, parameters)
	copy(state.m, weightedX)
	state.sigma *= math.Exp((parameters.cSigma / parameters.dSigma) *
		(vectorNorm(state.psigma)/expectedNormalNorm(len(state.m)) - 1))
}

func recombine(population []candidate, weights []float64) ([]float64, []float64) {
	weightedX := make([]float64, len(population[0].x))
	weightedY := make([]float64, len(population[0].y))

	for index, weight := range weights {
		for coordinate := range weightedX {
			weightedX[coordinate] += weight * population[index].x[coordinate]
			weightedY[coordinate] += weight * population[index].y[coordinate]
		}
	}

	return weightedX, weightedY
}

// inverseSquareRootProduct computes B*diag(1/D)*B^T*y. This is C^(-1/2)y,
// not C^(-1)y and not an element-wise reciprocal.
func inverseSquareRootProduct(b [][]float64, d, y []float64) []float64 {
	coordinates := make([]float64, len(y))

	for column := range b {
		for row := range b {
			coordinates[column] += b[row][column] * y[row]
		}

		if d[column] > 0 {
			coordinates[column] /= d[column]
		} else {
			coordinates[column] = 0
		}
	}

	return matrixVectorProduct(b, coordinates)
}

func updateStepPath(path, inverseStep []float64, parameters strategyParameters) {
	decay := 1 - parameters.cSigma
	scale := math.Sqrt(parameters.cSigma * (2 - parameters.cSigma) * parameters.muEff)

	for index := range path {
		path[index] = decay*path[index] + scale*inverseStep[index]
	}
}

func covariancePathIsActive(path []float64, generation, dimension int, cSigma float64) bool {
	normalizer := math.Sqrt(1 - math.Pow(1-cSigma, 2*float64(generation+1)))
	threshold := (1.4 + 2/(float64(dimension)+1)) * expectedNormalNorm(dimension)

	return vectorNorm(path)/normalizer < threshold
}

func updateCovariancePath(
	path, weightedStep []float64,
	hSigma bool,
	parameters strategyParameters,
) {
	decay := 1 - parameters.cc

	scale := 0.0
	if hSigma {
		scale = math.Sqrt(parameters.cc * (2 - parameters.cc) * parameters.muEff)
	}

	for index := range path {
		path[index] = decay*path[index] + scale*weightedStep[index]
	}
}

func updateCovariance(
	covariance [][]float64,
	path []float64,
	population []candidate,
	hSigma bool,
	parameters strategyParameters,
) {
	correction := 0.0
	if !hSigma {
		correction = parameters.cc * (2 - parameters.cc)
	}

	oldScale := 1 - parameters.c1 - parameters.cmu + parameters.c1*correction

	for row := range covariance {
		for column := row; column < len(covariance); column++ {
			value := oldScale * covariance[row][column]
			covariance[row][column] = value
			covariance[column][row] = value
		}
	}

	symmetricRankOneUpdate(covariance, parameters.c1, path)

	for index, weight := range parameters.weights {
		// updateDistribution checks that at least Mu candidates were evaluated,
		// and deriveStrategyParameters creates exactly Mu weights.
		symmetricRankOneUpdate(
			covariance,
			parameters.cmu*weight,
			population[index].y, //nolint:gosec // The population/weight size invariant is checked by the caller.
		)
	}
}

func refreshEigensystemIfStale(
	state *strategyState,
	evaluations, lambda int,
	parameters strategyParameters,
) bool {
	stalenessLimit := float64(lambda) /
		(10 * float64(len(state.m)) * (parameters.c1 + parameters.cmu))
	if float64(evaluations-state.eigenEval) <= stalenessLimit {
		return false
	}

	values, vectors := symmetricEigendecomposition(state.c)
	for index, value := range values {
		state.d[index] = math.Sqrt(value)
	}

	state.b = vectors
	state.eigenEval = evaluations

	return true
}

func expectedNormalNorm(dimension int) float64 {
	n := float64(dimension)

	return math.Sqrt(n) * (1 - 1/(4*n) + 1/(21*n*n))
}

func vectorNorm(vector []float64) float64 {
	var sum float64
	for _, value := range vector {
		sum += value * value
	}

	return math.Sqrt(sum)
}
