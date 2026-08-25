package cmaes

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// strategyState is the mutable CMA-ES distribution. Full covariance uses c
// and b, with eigenvectors in the columns of b. Separable covariance leaves
// both dense matrices nil and stores only the covariance diagonal.
type strategyState struct {
	mode      CovarianceMode
	m         []float64
	psigma    []float64
	pc        []float64
	c         [][]float64
	b         [][]float64
	diagonal  []float64
	d         []float64
	sigma     float64
	eigenEval int
}

type candidate struct {
	x                   []float64
	y                   []float64
	z                   []float64
	evaluationX         []float64
	cost                float64
	constraintViolation float64
	boundaryDistance    float64
	boundaryPenalty     float64
	evaluated           bool
}

type optimizationRun struct {
	config           *Config
	state            *strategyState
	tracker          *convergenceTracker
	reason           TerminationReason
	curve            []float64
	sigmaHistory     []float64
	conditionHistory []float64
	options          runOptions
	best             Best
	parameters       strategyParameters
	evaluations      int
	iterations       int
}

// Optimize runs CMA-ES with a background context.
func Optimize(config *Config) (*Result, error) {
	return OptimizeContext(context.Background(), config)
}

// OptimizeContext runs CMA-ES, honoring context cancellation.
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

	resolved, optionsErr := resolveRunOptions(options)
	if optionsErr != nil {
		return nil, optionsErr
	}

	validationErr := config.Validate()
	if validationErr != nil {
		return nil, validationErr
	}

	optionsErr = validateRunOptions(config, resolved)
	if optionsErr != nil {
		return nil, optionsErr
	}

	rng, seed, seedKnown := resolveRandomSource(config)
	run := newOptimizationRun(config, resolved)

	logOptimizationStarted(ctx, resolved.logger, config)

	runErr := run.execute(ctx, rng)
	if runErr != nil {
		return nil, runErr
	}

	result := run.result(seed, seedKnown)

	logOptimizationCompleted(ctx, resolved.logger, result)

	return result, nil
}

func newOptimizationRun(config *Config, options runOptions) *optimizationRun {
	state := newStrategyState(config)
	if options.hasInitialMean {
		copy(state.m, options.initialMean)
		state.sigma = options.initialSigma
	}

	return &optimizationRun{
		config: config,
		state:  state,
		tracker: newConvergenceTracker(config.Convergence, config.Constraints,
			state.sigma, config.ProblemSize, config.Lambda),
		curve:            make([]float64, 0, config.MaxIterations),
		sigmaHistory:     make([]float64, 0, config.MaxIterations),
		conditionHistory: make([]float64, 0, config.MaxIterations),
		best:             Best{Cost: math.Inf(1), ConstraintViolation: math.Inf(1)},
		options:          options,
		parameters:       deriveStrategyParameters(config),
		reason:           TerminationMaxIterations,
	}
}

func (run *optimizationRun) execute(ctx context.Context, rng *rand.Rand) error {
	for generation := range run.config.MaxIterations {
		if ctx.Err() != nil {
			run.reason = TerminationCancelled

			break
		}

		stop, err := run.executeGeneration(ctx, generation, rng)
		if err != nil {
			return err
		}

		if stop {
			break
		}
	}

	return nil
}

func (run *optimizationRun) executeGeneration(
	ctx context.Context,
	generation int,
	rng *rand.Rand,
) (bool, error) {
	populationSize := generationPopulationSize(run.config, run.evaluations)
	if populationSize == 0 {
		run.reason = TerminationMaxEvaluations

		return true, nil
	}

	refreshEigensystemIfStale(
		run.state, run.evaluations, run.config.Lambda, run.parameters,
	)

	population := samplePopulation(run.state, populationSize, rng)
	if generation == 0 {
		applyInitialPopulation(population, run.state, run.options.initialPopulation)
	}

	applyBoundaryHandling(population, run.state, run.config)

	evaluated, evaluationErr := evaluatePopulation(ctx, run.config, population)
	run.evaluations += evaluated

	if evaluationErr != nil {
		return run.handleEvaluationError(population, evaluationErr)
	}

	assignBoundaryPenalties(population, run.config.BoundaryMethod)
	updateBest(&run.best, population, run.config.Constraints)

	if len(population) < run.config.Mu {
		run.reason = TerminationMaxEvaluations

		return true, nil
	}

	run.completeGeneration(ctx, generation, population)

	return run.shouldStop(population), nil
}

func (run *optimizationRun) handleEvaluationError(
	population []candidate,
	evaluationErr error,
) (bool, error) {
	if !errors.Is(evaluationErr, context.Canceled) &&
		!errors.Is(evaluationErr, context.DeadlineExceeded) {
		return false, evaluationErr
	}

	partial := evaluatedCandidates(population)
	assignBoundaryPenalties(partial, run.config.BoundaryMethod)
	updateBest(&run.best, partial, run.config.Constraints)
	run.reason = TerminationCancelled

	return true, nil
}

func (run *optimizationRun) completeGeneration(
	ctx context.Context,
	generation int,
	population []candidate,
) {
	sortPopulation(population, run.config.Constraints)
	updateDistribution(run.state, population, run.parameters, generation)
	run.iterations = generation + 1

	condition := covarianceConditionNumber(run.state.d)
	run.curve = append(run.curve, run.best.Cost)
	run.sigmaHistory = append(run.sigmaHistory, run.state.sigma)
	run.conditionHistory = append(run.conditionHistory, condition)
	notifyLifecycle(
		run.options, run.iterations, run.evaluations, run.best, population, run.state,
	)
	logIterationCompleted(
		ctx, run.options.logger, run.iterations, run.evaluations,
		run.best, run.state.sigma, condition,
	)
}

func (run *optimizationRun) shouldStop(population []candidate) bool {
	stopReason, stop := run.tracker.observe(
		run.iterations, run.best, run.state, population,
	)
	if stop {
		run.reason = stopReason

		return true
	}

	if run.config.MaxEvaluations > 0 && run.evaluations >= run.config.MaxEvaluations {
		run.reason = TerminationMaxEvaluations

		return true
	}

	return false
}

func (run *optimizationRun) result(seed int64, seedKnown bool) *Result {
	return &Result{
		ConvergenceCurve:       run.curve,
		SigmaHistory:           run.sigmaHistory,
		ConditionNumberHistory: run.conditionHistory,
		TerminationReason:      run.reason,
		GlobalBest:             run.best,
		FuncEvalCount:          run.evaluations,
		IterationCount:         run.iterations,
		Seed:                   seed,
		SeedKnown:              seedKnown,
	}
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
	if config.CovarianceMode == CovarianceSeparable {
		return newSeparableStrategyState(config)
	}

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
		mode:   CovarianceFull,
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

		y := transformStrategyNormal(state, z)

		x := make([]float64, len(state.m))
		for coordinate := range x {
			x[coordinate] = state.m[coordinate] + state.sigma*y[coordinate]
		}

		population[index] = candidate{x: x, y: y, z: z}
	}

	return population
}

func transformStrategyNormal(state *strategyState, z []float64) []float64 {
	if state.mode == CovarianceSeparable {
		return transformSeparableNormal(state.d, z)
	}

	return transformNormal(state.b, state.d, z)
}

func applyInitialPopulation(
	population []candidate,
	state *strategyState,
	initialPositions [][]float64,
) {
	for index, position := range initialPositions {
		copy(population[index].x, position)
		recomputeStep(&population[index], state)
	}
}

func transformNormal(b [][]float64, d, z []float64) []float64 {
	scaled := make([]float64, len(z))
	for index := range scaled {
		scaled[index] = d[index] * z[index]
	}

	return matrixVectorProduct(b, scaled)
}

func evaluatePopulation(ctx context.Context, config *Config, population []candidate) (int, error) {
	if !config.EnableParallel || len(population) < 2 {
		for index := range population {
			ctxErr := ctx.Err()
			if ctxErr != nil {
				return index, ctxErr
			}

			evaluateCandidate(&population[index], config)
			population[index].evaluated = true
		}

		return len(population), nil
	}

	workerCount := config.MaxWorkers
	if workerCount == 0 {
		workerCount = runtime.NumCPU()
	}

	workerCount = min(workerCount, len(population))
	jobs := make(chan int)

	var evaluated atomic.Int64

	var workers sync.WaitGroup
	workers.Add(workerCount)

	for range workerCount {
		go func() {
			defer workers.Done()

			for index := range jobs {
				if ctx.Err() != nil {
					continue
				}

				evaluateCandidate(&population[index], config)
				population[index].evaluated = true
				evaluated.Add(1)
			}
		}()
	}

	for index := range population {
		select {
		case jobs <- index:
		case <-ctx.Done():
			close(jobs)
			workers.Wait()

			return int(evaluated.Load()), ctx.Err()
		}
	}

	close(jobs)
	workers.Wait()

	return int(evaluated.Load()), ctx.Err()
}

func evaluatedCandidates(population []candidate) []candidate {
	evaluated := make([]candidate, 0, len(population))
	for _, current := range population {
		if current.evaluated {
			evaluated = append(evaluated, current)
		}
	}

	return evaluated
}

func evaluateCandidate(current *candidate, config *Config) {
	position := current.evaluatedPosition()
	constraint := EvaluateConstraints(position, config.Constraints)
	current.cost = config.ObjectiveFunc(position)
	current.constraintViolation = constraint.Violation
}

func updateBest(best *Best, population []candidate, constraints *ConstraintConfig) {
	for _, current := range population {
		candidateEvaluation := CandidateEvaluation{
			Cost:                current.cost,
			ConstraintViolation: current.constraintViolation,
		}
		bestEvaluation := CandidateEvaluation{
			Cost:                best.Cost,
			ConstraintViolation: best.ConstraintViolation,
		}

		if !BetterConstrainedCandidate(candidateEvaluation, bestEvaluation, constraints) {
			continue
		}

		best.Cost = current.cost
		best.Position = append(best.Position[:0], current.evaluatedPosition()...)
		best.ConstraintViolation = current.constraintViolation
	}
}

func sortPopulation(population []candidate, constraints *ConstraintConfig) {
	sort.SliceStable(population, func(left, right int) bool {
		return BetterConstrainedCandidate(
			rankedCandidateEvaluation(population[left]),
			rankedCandidateEvaluation(population[right]),
			constraints,
		)
	})
}

func rankedCandidateEvaluation(current candidate) CandidateEvaluation {
	return CandidateEvaluation{
		Cost:                current.cost + current.boundaryPenalty,
		ConstraintViolation: current.constraintViolation,
	}
}

func updateDistribution(
	state *strategyState,
	population []candidate,
	parameters strategyParameters,
	generation int,
) {
	if len(population) < len(parameters.weights)+len(parameters.negativeWeights) {
		parameters.negativeWeights = nil
	}

	weightedX, weightedY := recombine(population, parameters.weights)
	inverseStep := inverseCovarianceSquareRootProduct(state, weightedY)
	updateStepPath(state.psigma, inverseStep, parameters)
	hSigma := covariancePathIsActive(state.psigma, generation, len(state.m), parameters.cSigma)
	updateCovariancePath(state.pc, weightedY, hSigma, parameters)
	updateStrategyCovariance(state, population, hSigma, parameters)
	copy(state.m, weightedX)
	state.sigma *= math.Exp((parameters.cSigma / parameters.dSigma) *
		(vectorNorm(state.psigma)/expectedNormalNorm(len(state.m)) - 1))
}

func inverseCovarianceSquareRootProduct(state *strategyState, step []float64) []float64 {
	if state.mode == CovarianceSeparable {
		return inverseSeparableSquareRootProduct(state.d, step)
	}

	return inverseSquareRootProduct(state.b, state.d, step)
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
	oldScale := covarianceDecay(parameters, hSigma)

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

func updateStrategyCovariance(
	state *strategyState,
	population []candidate,
	hSigma bool,
	parameters strategyParameters,
) {
	if state.mode == CovarianceSeparable {
		updateSeparableCovariance(state, population, hSigma, parameters)

		return
	}

	updateCovariance(state.c, state.pc, population, hSigma, parameters)

	for index, weight := range parameters.negativeWeights {
		populationIndex := len(parameters.weights) + index
		symmetricRankOneUpdate(
			state.c,
			parameters.cmu*weight,
			activeUpdateVector(state, population[populationIndex].y),
		)
	}
}

func refreshEigensystemIfStale(
	state *strategyState,
	evaluations, lambda int,
	parameters strategyParameters,
) bool {
	if state.mode == CovarianceSeparable {
		return false
	}

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
