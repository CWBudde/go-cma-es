package cmaes

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
)

// RestartStrategy selects the population restart schedule.
type RestartStrategy string

const (
	// RestartIPOP doubles the population size after every run.
	RestartIPOP RestartStrategy = "ipop"
	// RestartBIPOP alternates IPOP runs with shorter, smaller runs, always
	// advancing the regime that has consumed fewer objective evaluations.
	RestartBIPOP RestartStrategy = "bipop"
)

// RestartRegime identifies one of BIPOP's two budget accounts. IPOP records
// all of its runs as large-population runs.
type RestartRegime string

const (
	// RestartRegimeLarge is the population-doubling IPOP regime.
	RestartRegimeLarge RestartRegime = "large"
	// RestartRegimeSmall is BIPOP's randomized small-population regime.
	RestartRegimeSmall RestartRegime = "small"
)

// RestartRecord describes one independent CMA-ES run in a restart schedule.
// InitialMean and Best are deep copies owned by the result.
type RestartRecord struct {
	InitialMean []float64 `json:"initial_mean"`

	TerminationReason TerminationReason `json:"termination_reason"`
	Regime            RestartRegime     `json:"regime"`
	Best              Best              `json:"best"`

	InitialSigma float64 `json:"initial_sigma"`

	Lambda      int   `json:"lambda"`
	Evaluations int   `json:"evaluations"`
	Iterations  int   `json:"iterations"`
	Seed        int64 `json:"seed"`
	Restart     int   `json:"restart"`
	Budget      int   `json:"budget"`
	SeedKnown   bool  `json:"seed_known"`
}

// RestartResult holds the best solution and accounting across all runs. The
// first element of Restarts is the initial run; Restart is therefore zero for
// that record and one for the first actual restart.
type RestartResult struct {
	Restarts []RestartRecord `json:"restarts"`

	TerminationReason TerminationReason `json:"termination_reason"`
	Strategy          RestartStrategy   `json:"strategy"`
	GlobalBest        Best              `json:"global_best"`

	FuncEvalCount int   `json:"func_eval_count"`
	Seed          int64 `json:"seed"`
	SeedKnown     bool  `json:"seed_known"`
}

type restartSchedule struct {
	strategy         RestartStrategy
	baseLambda       int
	largeRuns        int
	largeEvaluations int
	smallEvaluations int
}

type restartRunPlan struct {
	regime        RestartRegime
	lambda        int
	budget        int
	maxIterations int
	sigma         float64
}

// OptimizeWithRestarts runs IPOP- or BIPOP-CMA-ES with a background context.
// Config.MaxEvaluations is the shared budget across all runs and must be
// positive. A target-cost termination ends the complete schedule immediately;
// other completed-run termination criteria start a fresh run for as long as any
// budget remains, down to a final single-evaluation run.
func OptimizeWithRestarts(
	config *Config,
	strategy RestartStrategy,
	options ...RunOption,
) (*RestartResult, error) {
	return OptimizeWithRestartsContext(context.Background(), config, strategy, options...)
}

// OptimizeWithRestartsContext runs IPOP- or BIPOP-CMA-ES, honoring context
// cancellation. The caller's Config is never modified. Lifecycle observers
// receive every run, with iteration and evaluation counts local to that run;
// initial-population and initial-mean options apply only to the first run.
func OptimizeWithRestartsContext(
	ctx context.Context,
	config *Config,
	strategy RestartStrategy,
	options ...RunOption,
) (*RestartResult, error) {
	if ctx == nil {
		return nil, errors.New("context must not be nil")
	}

	ctxErr := ctx.Err()
	if ctxErr != nil {
		return nil, ctxErr
	}

	if config == nil {
		return nil, errors.New("config must not be nil")
	}

	validationErr := config.Validate()
	if validationErr != nil {
		return nil, validationErr
	}

	if config.MaxEvaluations == 0 {
		return nil, errors.New("max_evaluations must be positive for restart optimization")
	}

	if strategy != RestartIPOP && strategy != RestartBIPOP {
		return nil, fmt.Errorf("unknown restart strategy %q", strategy)
	}

	resolved, err := resolveRunOptions(options)
	if err != nil {
		return nil, err
	}

	optionsErr := validateRunOptions(config, resolved)
	if optionsErr != nil {
		return nil, optionsErr
	}

	rng, seed, seedKnown := resolveRandomSource(config)
	result := &RestartResult{
		Restarts:          make([]RestartRecord, 0),
		TerminationReason: TerminationMaxEvaluations,
		Strategy:          strategy,
		GlobalBest:        Best{Cost: math.Inf(1), ConstraintViolation: math.Inf(1)},
		Seed:              seed,
		SeedKnown:         seedKnown,
	}
	schedule := restartSchedule{strategy: strategy, baseLambda: config.Lambda}

	for result.FuncEvalCount < config.MaxEvaluations {
		if contextCancelled(ctx) {
			result.TerminationReason = TerminationCancelled

			return result, nil
		}

		var (
			runResult *Result
			runErr    error
		)

		remaining := config.MaxEvaluations - result.FuncEvalCount
		planningBudget := max(2, remaining)
		plan := schedule.next(config, planningBudget, rng)
		plan.budget = min(plan.budget, remaining)
		runConfig := restartConfig(config, plan, len(result.Restarts), rng)
		perRunOptions := restartOptions(resolved, runConfig, len(result.Restarts))

		if remaining == 1 {
			// The public single-run contract rejects a budget below lambda.
			// A restart schedule can nevertheless arrive here after a
			// convergence stop with one evaluation left. Evaluate that final
			// partial population without completing a distribution update, just
			// as the ordinary run does for any other undersized final population.
			runConfig.Mu = runConfig.Lambda
			runResult, runErr = optimizeRestartTail(ctx, runConfig, perRunOptions)
		} else {
			runResult, runErr = OptimizeContext(
				ctx, runConfig, resolvedOptions(perRunOptions),
			)
		}

		if runErr != nil {
			return nil, runErr
		}

		record := makeRestartRecord(
			runConfig, runResult, perRunOptions, plan, len(result.Restarts),
		)
		result.Restarts = append(result.Restarts, record)
		result.FuncEvalCount += runResult.FuncEvalCount
		updateRestartBest(&result.GlobalBest, runResult.GlobalBest, config.Constraints)
		schedule.observe(plan.regime, runResult.FuncEvalCount)
		result.TerminationReason = runResult.TerminationReason

		if runResult.TerminationReason == TerminationTargetCost ||
			runResult.TerminationReason == TerminationCancelled {
			break
		}
	}

	if result.FuncEvalCount >= config.MaxEvaluations &&
		result.TerminationReason != TerminationTargetCost &&
		result.TerminationReason != TerminationCancelled {
		result.TerminationReason = TerminationMaxEvaluations
	}

	return result, nil
}

func contextCancelled(ctx context.Context) bool {
	select {
	case <-ctx.Done():
		return true
	default:
		return false
	}
}

// optimizeRestartTail runs the already-validated schedule configuration when
// only one shared-budget evaluation remains. Config.Validate deliberately
// rejects this shape for standalone runs; the restart layer can accept it
// because an earlier run has already produced a valid best-so-far result.
func optimizeRestartTail(
	ctx context.Context,
	config *Config,
	options runOptions,
) (*Result, error) {
	rng, seed, seedKnown := resolveRandomSource(config)
	run := newOptimizationRun(config, options)

	logOptimizationStarted(ctx, options.logger, config)

	runErr := run.execute(ctx, rng)
	if runErr != nil {
		logOptimizationFailed(ctx, options.logger, runErr)

		return nil, runErr
	}

	result := run.result(seed, seedKnown)

	logOptimizationCompleted(ctx, options.logger, result)

	return result, nil
}

func (schedule *restartSchedule) next(
	config *Config,
	remaining int,
	rng *rand.Rand,
) restartRunPlan {
	if schedule.strategy == RestartIPOP || schedule.largeRuns == 0 ||
		schedule.largeEvaluations <= schedule.smallEvaluations {
		return schedule.nextLarge(config, remaining)
	}

	return schedule.nextSmall(config, remaining, rng)
}

func (schedule *restartSchedule) nextLarge(config *Config, remaining int) restartRunPlan {
	lambda := doubledPopulation(schedule.baseLambda, schedule.largeRuns)
	lambda = min(lambda, remaining)

	return restartRunPlan{
		regime:        RestartRegimeLarge,
		lambda:        lambda,
		budget:        remaining,
		maxIterations: config.MaxIterations,
		sigma:         config.InitialSigma,
	}
}

func (schedule *restartSchedule) nextSmall(
	config *Config,
	remaining int,
	rng *rand.Rand,
) restartRunPlan {
	// Hansen's BIPOP schedule samples logarithmically between the base
	// population and *half* the next IPOP population, with a squared uniform
	// exponent to favor genuinely small runs:
	//
	//	lambda_small = lambda_base * (0.5 * lambda_large / lambda_base)^U^2
	//
	// with lambda_large = lambda_base * 2^largeRuns, so the exponent is
	// (largeRuns - 1) * U^2. Dropping the halving would let a small run be
	// drawn at the full large-regime population, which is the one thing the
	// second budget account exists to prevent. nextSmall is only reached once
	// a large run has completed, so the exponent never goes negative.
	exponent := float64(schedule.largeRuns-1) * math.Pow(rng.Float64(), 2)
	lambda := int(float64(schedule.baseLambda) * math.Pow(2, exponent))
	lambda = min(max(schedule.baseLambda, lambda), remaining)

	// Small runs receive a randomized fraction of half the large regime's
	// spend. Expressing the cap as whole generations avoids manufacturing a
	// partial small generation merely to meet its local allocation.
	maximumBudget := min(remaining, max(lambda, schedule.largeEvaluations/2))
	maximumGenerations := min(config.MaxIterations, maximumBudget/lambda)
	maximumGenerations = max(1, maximumGenerations)
	budgetGenerations := 1 + rng.Intn(maximumGenerations)
	budget := min(remaining, lambda*budgetGenerations)

	return restartRunPlan{
		regime:        RestartRegimeSmall,
		lambda:        lambda,
		budget:        budget,
		maxIterations: config.MaxIterations,
		sigma:         config.InitialSigma * math.Pow(0.01, rng.Float64()),
	}
}

func (schedule *restartSchedule) observe(regime RestartRegime, evaluations int) {
	if regime == RestartRegimeLarge {
		schedule.largeEvaluations += evaluations
		schedule.largeRuns++

		return
	}

	schedule.smallEvaluations += evaluations
}

func doubledPopulation(base, exponent int) int {
	population := base
	for range exponent {
		if population > int(^uint(0)>>1)/2 {
			return int(^uint(0) >> 1)
		}

		population *= 2
	}

	return population
}

func restartConfig(
	config *Config,
	plan restartRunPlan,
	restart int,
	rng *rand.Rand,
) *Config {
	runConfig := *config
	runConfig.InitialMean = append([]float64(nil), config.InitialMean...)
	runConfig.LowerBounds = append([]float64(nil), config.LowerBounds...)
	runConfig.UpperBounds = append([]float64(nil), config.UpperBounds...)
	runConfig.Lambda = plan.lambda
	runConfig.Mu = scaledParentCount(config.Mu, config.Lambda, plan.lambda)
	runConfig.InitialSigma = plan.sigma
	runConfig.MaxIterations = plan.maxIterations
	runConfig.MaxEvaluations = plan.budget

	if restart > 0 {
		runConfig.InitialMean = randomRestartMean(config, rng)
	}

	if config.Rand == nil {
		// Every run, the first included, draws its seed from the schedule's own
		// generator. Handing run 0 the seed that generator was constructed from
		// would make it replay that stream from position 0, correlating its
		// samples with every scheduling draw taken afterwards.
		runSeed := rng.Int63()

		runConfig.Seed = &runSeed
		runConfig.Rand = nil
	} else {
		runConfig.Seed = nil
		runConfig.Rand = rng
	}

	return &runConfig
}

func scaledParentCount(baseMu, baseLambda, lambda int) int {
	mu := int(math.Round(float64(lambda) * float64(baseMu) / float64(baseLambda)))

	return min(lambda, max(1, mu))
}

func randomRestartMean(config *Config, rng *rand.Rand) []float64 {
	mean := make([]float64, config.ProblemSize)
	for coordinate := range mean {
		lower, upper := coordinateBounds(config, coordinate)
		mean[coordinate] = lower + rng.Float64()*(upper-lower)
	}

	return mean
}

func restartOptions(options runOptions, config *Config, restart int) runOptions {
	restarted := cloneRunOptions(options)
	if restart == 0 {
		return restarted
	}

	restarted.initialPopulation = nil

	restarted.initialMean = append([]float64(nil), config.InitialMean...)
	restarted.initialSigma = config.InitialSigma
	restarted.hasInitialMean = true

	return restarted
}

func cloneRunOptions(options runOptions) runOptions {
	options.initialPopulation = clonePositions(options.initialPopulation)
	options.initialMean = append([]float64(nil), options.initialMean...)

	return options
}

func resolvedOptions(options runOptions) RunOption {
	snapshot := cloneRunOptions(options)

	return RunOption{apply: func(destination *runOptions) error {
		*destination = cloneRunOptions(snapshot)

		return nil
	}}
}

func makeRestartRecord(
	config *Config,
	result *Result,
	options runOptions,
	plan restartRunPlan,
	restart int,
) RestartRecord {
	initialMean := config.InitialMean
	initialSigma := config.InitialSigma

	if options.hasInitialMean {
		initialMean = options.initialMean
		initialSigma = options.initialSigma
	}

	return RestartRecord{
		InitialMean:       append([]float64(nil), initialMean...),
		Best:              cloneBest(result.GlobalBest),
		TerminationReason: result.TerminationReason,
		Regime:            plan.regime,
		InitialSigma:      initialSigma,
		Lambda:            config.Lambda,
		Evaluations:       result.FuncEvalCount,
		Iterations:        result.IterationCount,
		Seed:              result.Seed,
		Restart:           restart,
		Budget:            plan.budget,
		SeedKnown:         result.SeedKnown,
	}
}

func updateRestartBest(best *Best, candidate Best, constraints *ConstraintConfig) {
	current := CandidateEvaluation{
		Cost:                best.Cost,
		ConstraintViolation: best.ConstraintViolation,
	}
	proposed := CandidateEvaluation{
		Cost:                candidate.Cost,
		ConstraintViolation: candidate.ConstraintViolation,
	}

	if BetterConstrainedCandidate(proposed, current, constraints) {
		*best = cloneBest(candidate)
	}
}
