package cmaes

import (
	"context"
	"fmt"
	"log/slog"
)

// Progress describes the best solution known after a completed iteration.
// Best and its Position are deep copies owned by the observer.
type Progress struct {
	Best            Best
	Iteration       int
	EvaluationCount int
}

// ProgressObserver receives progress synchronously on the optimization
// goroutine after each completed iteration.
//
// A panic raised by the observer is recovered and reported through the
// registered Logger; it never aborts the run.
type ProgressObserver func(Progress)

// PopulationCandidate is one evaluated member of a population snapshot.
type PopulationCandidate struct {
	Position            []float64
	Cost                float64
	ConstraintViolation float64
}

// PopulationSnapshot contains the evaluated population after a completed
// iteration. Population is in selection order and all slices are deep copies.
type PopulationSnapshot struct {
	Population      []PopulationCandidate
	Best            Best
	Iteration       int
	EvaluationCount int
}

// PopulationObserver receives an opt-in deep copy of each evaluated
// population. No population copying occurs when it is nil.
//
// A panic raised by the observer is recovered and reported through the
// registered Logger; it never aborts the run.
type PopulationObserver func(PopulationSnapshot)

// BlockDistributionSnapshot describes one independently adapted covariance
// block. Coordinates maps its matrix rows to problem coordinates;
// Eigenvalues contains the block's axis scales D, and Eigenvectors contains B
// with the corresponding eigenvectors in its columns.
type BlockDistributionSnapshot struct {
	Coordinates  []int
	Eigenvalues  []float64
	Eigenvectors [][]float64
}

// DistributionSnapshot describes the sampling distribution after a completed
// iteration. Eigenvalues contains D, the square roots of C's eigenvalues. For
// full and separable covariance, Eigenvectors contains the dense B matrix. For
// block covariance it remains nil and Blocks carries the sparse representation,
// avoiding an O(n^2) observer allocation for an O(n*k) strategy. That holds for
// every block configuration, including the BlockSize of one or of ProblemSize
// that the strategy runs as separable or full covariance internally.
//
// All distribution fields describe the same iteration: B and D are
// decomposed from the covariance that Mean and Sigma were updated with, not
// from the strategy's lazily refreshed eigensystem, so the ellipse this
// snapshot describes is the one the iteration actually ended with. The same
// eigensystem produces ConditionNumber and the matching entry of
// Result.ConditionNumberHistory.
type DistributionSnapshot struct {
	Mean         []float64
	Eigenvalues  []float64
	Eigenvectors [][]float64
	Blocks       []BlockDistributionSnapshot

	Sigma           float64
	ConditionNumber float64
	Iteration       int
	EvaluationCount int
}

// DistributionObserver receives an opt-in deep copy of each distribution.
//
// A panic raised by the observer is recovered and reported through the
// registered Logger; it never aborts the run.
type DistributionObserver func(DistributionSnapshot)

// Logger receives structured optimization lifecycle events. *slog.Logger
// implements Logger. Iteration events are emitted at slog.LevelDebug, run
// start and termination at slog.LevelInfo, and a failed run at
// slog.LevelError.
//
// A panic raised by Log is recovered and discarded; it never aborts the run.
type Logger interface {
	Log(ctx context.Context, level slog.Level, message string, args ...any)
}

// RunOption customizes one optimization run. Construct options with the
// With* functions in this file.
type RunOption struct {
	apply func(*runOptions) error
}

type runOptions struct {
	observer             ProgressObserver
	populationObserver   PopulationObserver
	distributionObserver DistributionObserver
	logger               Logger
	initialPopulation    [][]float64
	initialMean          []float64
	initialSigma         float64
	hasInitialMean       bool
}

// WithInitialPopulation seeds leading members of the first generation. The
// remaining members are sampled normally. Positions are copied when the
// option is constructed and again when it is applied.
func WithInitialPopulation(positions [][]float64) RunOption {
	snapshot := clonePositions(positions)

	return RunOption{apply: func(options *runOptions) error {
		options.initialPopulation = clonePositions(snapshot)

		return nil
	}}
}

// WithInitialMean overrides the starting mean and sigma for one run without
// mutating Config. The mean is copied at construction and application time.
func WithInitialMean(mean []float64, sigma float64) RunOption {
	snapshot := append([]float64(nil), mean...)

	return RunOption{apply: func(options *runOptions) error {
		options.initialMean = append([]float64(nil), snapshot...)
		options.initialSigma = sigma
		options.hasInitialMean = true

		return nil
	}}
}

// WithProgressObserver registers a lightweight iteration observer. Nil
// disables progress reporting.
func WithProgressObserver(observer ProgressObserver) RunOption {
	return RunOption{apply: func(options *runOptions) error {
		options.observer = observer

		return nil
	}}
}

// WithPopulationObserver registers an evaluated-population observer. Nil
// disables population reporting.
func WithPopulationObserver(observer PopulationObserver) RunOption {
	return RunOption{apply: func(options *runOptions) error {
		options.populationObserver = observer

		return nil
	}}
}

// WithDistributionObserver registers a sampling-distribution observer. Nil
// disables distribution reporting.
func WithDistributionObserver(observer DistributionObserver) RunOption {
	return RunOption{apply: func(options *runOptions) error {
		options.distributionObserver = observer

		return nil
	}}
}

// WithLogger registers a structured lifecycle logger. Nil disables logging.
func WithLogger(logger Logger) RunOption {
	return RunOption{apply: func(options *runOptions) error {
		options.logger = logger

		return nil
	}}
}

func resolveRunOptions(options []RunOption) (runOptions, error) {
	var resolved runOptions

	for index, option := range options {
		if option.apply == nil {
			return runOptions{}, fmt.Errorf("run option %d is invalid", index)
		}

		applyErr := option.apply(&resolved)
		if applyErr != nil {
			return runOptions{}, fmt.Errorf("apply run option %d: %w", index, applyErr)
		}
	}

	return resolved, nil
}

func validateRunOptions(config *Config, options runOptions) error {
	initialLimit := generationPopulationSize(config, 0)
	if len(options.initialPopulation) > initialLimit {
		return fmt.Errorf("initial population has %d positions, exceeds first generation size %d",
			len(options.initialPopulation), initialLimit)
	}

	for index, position := range options.initialPopulation {
		err := validateRunPosition("initial population", index, position, config)
		if err != nil {
			return err
		}
	}

	if !options.hasInitialMean {
		return nil
	}

	if len(options.initialMean) != config.ProblemSize {
		return fmt.Errorf("initial mean has dimension %d, want %d",
			len(options.initialMean), config.ProblemSize)
	}

	if !isFinite(options.initialSigma) || options.initialSigma <= 0 {
		return fmt.Errorf("initial mean sigma must be finite and positive, got %v",
			options.initialSigma)
	}

	for coordinate, value := range options.initialMean {
		if !isFinite(value) {
			return fmt.Errorf("initial mean coordinate %d must be finite, got %v", coordinate, value)
		}
	}

	return nil
}

func validateRunPosition(kind string, index int, position []float64, config *Config) error {
	if len(position) != config.ProblemSize {
		return fmt.Errorf("%s position %d has dimension %d, want %d",
			kind, index, len(position), config.ProblemSize)
	}

	for coordinate, value := range position {
		if !isFinite(value) {
			return fmt.Errorf("%s position %d coordinate %d must be finite, got %v",
				kind, index, coordinate, value)
		}

		lower, upper := coordinateBounds(config, coordinate)
		if value < lower || value > upper {
			return fmt.Errorf("%s position %d coordinate %d is outside bounds [%v, %v]: %v",
				kind, index, coordinate, lower, upper, value)
		}
	}

	return nil
}

func clonePositions(positions [][]float64) [][]float64 {
	if positions == nil {
		return nil
	}

	cloned := make([][]float64, len(positions))
	for index, position := range positions {
		cloned[index] = append([]float64(nil), position...)
	}

	return cloned
}

func cloneBest(best Best) Best {
	return Best{
		Position:            append([]float64(nil), best.Position...),
		Cost:                best.Cost,
		ConstraintViolation: best.ConstraintViolation,
	}
}

func notifyLifecycle(
	ctx context.Context,
	options runOptions,
	iteration, evaluations int,
	best Best,
	population []candidate,
	state *strategyState,
) {
	if options.observer != nil {
		progress := Progress{
			Best:            cloneBest(best),
			Iteration:       iteration,
			EvaluationCount: evaluations,
		}

		notifyContained(ctx, options.logger, "progress_observer", func() {
			options.observer(progress)
		})
	}

	if options.populationObserver != nil {
		snapshot := populationSnapshot(best, population, iteration, evaluations)

		notifyContained(ctx, options.logger, "population_observer", func() {
			options.populationObserver(snapshot)
		})
	}

	if options.distributionObserver != nil {
		snapshot := distributionSnapshot(state, iteration, evaluations)

		notifyContained(ctx, options.logger, "distribution_observer", func() {
			options.distributionObserver(snapshot)
		})
	}
}

// notifyContained invokes one caller-supplied observer and contains a panic it
// raises. Reporting is a side channel: a faulty observer must not destroy an
// in-progress run and the best solution found so far with it.
func notifyContained(ctx context.Context, logger Logger, source string, notify func()) {
	defer func() {
		recovered := recover()
		if recovered != nil {
			logObserverPanic(ctx, logger, source, recovered)
		}
	}()

	notify()
}

func populationSnapshot(
	best Best,
	population []candidate,
	iteration, evaluations int,
) PopulationSnapshot {
	candidates := make([]PopulationCandidate, len(population))
	for index, current := range population {
		candidates[index] = PopulationCandidate{
			Position:            append([]float64(nil), current.evaluatedPosition()...),
			Cost:                current.cost,
			ConstraintViolation: current.constraintViolation,
		}
	}

	return PopulationSnapshot{
		Population:      candidates,
		Best:            cloneBest(best),
		Iteration:       iteration,
		EvaluationCount: evaluations,
	}
}

// distributionSnapshot reports the distribution described by state. The caller
// supplies the state whose B and D belong to the reported iteration; see
// optimizationRun.reportedState.
func distributionSnapshot(
	state *strategyState,
	iteration, evaluations int,
) DistributionSnapshot {
	var blocks []BlockDistributionSnapshot

	eigenvectors := state.b

	switch {
	case state.mode == CovarianceBlock:
		blocks = blockDistributionSnapshots(state)
	case state.reportsBlocks:
		blocks = canonicalizedBlockSnapshots(state)
		eigenvectors = nil
	case state.mode == CovarianceSeparable:
		eigenvectors = separableEigenvectors(len(state.m))
	}

	return DistributionSnapshot{
		Mean:            append([]float64(nil), state.m...),
		Eigenvalues:     append([]float64(nil), state.d...),
		Eigenvectors:    clonePositions(eigenvectors),
		Blocks:          blocks,
		Sigma:           state.sigma,
		ConditionNumber: covarianceConditionNumber(state.d),
		Iteration:       iteration,
		EvaluationCount: evaluations,
	}
}
