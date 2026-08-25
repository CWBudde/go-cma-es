package cmaes

import (
	"context"
	"errors"
	"log/slog"
	"math"
	"math/rand"
	"reflect"
	"sync"
	"testing"
)

type lifecycleLog struct {
	messages []string
	levels   []slog.Level
	mutex    sync.Mutex
}

func (logger *lifecycleLog) Log(
	_ context.Context,
	level slog.Level,
	message string,
	_ ...any,
) {
	logger.mutex.Lock()
	defer logger.mutex.Unlock()

	logger.messages = append(logger.messages, message)
	logger.levels = append(logger.levels, level)
}

func (logger *lifecycleLog) count(message string) int {
	logger.mutex.Lock()
	defer logger.mutex.Unlock()

	total := 0

	for _, recorded := range logger.messages {
		if recorded == message {
			total++
		}
	}

	return total
}

func TestLifecycleObserversHistoriesAndInitialPopulation(t *testing.T) {
	config := optimizationConfig(2, 91, sphere)
	config.Convergence = nil
	config.MaxIterations = 2
	initial := [][]float64{{0, 0}}
	logger := &lifecycleLog{}

	var (
		progress      []Progress
		populations   []PopulationSnapshot
		distributions []DistributionSnapshot
	)

	result, err := OptimizeContext(
		context.Background(),
		config,
		WithInitialPopulation(initial),
		WithProgressObserver(func(update Progress) {
			progress = append(progress, update)
			update.Best.Position[0] = 999
		}),
		WithPopulationObserver(func(snapshot PopulationSnapshot) {
			populations = append(populations, snapshot)
		}),
		WithDistributionObserver(func(snapshot DistributionSnapshot) {
			distributions = append(distributions, snapshot)
			if len(distributions) == 1 {
				snapshot.Mean[0] = 999
				snapshot.Eigenvectors[0][0] = 999
			}
		}),
		WithLogger(logger),
	)
	if err != nil {
		t.Fatalf("OptimizeContext: %v", err)
	}

	if result.IterationCount != 2 || len(result.ConvergenceCurve) != 2 ||
		len(result.SigmaHistory) != 2 || len(result.ConditionNumberHistory) != 2 {
		t.Fatalf("history lengths = (%d, %d, %d), iterations %d, want all 2",
			len(result.ConvergenceCurve), len(result.SigmaHistory),
			len(result.ConditionNumberHistory), result.IterationCount)
	}

	if len(progress) != 2 || len(populations) != 2 || len(distributions) != 2 {
		t.Fatalf("observer calls = (%d, %d, %d), want all 2",
			len(progress), len(populations), len(distributions))
	}

	if result.GlobalBest.Cost != 0 || result.GlobalBest.Position[0] == 999 {
		t.Errorf("best = %+v, want untouched seeded origin", result.GlobalBest)
	}

	foundInitial := false

	for _, current := range populations[0].Population {
		if reflect.DeepEqual(current.Position, []float64{0, 0}) {
			foundInitial = true
		}
	}

	if !foundInitial {
		t.Errorf("first population does not contain seeded position: %+v", populations[0])
	}

	if distributions[1].Mean[0] == 999 || distributions[1].Eigenvectors[0][0] == 999 {
		t.Error("distribution observer mutation leaked into a later snapshot")
	}

	wantLogs := []string{
		"optimization started",
		"optimization iteration completed",
		"optimization iteration completed",
		"optimization completed",
	}
	if !reflect.DeepEqual(logger.messages, wantLogs) {
		t.Errorf("log messages = %v, want %v", logger.messages, wantLogs)
	}

	// Per-iteration events belong on debug: an info sink must not receive one
	// line per generation for the whole iteration budget.
	wantLevels := []slog.Level{
		slog.LevelInfo, slog.LevelDebug, slog.LevelDebug, slog.LevelInfo,
	}
	if !reflect.DeepEqual(logger.levels, wantLevels) {
		t.Errorf("log levels = %v, want %v", logger.levels, wantLevels)
	}
}

// reconstructCovariance rebuilds C from an eigensystem as B*diag(D^2)*B^T.
func reconstructCovariance(eigenvectors [][]float64, axisScales []float64) [][]float64 {
	size := len(axisScales)
	covariance := make([][]float64, size)

	for row := range covariance {
		covariance[row] = make([]float64, size)
		for column := range covariance[row] {
			for axis, scale := range axisScales {
				covariance[row][column] +=
					eigenvectors[row][axis] * scale * scale * eigenvectors[column][axis]
			}
		}
	}

	return covariance
}

func TestDistributionSnapshotDescribesTheSameIteration(t *testing.T) {
	const generations = 12

	config := optimizationConfig(4, 94, sphere)
	config.Convergence = nil
	config.MaxIterations = generations

	var snapshots []DistributionSnapshot

	options, err := resolveRunOptions([]RunOption{
		WithDistributionObserver(func(snapshot DistributionSnapshot) {
			snapshots = append(snapshots, snapshot)
		}),
	})
	if err != nil {
		t.Fatalf("resolveRunOptions: %v", err)
	}

	run := newOptimizationRun(config, options)
	rng := rand.New(rand.NewSource(94))

	for generation := range generations {
		stop, generationErr := run.executeGeneration(context.Background(), generation, rng)
		if generationErr != nil {
			t.Fatalf("generation %d: %v", generation, generationErr)
		}

		if stop {
			t.Fatalf("generation %d stopped early", generation)
		}

		snapshot := snapshots[len(snapshots)-1]

		assertMatrixClose(t,
			reconstructCovariance(snapshot.Eigenvectors, snapshot.Eigenvalues),
			run.state.c, 1e-9)
		assertVectorClose(t, snapshot.Mean, run.state.m, 0)

		if snapshot.Sigma != run.state.sigma {
			t.Fatalf("generation %d snapshot sigma = %v, want %v",
				generation, snapshot.Sigma, run.state.sigma)
		}

		if snapshot.ConditionNumber != covarianceConditionNumber(snapshot.Eigenvalues) {
			t.Fatalf("generation %d condition number = %v, does not match its own axis scales",
				generation, snapshot.ConditionNumber)
		}

		if run.conditionHistory[generation] != snapshot.ConditionNumber {
			t.Fatalf("generation %d condition history = %v, snapshot = %v",
				generation, run.conditionHistory[generation], snapshot.ConditionNumber)
		}
	}
}

func TestReportedEigensystemIsReusedByTheLazyRefresh(t *testing.T) {
	config := optimizationConfig(4, 95, sphere)
	config.Convergence = nil

	run := newOptimizationRun(config, runOptions{})
	rng := rand.New(rand.NewSource(95))

	for generation := range 5 {
		_, err := run.executeGeneration(context.Background(), generation, rng)
		if err != nil {
			t.Fatalf("generation %d: %v", generation, err)
		}
	}

	// The cached decomposition is what the lazy refresh will install at the top
	// of the next generation, so it has to equal a decomposition taken there.
	values, vectors := symmetricEigendecomposition(run.state.c)
	axisScales := make([]float64, len(values))

	for index, value := range values {
		axisScales[index] = math.Sqrt(value)
	}

	assertVectorClose(t, run.state.pendingD, axisScales, 0)
	assertMatrixClose(t, run.state.pendingB, vectors, 0)
}

func TestObserverAndLoggerPanicsAreContained(t *testing.T) {
	config := optimizationConfig(3, 96, sphere)
	config.Convergence = nil
	config.MaxIterations = 3
	logger := &lifecycleLog{}

	result, err := OptimizeContext(
		context.Background(),
		config,
		WithProgressObserver(func(Progress) { panic("progress") }),
		WithPopulationObserver(func(PopulationSnapshot) { panic("population") }),
		WithDistributionObserver(func(DistributionSnapshot) { panic("distribution") }),
		WithLogger(logger),
	)
	if err != nil {
		t.Fatalf("OptimizeContext: %v", err)
	}

	if result.IterationCount != config.MaxIterations {
		t.Errorf("iterations = %d, want %d: a panicking observer aborted the run",
			result.IterationCount, config.MaxIterations)
	}

	if len(result.GlobalBest.Position) != config.ProblemSize {
		t.Errorf("best = %+v, want the best found before the panics", result.GlobalBest)
	}

	wantPanics := 3 * config.MaxIterations
	if got := logger.count("optimization observer panicked"); got != wantPanics {
		t.Errorf("reported observer panics = %d, want %d", got, wantPanics)
	}
}

func TestPanickingLoggerDoesNotAbortTheRun(t *testing.T) {
	config := optimizationConfig(3, 97, sphere)
	config.Convergence = nil
	config.MaxIterations = 3

	result, err := OptimizeContext(
		context.Background(),
		config,
		WithLogger(loggerFunc(func(context.Context, slog.Level, string, ...any) {
			panic("logger")
		})),
	)
	if err != nil {
		t.Fatalf("OptimizeContext: %v", err)
	}

	if result.IterationCount != config.MaxIterations {
		t.Errorf("iterations = %d, want %d: a panicking logger aborted the run",
			result.IterationCount, config.MaxIterations)
	}
}

type loggerFunc func(ctx context.Context, level slog.Level, message string, args ...any)

func (log loggerFunc) Log(ctx context.Context, level slog.Level, message string, args ...any) {
	log(ctx, level, message, args...)
}

func TestFailedRunLogsATerminalEvent(t *testing.T) {
	logger := &lifecycleLog{}
	logOptimizationFailed(context.Background(), logger, errors.New("objective backend down"))

	wantMessages := []string{"optimization failed"}
	wantLevels := []slog.Level{slog.LevelError}

	if !reflect.DeepEqual(logger.messages, wantMessages) ||
		!reflect.DeepEqual(logger.levels, wantLevels) {
		t.Errorf("failure log = (%v, %v), want (%v, %v)",
			logger.messages, logger.levels, wantMessages, wantLevels)
	}
}

func TestRunOptionsSnapshotAndValidateSeeds(t *testing.T) {
	positions := [][]float64{{1, 2}}
	option := WithInitialPopulation(positions)
	positions[0][0] = 99

	resolved, err := resolveRunOptions([]RunOption{option, WithInitialMean([]float64{2, 3}, 0.5)})
	if err != nil {
		t.Fatalf("resolveRunOptions: %v", err)
	}

	if !reflect.DeepEqual(resolved.initialPopulation, [][]float64{{1, 2}}) {
		t.Errorf("initial population = %v, want construction-time snapshot", resolved.initialPopulation)
	}

	config := optimizationConfig(2, 1, sphere)

	err = validateRunOptions(config, resolved)
	if err != nil {
		t.Fatalf("validateRunOptions: %v", err)
	}

	tests := []RunOption{
		WithInitialPopulation(make([][]float64, config.Lambda+1)),
		WithInitialPopulation([][]float64{{1}}),
		WithInitialPopulation([][]float64{{math.NaN(), 0}}),
		WithInitialPopulation([][]float64{{11, 0}}),
		WithInitialMean([]float64{0}, 1),
		WithInitialMean([]float64{0, 0}, 0),
		WithInitialMean([]float64{0, math.Inf(1)}, 1),
	}

	for index, invalid := range tests {
		resolved, resolveErr := resolveRunOptions([]RunOption{invalid})
		if resolveErr != nil {
			t.Fatalf("case %d resolve: %v", index, resolveErr)
		}

		validationErr := validateRunOptions(config, resolved)
		if validationErr == nil {
			t.Errorf("case %d validation = nil, want error", index)
		}
	}
}

func TestWithInitialMeanChangesRunWithoutMutatingConfig(t *testing.T) {
	config := optimizationConfig(2, 92, sphere)
	config.Convergence = nil
	config.MaxIterations = 1
	originalMean := append([]float64(nil), config.InitialMean...)
	originalSigma := config.InitialSigma

	result, err := OptimizeContext(
		context.Background(), config, WithInitialMean([]float64{4, 4}, 0.1),
	)
	if err != nil {
		t.Fatalf("OptimizeContext: %v", err)
	}

	if result.GlobalBest.Cost < 20 {
		t.Errorf("best cost = %v, seeded distribution did not start near [4,4]", result.GlobalBest.Cost)
	}

	if !reflect.DeepEqual(config.InitialMean, originalMean) || config.InitialSigma != originalSigma {
		t.Errorf("config distribution mutated to (%v, %v)", config.InitialMean, config.InitialSigma)
	}
}

func TestCancellationReturnsBestSoFar(t *testing.T) {
	config := optimizationConfig(3, 93, sphere)
	config.Convergence = nil
	config.MaxIterations = 20
	ctx, cancel := context.WithCancel(context.Background())

	result, err := OptimizeContext(ctx, config, WithProgressObserver(func(Progress) {
		cancel()
	}))
	if err != nil {
		t.Fatalf("OptimizeContext: %v", err)
	}

	if result == nil || result.TerminationReason != TerminationCancelled {
		t.Fatalf("result = %+v, want canceled partial result", result)
	}

	if result.IterationCount != 1 || len(result.GlobalBest.Position) != 3 ||
		len(result.ConvergenceCurve) != 1 {
		t.Errorf("partial result = %+v, want one completed iteration", result)
	}
}

func TestNilLifecycleOptionsDisableReporting(t *testing.T) {
	resolved, err := resolveRunOptions([]RunOption{
		WithInitialPopulation(nil),
		WithProgressObserver(nil),
		WithPopulationObserver(nil),
		WithDistributionObserver(nil),
		WithLogger(nil),
	})
	if err != nil {
		t.Fatalf("resolveRunOptions: %v", err)
	}

	if !reflect.DeepEqual(resolved, runOptions{}) {
		t.Errorf("resolved = %+v, want zero options", resolved)
	}
}
