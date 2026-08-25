package cmaes

import (
	"context"
	"log/slog"
	"math"
	"reflect"
	"testing"
)

type lifecycleLog struct {
	messages []string
}

func (logger *lifecycleLog) Log(
	_ context.Context,
	_ slog.Level,
	message string,
	_ ...any,
) {
	logger.messages = append(logger.messages, message)
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
