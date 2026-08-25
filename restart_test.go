package cmaes

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestIPOPDoublesPopulationAndConsumesSharedBudget(t *testing.T) {
	t.Parallel()

	config := restartTestConfig(2, 17)
	config.Lambda = 4
	config.Mu = 2
	config.MaxIterations = 1
	config.MaxEvaluations = 28
	config.Convergence = nil
	originalMean := append([]float64(nil), config.InitialMean...)

	result, err := OptimizeWithRestarts(config, RestartIPOP)
	if err != nil {
		t.Fatalf("OptimizeWithRestarts: %v", err)
	}

	if result.FuncEvalCount != config.MaxEvaluations {
		t.Fatalf("evaluations = %d, want %d", result.FuncEvalCount, config.MaxEvaluations)
	}

	if result.TerminationReason != TerminationMaxEvaluations {
		t.Errorf("termination = %q, want %q", result.TerminationReason,
			TerminationMaxEvaluations)
	}

	wantLambda := []int{4, 8, 16}
	if len(result.Restarts) != len(wantLambda) {
		t.Fatalf("runs = %d, want %d", len(result.Restarts), len(wantLambda))
	}

	for index, record := range result.Restarts {
		if record.Restart != index {
			t.Errorf("record %d restart = %d", index, record.Restart)
		}

		if record.Lambda != wantLambda[index] {
			t.Errorf("record %d lambda = %d, want %d", index, record.Lambda,
				wantLambda[index])
		}

		if record.Evaluations != record.Lambda {
			t.Errorf("record %d evaluations = %d, want %d", index,
				record.Evaluations, record.Lambda)
		}

		if record.Regime != RestartRegimeLarge {
			t.Errorf("record %d regime = %q, want large", index, record.Regime)
		}
	}

	if !reflect.DeepEqual(config.InitialMean, originalMean) || config.Lambda != 4 ||
		config.Mu != 2 || config.MaxEvaluations != 28 {
		t.Errorf("caller config was mutated: %+v", config)
	}
}

func TestIPOPRestartsAfterConvergenceCriterion(t *testing.T) {
	t.Parallel()

	config := restartTestConfig(2, 23)
	config.ObjectiveFunc = func([]float64) float64 { return 1 }
	config.Lambda = 4
	config.Mu = 2
	config.MaxIterations = 50
	config.MaxEvaluations = 245
	config.Convergence = &ConvergenceConfig{TolFun: 1e-12}

	result, err := OptimizeWithRestarts(config, RestartIPOP)
	if err != nil {
		t.Fatalf("OptimizeWithRestarts: %v", err)
	}

	if len(result.Restarts) != 3 {
		t.Fatalf("runs = %d, want 3", len(result.Restarts))
	}

	for index, record := range result.Restarts[:2] {
		if record.TerminationReason != TerminationTolFun {
			t.Errorf("record %d termination = %q, want %q", index,
				record.TerminationReason, TerminationTolFun)
		}
	}

	tail := result.Restarts[2]
	if tail.Evaluations != 1 || tail.Iterations != 0 ||
		tail.TerminationReason != TerminationMaxEvaluations {
		t.Errorf("tail = %+v, want one evaluation and no completed iteration", tail)
	}

	if result.FuncEvalCount != config.MaxEvaluations {
		t.Errorf("evaluations = %d, want exact budget %d", result.FuncEvalCount,
			config.MaxEvaluations)
	}
}

func TestBIPOPInterleavesRegimesDeterministically(t *testing.T) {
	t.Parallel()

	serial := restartTestConfig(3, 101)
	serial.Lambda = 4
	serial.Mu = 2
	serial.MaxIterations = 2
	serial.MaxEvaluations = 240
	serial.Convergence = nil

	parallel := *serial
	parallel.InitialMean = append([]float64(nil), serial.InitialMean...)
	parallel.EnableParallel = true
	parallel.MaxWorkers = 3

	serialResult, err := OptimizeWithRestarts(serial, RestartBIPOP)
	if err != nil {
		t.Fatalf("serial BIPOP: %v", err)
	}

	parallelResult, err := OptimizeWithRestarts(&parallel, RestartBIPOP)
	if err != nil {
		t.Fatalf("parallel BIPOP: %v", err)
	}

	if !reflect.DeepEqual(serialResult, parallelResult) {
		t.Fatalf("serial and parallel restart results differ:\nserial:   %#v\nparallel: %#v",
			serialResult, parallelResult)
	}

	if serialResult.FuncEvalCount != serial.MaxEvaluations {
		t.Errorf("evaluations = %d, want exact budget %d", serialResult.FuncEvalCount,
			serial.MaxEvaluations)
	}

	largeLambda := make([]int, 0)
	sawSmall := false
	largeSpend, smallSpend := 0, 0

	for index, record := range serialResult.Restarts {
		if index > 0 {
			wantRegime := RestartRegimeLarge
			if largeSpend > smallSpend {
				wantRegime = RestartRegimeSmall
			}

			if record.Regime != wantRegime {
				t.Errorf("record %d regime = %q, want %q from spend large=%d small=%d",
					index, record.Regime, wantRegime, largeSpend, smallSpend)
			}
		}

		switch record.Regime {
		case RestartRegimeLarge:
			largeLambda = append(largeLambda, record.Lambda)
			largeSpend += record.Evaluations
		case RestartRegimeSmall:
			sawSmall = true
			smallSpend += record.Evaluations
		default:
			t.Fatalf("record %d has unknown regime %q", index, record.Regime)
		}
	}

	if !sawSmall || len(largeLambda) < 2 {
		t.Fatalf("regimes = %#v, want small runs and at least two large runs",
			serialResult.Restarts)
	}

	for index := 1; index < len(largeLambda); index++ {
		if largeLambda[index] != 2*largeLambda[index-1] {
			t.Errorf("large lambda[%d] = %d, want %d", index, largeLambda[index],
				2*largeLambda[index-1])
		}
	}
}

func TestRestartOptimizationStopsAtTarget(t *testing.T) {
	t.Parallel()

	config := restartTestConfig(2, 31)
	target := 0.0
	config.ObjectiveFunc = func([]float64) float64 { return target }
	config.MaxEvaluations = config.Lambda
	config.Convergence = &ConvergenceConfig{TargetCost: &target}

	result, err := OptimizeWithRestarts(config, RestartIPOP)
	if err != nil {
		t.Fatalf("OptimizeWithRestarts: %v", err)
	}

	if len(result.Restarts) != 1 {
		t.Fatalf("runs = %d, want 1", len(result.Restarts))
	}

	if result.TerminationReason != TerminationTargetCost {
		t.Errorf("termination = %q, want %q", result.TerminationReason,
			TerminationTargetCost)
	}

	if result.FuncEvalCount != config.Lambda {
		t.Errorf("evaluations = %d, want first generation %d", result.FuncEvalCount,
			config.Lambda)
	}
}

// TestBIPOPFindsRastriginGlobalOptimumWhereSingleRunDoesNot demonstrates the
// property restarts exist for, so it fixes a seed on which the schedule
// succeeds rather than asserting a success rate. Restarting 10-D Rastrigin from
// a single basin is a stochastic search: roughly one seed in ten reaches the
// global optimum within this budget, while a single run reaches none.
func TestBIPOPFindsRastriginGlobalOptimumWhereSingleRunDoesNot(t *testing.T) {
	t.Parallel()

	const budget = 80_000

	config := NewDefaultConfig(10)
	config.ObjectiveFunc = Rastrigin
	config.LowerBound = -5.12
	config.UpperBound = 5.12

	for index := range config.InitialMean {
		config.InitialMean[index] = 2.5
	}

	config.InitialSigma = 1.5
	config.MaxEvaluations = budget
	seed := int64(15)
	config.Seed = &seed

	single, err := Optimize(config)
	if err != nil {
		t.Fatalf("single run: %v", err)
	}

	restarted, err := OptimizeWithRestarts(config, RestartBIPOP)
	if err != nil {
		t.Fatalf("BIPOP: %v", err)
	}

	if single.GlobalBest.Cost < 1 {
		t.Fatalf("single-run cost = %g, want a non-global local basin",
			single.GlobalBest.Cost)
	}

	if restarted.GlobalBest.Cost > 1e-10 {
		t.Errorf("BIPOP cost = %g, want the Rastrigin global optimum",
			restarted.GlobalBest.Cost)
	}

	if restarted.FuncEvalCount != budget {
		t.Errorf("BIPOP evaluations = %d, want exact budget %d",
			restarted.FuncEvalCount, budget)
	}
}

func TestRestartValidation(t *testing.T) {
	t.Parallel()

	valid := restartTestConfig(2, 7)
	valid.MaxEvaluations = 20

	_, err := OptimizeWithRestartsContext(nil, valid, RestartIPOP) //nolint:staticcheck // Intentionally verify nil-context rejection.
	if err == nil || !strings.Contains(err.Error(), "context must not be nil") {
		t.Errorf("nil context error = %v", err)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = OptimizeWithRestartsContext(canceled, valid, RestartIPOP)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("canceled context error = %v, want context.Canceled", err)
	}

	tests := []struct {
		name     string
		config   *Config
		strategy RestartStrategy
		wantErr  string
	}{
		{
			name: "nil config", strategy: RestartIPOP,
			wantErr: "config must not be nil",
		},
		{name: "missing budget", config: func() *Config {
			config := restartTestConfig(2, 7)
			config.MaxEvaluations = 0

			return config
		}(), strategy: RestartIPOP, wantErr: "max_evaluations must be positive"},
		{
			name: "unknown strategy", config: valid,
			strategy: "random", wantErr: "unknown restart strategy",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			_, err := OptimizeWithRestartsContext(
				context.Background(), test.config, test.strategy,
			)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %v, want containing %q", err, test.wantErr)
			}
		})
	}
}

func restartTestConfig(problemSize int, seed int64) *Config {
	config := NewDefaultConfig(problemSize)
	config.ObjectiveFunc = func(position []float64) float64 {
		cost := 0.0
		for _, coordinate := range position {
			cost += coordinate * coordinate
		}

		return cost
	}
	config.LowerBound = -5
	config.UpperBound = 5
	config.InitialSigma = 1
	config.Seed = &seed

	return config
}
