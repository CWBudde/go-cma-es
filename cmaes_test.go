package cmaes

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"reflect"
	"sync/atomic"
	"testing"
)

func optimizationConfig(problemSize int, seed int64, objective ObjectiveFunction) *Config {
	config := NewDefaultConfig(problemSize)
	config.ObjectiveFunc = objective
	config.Seed = &seed
	config.LowerBound = -10
	config.UpperBound = 10

	return config
}

func sphere(position []float64) float64 {
	var cost float64
	for _, value := range position {
		cost += value * value
	}

	return cost
}

func TestOptimizeSphere(t *testing.T) {
	config := optimizationConfig(10, 11, sphere)
	config.InitialMean = filledVector(10, 3)
	config.InitialSigma = 1
	config.MaxIterations = 500

	result, err := Optimize(config)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	if result.GlobalBest.Cost >= 1e-10 {
		t.Errorf("sphere cost = %g, want < 1e-10", result.GlobalBest.Cost)
	}
}

func TestOptimizeConditionedEllipsoid(t *testing.T) {
	const condition = 1e6

	objective := func(position []float64) float64 {
		var cost float64

		for index, value := range position {
			exponent := float64(index) / float64(len(position)-1)
			cost += math.Pow(condition, exponent) * value * value
		}

		return cost
	}

	config := optimizationConfig(10, 22, objective)
	config.InitialMean = filledVector(10, 2)
	config.InitialSigma = 0.7
	config.MaxIterations = 1500

	result, err := Optimize(config)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	if result.GlobalBest.Cost >= 1e-10 {
		t.Errorf("condition-1e6 ellipsoid cost = %g, want < 1e-10", result.GlobalBest.Cost)
	}
}

func TestOptimizeRosenbrock(t *testing.T) {
	objective := func(position []float64) float64 {
		var cost float64

		for index := range len(position) - 1 {
			difference := position[index+1] - position[index]*position[index]
			cost += 100*difference*difference + (1-position[index])*(1-position[index])
		}

		return cost
	}

	config := optimizationConfig(10, 33, objective)
	config.InitialSigma = 0.5
	config.MaxIterations = 2500

	result, err := Optimize(config)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	if result.GlobalBest.Cost >= 1e-6 {
		t.Errorf("Rosenbrock cost = %g, want < 1e-6", result.GlobalBest.Cost)
	}
}

func TestSeededRunsAreBitIdentical(t *testing.T) {
	first, err := Optimize(optimizationConfig(6, 44, sphere))
	if err != nil {
		t.Fatalf("first Optimize: %v", err)
	}

	second, err := Optimize(optimizationConfig(6, 44, sphere))
	if err != nil {
		t.Fatalf("second Optimize: %v", err)
	}

	if !reflect.DeepEqual(first, second) {
		t.Fatalf("same-seed results differ:\nfirst  = %#v\nsecond = %#v", first, second)
	}
}

func TestParallelEvaluationMatchesSerial(t *testing.T) {
	serialConfig := optimizationConfig(8, 55, sphere)
	serialConfig.MaxIterations = 250

	serial, err := Optimize(serialConfig)
	if err != nil {
		t.Fatalf("serial Optimize: %v", err)
	}

	parallelConfig := optimizationConfig(8, 55, sphere)
	parallelConfig.MaxIterations = serialConfig.MaxIterations
	parallelConfig.EnableParallel = true
	parallelConfig.MaxWorkers = 3

	parallel, err := Optimize(parallelConfig)
	if err != nil {
		t.Fatalf("parallel Optimize: %v", err)
	}

	if !reflect.DeepEqual(serial, parallel) {
		t.Fatalf("serial and parallel results differ:\nserial   = %#v\nparallel = %#v", serial, parallel)
	}
}

func TestFunctionEvaluationAccounting(t *testing.T) {
	var calls atomic.Int64

	config := optimizationConfig(4, 66, func(position []float64) float64 {
		calls.Add(1)

		return sphere(position)
	})
	config.MaxIterations = 100
	config.MaxEvaluations = 37

	result, err := Optimize(config)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	if result.FuncEvalCount != int(calls.Load()) || result.FuncEvalCount != config.MaxEvaluations {
		t.Errorf("evaluation counts = (result %d, objective %d), want %d",
			result.FuncEvalCount, calls.Load(), config.MaxEvaluations)
	}

	if result.TerminationReason != TerminationMaxEvaluations {
		t.Errorf("termination = %q, want %q", result.TerminationReason, TerminationMaxEvaluations)
	}
}

func TestInverseSquareRootProduct(t *testing.T) {
	rootHalf := math.Sqrt(0.5)
	b := [][]float64{{rootHalf, -rootHalf}, {rootHalf, rootHalf}}
	d := []float64{2, 5}
	z := []float64{3, -4}
	y := transformNormal(b, d, z)
	got := inverseSquareRootProduct(b, d, y)
	want := matrixVectorProduct(b, z)

	assertVectorClose(t, got, want, 1e-14)
}

func TestHeavisideCorrectionGatesPathAndCompensatesCovariance(t *testing.T) {
	parameters := strategyParameters{
		weights: []float64{1},
		muEff:   1,
		cc:      0.4,
		c1:      0.2,
		cmu:     0.1,
	}
	path := []float64{2, -1}
	weightedStep := []float64{10, 10}
	updateCovariancePath(path, weightedStep, false, parameters)
	assertVectorClose(t, path, []float64{1.2, -0.6}, 0)

	covariance := [][]float64{{2, 0.5}, {0.5, 3}}
	population := []candidate{{y: []float64{1, 2}}}
	oldScale := 1 - parameters.c1 - parameters.cmu +
		parameters.c1*parameters.cc*(2-parameters.cc)
	want := [][]float64{
		{oldScale*2 + parameters.c1*1.2*1.2 + parameters.cmu, oldScale*0.5 + parameters.c1*1.2*-0.6 + 2*parameters.cmu},
		{oldScale*0.5 + parameters.c1*1.2*-0.6 + 2*parameters.cmu, oldScale*3 + parameters.c1*0.6*0.6 + 4*parameters.cmu},
	}

	updateCovariance(covariance, path, population, false, parameters)
	assertMatrixClose(t, covariance, want, 1e-15)
}

func TestLazyEigendecompositionUsesPublishedStrictTrigger(t *testing.T) {
	config := optimizationConfig(2, 1, sphere)
	parameters := deriveStrategyParameters(config)
	state := newStrategyState(config)
	state.c = [][]float64{{2, 0.5}, {0.5, 1}}
	limit := float64(config.Lambda) /
		(10 * float64(config.ProblemSize) * (parameters.c1 + parameters.cmu))
	atLimit := int(math.Floor(limit))

	if refreshEigensystemIfStale(state, atLimit, config.Lambda, parameters) {
		t.Fatal("eigendecomposition refreshed before the strict staleness trigger")
	}

	firstStale := atLimit + 1
	if !refreshEigensystemIfStale(state, firstStale, config.Lambda, parameters) {
		t.Fatal("eigendecomposition did not refresh after the staleness trigger")
	}

	if state.eigenEval != firstStale || state.d[0] == 1 || state.d[1] == 1 {
		t.Errorf("refreshed state = eigenEval %d, D %v", state.eigenEval, state.d)
	}
}

func TestSamplingAndRecombinationEquations(t *testing.T) {
	state := &strategyState{
		m:     []float64{1, -2},
		b:     identityMatrix(2),
		d:     []float64{2, 3},
		sigma: 0.5,
	}
	population := samplePopulation(state, 2, rand.New(rand.NewSource(7)))

	for _, current := range population {
		for coordinate := range current.x {
			wantY := state.d[coordinate] * current.z[coordinate]
			wantX := state.m[coordinate] + state.sigma*wantY

			if current.y[coordinate] != wantY || current.x[coordinate] != wantX {
				t.Errorf("sample coordinate %d = (x %v, y %v), want (%v, %v)",
					coordinate, current.x[coordinate], current.y[coordinate], wantX, wantY)
			}
		}
	}

	weightedX, weightedY := recombine(population, []float64{0.75, 0.25})
	for coordinate := range weightedX {
		wantX := 0.75*population[0].x[coordinate] + 0.25*population[1].x[coordinate]
		wantY := 0.75*population[0].y[coordinate] + 0.25*population[1].y[coordinate]

		if weightedX[coordinate] != wantX || weightedY[coordinate] != wantY {
			t.Errorf("recombined coordinate %d = (%v, %v), want (%v, %v)",
				coordinate, weightedX[coordinate], weightedY[coordinate], wantX, wantY)
		}
	}
}

func TestOptimizeContextValidationAndCancellation(t *testing.T) {
	config := optimizationConfig(2, 77, sphere)

	result, err := OptimizeContext(nil, config) //nolint:staticcheck // Intentionally verify nil-context rejection.
	if result != nil || err == nil {
		t.Fatalf("nil context returned (%v, %v), want (nil, error)", result, err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result, err = OptimizeContext(ctx, config)
	if result != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled context returned (%v, %v), want (nil, context.Canceled)", result, err)
	}

	result, err = OptimizeContext(context.Background(), config, RunOption{})
	if result != nil || err == nil {
		t.Fatalf("invalid run option returned (%v, %v), want (nil, error)", result, err)
	}

	separable := NewSeparableConfig(2)
	separable.ObjectiveFunc = sphere
	separable.LowerBound = -1
	separable.UpperBound = 1

	result, err = Optimize(separable)
	if result != nil || err == nil {
		t.Fatalf("separable mode returned (%v, %v), want (nil, not implemented error)", result, err)
	}

	constrained := optimizationConfig(2, 78, sphere)
	constrained.Constraints = &ConstraintConfig{}
	constrained.MaxIterations = 1

	result, err = Optimize(constrained)
	if result == nil || err != nil {
		t.Fatalf("constraints returned (%v, %v), want (result, nil)", result, err)
	}

	withConvergence := optimizationConfig(2, 79, sphere)
	withConvergence.Convergence = &ConvergenceConfig{}
	withConvergence.MaxIterations = 1

	result, err = Optimize(withConvergence)
	if result == nil || err != nil {
		t.Fatalf("convergence config returned (%v, %v), want (result, nil)", result, err)
	}
}

func TestGeneratedRandomSourceIsRecorded(t *testing.T) {
	seed := int64(88)
	config := optimizationConfig(2, seed, sphere)
	config.MaxIterations = 1

	result, err := Optimize(config)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	if config.Rand == nil || !result.SeedKnown || result.Seed != seed {
		t.Errorf("random metadata = (Rand %p, Seed %d, known %v), want non-nil, %d, true",
			config.Rand, result.Seed, result.SeedKnown, seed)
	}

	direct := optimizationConfig(2, 0, sphere)
	direct.Seed = nil
	direct.Rand = rand.New(rand.NewSource(99))
	direct.MaxIterations = 1

	directResult, err := Optimize(direct)
	if err != nil {
		t.Fatalf("Optimize with direct Rand: %v", err)
	}

	if directResult.SeedKnown || directResult.Seed != 0 {
		t.Errorf("direct Rand metadata = (%d, %v), want (0, false)",
			directResult.Seed, directResult.SeedKnown)
	}
}

func filledVector(size int, value float64) []float64 {
	vector := make([]float64, size)
	for index := range vector {
		vector[index] = value
	}

	return vector
}

func assertVectorClose(t *testing.T, got, want []float64, tolerance float64) {
	t.Helper()

	for index := range want {
		if math.Abs(got[index]-want[index]) > tolerance {
			t.Errorf("vector[%d] = %.17g, want %.17g (tolerance %g)",
				index, got[index], want[index], tolerance)
		}
	}
}

func assertMatrixClose(t *testing.T, got, want [][]float64, tolerance float64) {
	t.Helper()

	for row := range want {
		assertVectorClose(t, got[row], want[row], tolerance)
	}
}
