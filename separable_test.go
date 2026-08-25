package cmaes

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func TestSeparableLearningRateCorrection(t *testing.T) {
	full := NewDefaultConfig(20)
	full.ActiveCMA = false
	separable := NewSeparableConfig(20)
	separable.ActiveCMA = false

	fullParameters := deriveStrategyParameters(full)
	separableParameters := deriveStrategyParameters(separable)
	want := math.Min(
		1-separableParameters.c1,
		fullParameters.cmu*float64(separable.ProblemSize+2)/3,
	)

	if separableParameters.cmu != want {
		t.Errorf("separable cmu = %.17g, want %.17g", separableParameters.cmu, want)
	}

	if separableParameters.c1 != fullParameters.c1 {
		t.Errorf("separable c1 = %g, want unchanged full rate %g",
			separableParameters.c1, fullParameters.c1)
	}
}

func TestSeparableStateHasOnlyLinearCovarianceStorage(t *testing.T) {
	config := NewSeparableConfig(200)
	state := newStrategyState(config)

	if state.c != nil || state.b != nil {
		t.Fatalf("separable dense state = (C %v, B %v), want both nil", state.c, state.b)
	}

	if len(state.diagonal) != config.ProblemSize || len(state.d) != config.ProblemSize {
		t.Errorf("diagonal storage = (%d, %d), want (%d, %d)",
			len(state.diagonal), len(state.d), config.ProblemSize, config.ProblemSize)
	}
}

func TestSeparableSamplingAndInverseTransform(t *testing.T) {
	config := NewSeparableConfig(3)
	state := newStrategyState(config)
	state.d = []float64{2, 3, 4}

	population := samplePopulation(state, 1, rand.New(rand.NewSource(611)))

	current := population[0]
	for coordinate := range current.y {
		want := state.d[coordinate] * current.z[coordinate]
		if current.y[coordinate] != want {
			t.Errorf("y[%d] = %g, want %g", coordinate, current.y[coordinate], want)
		}
	}

	got := inverseCovarianceSquareRootProduct(state, current.y)
	if !reflect.DeepEqual(got, current.z) {
		t.Errorf("inverse transformed step = %v, want z %v", got, current.z)
	}
}

func TestSeparableCovarianceUpdateEquation(t *testing.T) {
	state := &strategyState{
		pc:       []float64{2, 3},
		diagonal: []float64{4, 9},
		d:        []float64{2, 3},
		mode:     CovarianceSeparable,
	}
	population := []candidate{
		{y: []float64{1, 2}},
		{y: []float64{3, 4}},
	}
	parameters := strategyParameters{
		weights: []float64{0.75, 0.25},
		cc:      0.4,
		c1:      0.2,
		cmu:     0.1,
	}
	decay := 1 - parameters.c1 - parameters.cmu
	want := []float64{
		decay*4 + parameters.c1*4 + parameters.cmu*(0.75*1+0.25*9),
		decay*9 + parameters.c1*9 + parameters.cmu*(0.75*4+0.25*16),
	}

	updateSeparableCovariance(state, population, true, parameters)
	assertVectorClose(t, state.diagonal, want, 3e-15)
	assertVectorClose(t, state.d, []float64{math.Sqrt(want[0]), math.Sqrt(want[1])}, 3e-15)
}

func TestSeparableNeverRefreshesAnEigensystem(t *testing.T) {
	config := NewSeparableConfig(4)
	parameters := deriveStrategyParameters(config)
	state := newStrategyState(config)

	if refreshEigensystemIfStale(state, 1_000_000, config.Lambda, parameters) {
		t.Fatal("separable state reported an eigensystem refresh")
	}

	if state.eigenEval != 0 || state.b != nil {
		t.Errorf("separable eigensystem state = (eval %d, B %v), want (0, nil)",
			state.eigenEval, state.b)
	}
}

func TestSeparableDistributionSnapshotUsesCoordinateAxes(t *testing.T) {
	config := NewSeparableConfig(3)
	state := newStrategyState(config)
	state.m = []float64{1, 2, 3}
	state.diagonal = []float64{4, 9, 16}
	state.d = []float64{2, 3, 4}

	snapshot := distributionSnapshot(state, 5, 50)
	assertMatrixClose(t, snapshot.Eigenvectors, identityMatrix(3), 0)

	if !reflect.DeepEqual(snapshot.Eigenvalues, state.d) {
		t.Errorf("snapshot axis scales = %v, want %v", snapshot.Eigenvalues, state.d)
	}

	snapshot.Mean[0] = 99
	snapshot.Eigenvalues[0] = 99

	if state.m[0] == 99 || state.d[0] == 99 {
		t.Error("separable distribution snapshot aliases strategy state")
	}
}

func TestOptimizeSeparableIsDeterministicAndParallelEquivalent(t *testing.T) {
	serialConfig := NewSeparableConfig(20)
	serialConfig.ObjectiveFunc = sphere
	serialConfig.Seed = new(int64)
	*serialConfig.Seed = 612
	serialConfig.LowerBound = -10
	serialConfig.UpperBound = 10
	serialConfig.InitialMean = filledVector(20, 3)
	serialConfig.InitialSigma = 1
	serialConfig.MaxIterations = 400

	serial, err := Optimize(serialConfig)
	if err != nil {
		t.Fatalf("serial Optimize: %v", err)
	}

	parallelConfig := NewSeparableConfig(20)
	parallelConfig.ObjectiveFunc = sphere
	parallelConfig.Seed = new(int64)
	*parallelConfig.Seed = 612
	parallelConfig.LowerBound = -10
	parallelConfig.UpperBound = 10
	parallelConfig.InitialMean = filledVector(20, 3)
	parallelConfig.InitialSigma = 1
	parallelConfig.MaxIterations = 400
	parallelConfig.EnableParallel = true
	parallelConfig.MaxWorkers = 3

	parallel, err := Optimize(parallelConfig)
	if err != nil {
		t.Fatalf("parallel Optimize: %v", err)
	}

	if !reflect.DeepEqual(serial, parallel) {
		t.Fatalf("serial and parallel separable results differ")
	}

	if serial.GlobalBest.Cost >= 1e-10 {
		t.Errorf("separable sphere cost = %g, want < 1e-10", serial.GlobalBest.Cost)
	}
}

func BenchmarkCovarianceModesN200(b *testing.B) {
	for _, mode := range []CovarianceMode{CovarianceFull, CovarianceSeparable} {
		b.Run(string(mode), func(b *testing.B) {
			for range b.N {
				config := NewDefaultConfig(200)
				config.ObjectiveFunc = sphere
				config.Rand = rand.New(rand.NewSource(613))
				config.LowerBound = -10
				config.UpperBound = 10
				config.InitialMean = filledVector(200, 2)
				config.InitialSigma = 1
				config.CovarianceMode = mode
				config.Convergence = nil
				config.MaxIterations = 1

				_, err := Optimize(config)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestSeparableCovarianceUpdateLearnsFromTheSampledStep repeats the
// genotype/phenotype guard for the separable update, which maintains the
// covariance diagonal through its own code path.
func TestSeparableCovarianceUpdateLearnsFromTheSampledStep(t *testing.T) {
	config := NewSeparableConfig(5)
	parameters := deriveStrategyParameters(config)

	if len(parameters.negativeWeights) == 0 {
		t.Fatal("default separable configuration has no negative weights to guard")
	}

	steps := sampledSteps(config.Lambda, config.ProblemSize)

	unrepaired := make([]candidate, config.Lambda)
	for index, step := range steps {
		unrepaired[index] = candidate{y: append([]float64(nil), step...), sampledY: step}
	}

	want := newStrategyState(config)
	updateStrategyCovariance(want, unrepaired, true, parameters)

	got := newStrategyState(config)
	updateStrategyCovariance(got, adaptationStepFixture(steps), true, parameters)

	assertVectorClose(t, got.diagonal, want.diagonal, 0)
	assertVectorClose(t, got.d, want.d, 0)

	folded := newStrategyState(config)

	for index := range unrepaired {
		for coordinate := range unrepaired[index].y {
			unrepaired[index].y[coordinate] *= -0.1
			unrepaired[index].sampledY = unrepaired[index].y
		}
	}

	updateStrategyCovariance(folded, unrepaired, true, parameters)

	if reflect.DeepEqual(folded.diagonal, want.diagonal) {
		t.Fatal("folded steps produce the same diagonal; the test cannot fail")
	}
}
