package cmaes

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func TestBoundaryRepairs(t *testing.T) {
	tests := []struct {
		name     string
		method   BoundaryMethod
		position []float64
		want     []float64
	}{
		{
			name:     "clamp",
			method:   BoundaryClamp,
			position: []float64{-3, -1, 0.5, 2, 4},
			want:     []float64{-1, -1, 0.5, 2, 2},
		},
		{
			name:     "reflect repeatedly",
			method:   BoundaryReflect,
			position: []float64{-8, -2, 0.5, 3, 9},
			want:     []float64{0, 0, 0.5, 1, 1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := &strategyState{
				m:     []float64{0, 0, 0, 0, 0},
				sigma: 2,
			}
			population := []candidate{{
				x: append([]float64(nil), test.position...),
				y: make([]float64, len(test.position)),
			}}
			config := &Config{BoundaryMethod: test.method, LowerBound: -1, UpperBound: 2}

			applyBoundaryHandling(population, state, config)
			assertVectorClose(t, population[0].x, test.want, 0)

			for coordinate, value := range test.want {
				if population[0].y[coordinate] != value/2 {
					t.Errorf("y[%d] = %v, want %v", coordinate, population[0].y[coordinate], value/2)
				}
			}
		})
	}
}

func TestHansenLinearQuadraticTransformation(t *testing.T) {
	const margin = 0.05

	tests := []struct {
		name      string
		value     float64
		wantValue float64
		wantRef   float64
	}{
		{name: "identity", value: 0, wantValue: 0, wantRef: 0},
		{name: "lower shoulder", value: -1, wantValue: -1 + margin/4, wantRef: -1},
		{name: "lower bound", value: -1 - margin, wantValue: -1, wantRef: -1 - margin},
		{name: "upper shoulder", value: 1, wantValue: 1 - margin/4, wantRef: 1},
		{name: "upper bound", value: 1 + margin, wantValue: 1, wantRef: 1 + margin},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			value, reference := transformBounded(test.value, -1, 1)
			if math.Abs(value-test.wantValue) > 1e-15 || math.Abs(reference-test.wantRef) > 1e-15 {
				t.Errorf("transformBounded(%v) = (%v, %v), want (%v, %v)",
					test.value, value, reference, test.wantValue, test.wantRef)
			}
		})
	}

	for _, value := range []float64{-1e6, -17, -1.2, 1.2, 19, 1e6} {
		transformed, _ := transformBounded(value, -1, 1)
		if transformed < -1 || transformed > 1 {
			t.Errorf("transformBounded(%v) = %v, outside [-1, 1]", value, transformed)
		}
	}
}

func TestPenaltyBoundaryLeavesInteriorStepsUntouched(t *testing.T) {
	state := &strategyState{m: []float64{0.25, -0.5}, sigma: 0.75}
	population := []candidate{{
		x: []float64{1.5, -2},
		y: []float64{5.0 / 3, -2},
	}}
	wantX := append([]float64(nil), population[0].x...)
	wantY := append([]float64(nil), population[0].y...)
	config := &Config{BoundaryMethod: BoundaryPenalty, LowerBound: -10, UpperBound: 10}

	applyBoundaryHandling(population, state, config)
	population[0].cost = 7
	assignBoundaryPenalties(population, config.BoundaryMethod)

	assertVectorClose(t, population[0].x, wantX, 0)
	assertVectorClose(t, population[0].y, wantY, 0)
	assertVectorClose(t, population[0].evaluatedPosition(), wantX, 0)

	if population[0].boundaryDistance != 0 || population[0].boundaryPenalty != 0 {
		t.Errorf("interior penalty = (%v, %v), want zero",
			population[0].boundaryDistance, population[0].boundaryPenalty)
	}
}

func TestPenaltyBoundaryDoesNotDistortSigmaOnInteriorProblem(t *testing.T) {
	config := optimizationConfig(4, 402, sphere)
	config.LowerBound = -100
	config.UpperBound = 100
	config.InitialMean = []float64{1, -2, 3, -4}
	parameters := deriveStrategyParameters(config)
	controlState := newStrategyState(config)
	penaltyState := newStrategyState(config)
	controlPopulation := samplePopulation(
		controlState,
		config.Lambda,
		rand.New(rand.NewSource(402)),
	)
	penaltyPopulation := samplePopulation(
		penaltyState,
		config.Lambda,
		rand.New(rand.NewSource(402)),
	)

	applyBoundaryHandling(penaltyPopulation, penaltyState, config)

	for index := range controlPopulation {
		controlPopulation[index].cost = sphere(controlPopulation[index].x)
		penaltyPopulation[index].cost = sphere(penaltyPopulation[index].evaluatedPosition())
	}

	assignBoundaryPenalties(penaltyPopulation, BoundaryPenalty)
	sortPopulation(controlPopulation, nil)
	sortPopulation(penaltyPopulation, nil)
	updateDistribution(controlState, controlPopulation, parameters, 0)
	updateDistribution(penaltyState, penaltyPopulation, parameters, 0)

	if !reflect.DeepEqual(controlState, penaltyState) {
		t.Fatalf("interior penalty update changed the distribution:\ncontrol = %#v\npenalty = %#v",
			controlState, penaltyState)
	}
}

func TestBoundaryPenaltyScalesWithObjectiveSpread(t *testing.T) {
	population := []candidate{
		{cost: 1, boundaryDistance: 0.5},
		{cost: 2, boundaryDistance: 0.5},
		{cost: 5, boundaryDistance: 0.5},
		{cost: 9, boundaryDistance: 0.5},
	}

	assignBoundaryPenalties(population, BoundaryPenalty)

	// With four values, the reference quartile indices are one and three,
	// giving an objective spread of 9-2=7.
	if population[0].boundaryPenalty != 3.5 {
		t.Errorf("boundary penalty = %v, want 3.5", population[0].boundaryPenalty)
	}
}

func TestOptimizeFindsBoundActiveOptimumForEveryBoundaryMethod(t *testing.T) {
	methods := []BoundaryMethod{BoundaryClamp, BoundaryReflect, BoundaryPenalty}

	for _, method := range methods {
		t.Run(string(method), func(t *testing.T) {
			objective := func(position []float64) float64 {
				var cost float64

				for _, value := range position {
					delta := value - 2
					cost += delta * delta
				}

				return cost
			}

			config := optimizationConfig(3, 401, objective)
			config.LowerBound = -1
			config.UpperBound = 1
			config.InitialSigma = 0.8
			config.MaxIterations = 500
			config.BoundaryMethod = method
			// This test isolates boundary repair from Phase 6's active update.
			config.ActiveCMA = false

			result, err := Optimize(config)
			if err != nil {
				t.Fatalf("Optimize: %v", err)
			}

			if result.GlobalBest.Cost > 3.001 {
				t.Errorf("bound-active cost = %g, want <= 3.001", result.GlobalBest.Cost)
			}

			for coordinate, value := range result.GlobalBest.Position {
				if value < -1 || value > 1 {
					t.Errorf("position[%d] = %v, outside [-1, 1]", coordinate, value)
				}
			}
		})
	}
}
