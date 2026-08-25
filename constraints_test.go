package cmaes

import (
	"math"
	"reflect"
	"testing"
)

func TestEvaluateConstraints(t *testing.T) {
	config := &ConstraintConfig{
		Inequalities: []ConstraintFunction{
			func(position []float64) float64 { return position[0] - 1 },
			func(position []float64) float64 { return -position[1] },
		},
		Equalities: []ConstraintFunction{
			func(position []float64) float64 { return position[0] + position[1] },
		},
		EqualityTolerance: 0.25,
	}

	tests := []struct {
		name     string
		position []float64
		want     ConstraintEvaluation
	}{
		{
			name:     "feasible",
			position: []float64{0.1, 0},
			want:     ConstraintEvaluation{Feasible: true},
		},
		{
			name:     "violations are aggregated",
			position: []float64{2, -1},
			want:     ConstraintEvaluation{Violation: 2.75},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := EvaluateConstraints(test.position, config)
			if !reflect.DeepEqual(got, test.want) {
				t.Errorf("EvaluateConstraints() = %#v, want %#v", got, test.want)
			}
		})
	}

	if got := EvaluateConstraints([]float64{0}, nil); !got.Feasible || got.Violation != 0 {
		t.Errorf("nil config evaluation = %#v, want feasible", got)
	}

	nonFinite := EvaluateConstraints([]float64{0}, &ConstraintConfig{
		Inequalities: []ConstraintFunction{func([]float64) float64 { return math.NaN() }},
	})
	if nonFinite.Feasible || !math.IsInf(nonFinite.Violation, 1) {
		t.Errorf("non-finite evaluation = %#v, want infinite violation", nonFinite)
	}
}

func TestConstraintRankingMethods(t *testing.T) {
	feasibleHighCost := CandidateEvaluation{Cost: 10}
	infeasibleLowCost := CandidateEvaluation{Cost: 0, ConstraintViolation: 1}
	infeasibleSmallerViolation := CandidateEvaluation{Cost: 100, ConstraintViolation: 0.5}

	if !BetterConstrainedCandidate(feasibleHighCost, infeasibleLowCost, &ConstraintConfig{}) {
		t.Fatal("Deb rules did not prefer a feasible candidate")
	}

	if !BetterConstrainedCandidate(
		infeasibleSmallerViolation,
		infeasibleLowCost,
		&ConstraintConfig{},
	) {
		t.Fatal("Deb rules did not prefer the smaller violation")
	}

	penaltyConfig := &ConstraintConfig{
		Handling:      ConstraintHandlingPenalty,
		PenaltyMethod: PenaltyLinear,
		PenaltyFactor: 2,
	}
	if !BetterConstrainedCandidate(infeasibleLowCost, feasibleHighCost, penaltyConfig) {
		t.Fatal("penalty ranking did not prefer the smaller penalized score")
	}

	if got := PenalizedCost(3, 2, 4, PenaltyLinear); got != 11 {
		t.Errorf("linear PenalizedCost = %v, want 11", got)
	}

	if got := PenalizedCost(3, 2, 4, PenaltyQuadratic); got != 19 {
		t.Errorf("quadratic PenalizedCost = %v, want 19", got)
	}
}

func TestOptimizeWithDebConstraintsFindsFeasibleBoundary(t *testing.T) {
	config := optimizationConfig(2, 501, func(position []float64) float64 {
		dx := position[0] - 1
		dy := position[1] - 1

		return dx*dx + dy*dy
	})
	config.LowerBound = -2
	config.UpperBound = 2
	config.InitialSigma = 0.8
	config.MaxIterations = 500
	config.Constraints = &ConstraintConfig{
		Inequalities: []ConstraintFunction{
			func(position []float64) float64 { return position[0] + position[1] - 0.5 },
		},
	}

	result, err := Optimize(config)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	if result.GlobalBest.ConstraintViolation != 0 {
		t.Fatalf("best violation = %g, want zero", result.GlobalBest.ConstraintViolation)
	}

	if result.GlobalBest.Cost > 1.126 {
		t.Errorf("constrained cost = %g, want <= 1.126", result.GlobalBest.Cost)
	}
}

func TestConstrainedParallelEvaluationMatchesSerial(t *testing.T) {
	newConfig := func(parallel bool) *Config {
		config := optimizationConfig(4, 502, sphere)
		config.MaxIterations = 100
		config.EnableParallel = parallel
		config.MaxWorkers = 3
		config.Constraints = &ConstraintConfig{
			Inequalities: []ConstraintFunction{
				func(position []float64) float64 { return position[0] - 0.25 },
			},
		}

		return config
	}

	serial, err := Optimize(newConfig(false))
	if err != nil {
		t.Fatalf("serial Optimize: %v", err)
	}

	parallel, err := Optimize(newConfig(true))
	if err != nil {
		t.Fatalf("parallel Optimize: %v", err)
	}

	if !reflect.DeepEqual(serial, parallel) {
		t.Fatalf("serial and parallel constrained results differ:\nserial   = %#v\nparallel = %#v",
			serial, parallel)
	}
}
