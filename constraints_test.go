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

func TestEvaluateConstraintsEdgeCases(t *testing.T) {
	tests := []struct {
		name   string
		config *ConstraintConfig
		want   ConstraintEvaluation
	}{
		{
			name:   "nil inequality callback",
			config: &ConstraintConfig{Inequalities: []ConstraintFunction{nil}},
			want:   ConstraintEvaluation{Violation: math.Inf(1)},
		},
		{
			name: "nil equality callback",
			config: &ConstraintConfig{
				Inequalities: []ConstraintFunction{constant(-1)},
				Equalities:   []ConstraintFunction{nil},
			},
			want: ConstraintEvaluation{Violation: math.Inf(1)},
		},
		{
			name:   "positive infinite inequality",
			config: &ConstraintConfig{Inequalities: []ConstraintFunction{constant(math.Inf(1))}},
			want:   ConstraintEvaluation{Violation: math.Inf(1)},
		},
		{
			name:   "negative infinite inequality",
			config: &ConstraintConfig{Inequalities: []ConstraintFunction{constant(math.Inf(-1))}},
			want:   ConstraintEvaluation{Violation: math.Inf(1)},
		},
		{
			name: "NaN equality",
			config: &ConstraintConfig{
				Equalities: []ConstraintFunction{constant(math.NaN())},
			},
			want: ConstraintEvaluation{Violation: math.Inf(1)},
		},
		{
			name: "equality exactly at the tolerance is feasible",
			config: &ConstraintConfig{
				Equalities:        []ConstraintFunction{constant(0.25)},
				EqualityTolerance: 0.25,
			},
			want: ConstraintEvaluation{Feasible: true},
		},
		{
			name: "negative equality at the tolerance is feasible",
			config: &ConstraintConfig{
				Equalities:        []ConstraintFunction{constant(-0.25)},
				EqualityTolerance: 0.25,
			},
			want: ConstraintEvaluation{Feasible: true},
		},
		{
			name: "equality just past the tolerance violates",
			config: &ConstraintConfig{
				Equalities:        []ConstraintFunction{constant(-0.75)},
				EqualityTolerance: 0.25,
			},
			want: ConstraintEvaluation{Violation: 0.5},
		},
		{
			name: "an aggregate that overflows to infinity is caught",
			config: &ConstraintConfig{
				Inequalities: []ConstraintFunction{
					constant(math.MaxFloat64),
					constant(math.MaxFloat64),
				},
			},
			want: ConstraintEvaluation{Violation: math.Inf(1)},
		},
		{
			name: "satisfied inequality contributes nothing",
			config: &ConstraintConfig{
				Inequalities: []ConstraintFunction{constant(-3)},
			},
			want: ConstraintEvaluation{Feasible: true},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := EvaluateConstraints([]float64{0, 0}, test.config)
			if !reflect.DeepEqual(got, test.want) {
				t.Errorf("EvaluateConstraints() = %#v, want %#v", got, test.want)
			}
		})
	}
}

func TestIsFeasible(t *testing.T) {
	tests := []struct {
		name      string
		violation float64
		want      bool
	}{
		{name: "zero", violation: 0, want: true},
		{name: "negative aggregates are clamped to feasible", violation: -1, want: true},
		{name: "positive", violation: 1e-12, want: false},
		{name: "infinite", violation: math.Inf(1), want: false},
		{name: "NaN is never feasible", violation: math.NaN(), want: false},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := IsFeasible(test.violation); got != test.want {
				t.Errorf("IsFeasible(%v) = %v, want %v", test.violation, got, test.want)
			}
		})
	}
}

func TestDebFeasibilityRules(t *testing.T) {
	config := &ConstraintConfig{}

	tests := []struct {
		name      string
		candidate CandidateEvaluation
		incumbent CandidateEvaluation
		want      bool
	}{
		{
			name:      "feasible beats infeasible regardless of cost",
			candidate: CandidateEvaluation{Cost: 1e6},
			incumbent: CandidateEvaluation{Cost: -1e6, ConstraintViolation: 1e-9},
			want:      true,
		},
		{
			name:      "infeasible loses to feasible regardless of cost",
			candidate: CandidateEvaluation{Cost: -1e6, ConstraintViolation: 1e-9},
			incumbent: CandidateEvaluation{Cost: 1e6},
			want:      false,
		},
		{
			name:      "two feasible compare by cost",
			candidate: CandidateEvaluation{Cost: 1},
			incumbent: CandidateEvaluation{Cost: 2},
			want:      true,
		},
		{
			name:      "two feasible with the worse cost lose",
			candidate: CandidateEvaluation{Cost: 2},
			incumbent: CandidateEvaluation{Cost: 1},
			want:      false,
		},
		{
			name:      "two infeasible compare by violation",
			candidate: CandidateEvaluation{Cost: 100, ConstraintViolation: 0.5},
			incumbent: CandidateEvaluation{Cost: 0, ConstraintViolation: 1},
			want:      true,
		},
		{
			name:      "equal violation falls through to cost",
			candidate: CandidateEvaluation{Cost: 3, ConstraintViolation: 1},
			incumbent: CandidateEvaluation{Cost: 4, ConstraintViolation: 1},
			want:      true,
		},
		{
			name:      "equal violation and worse cost loses",
			candidate: CandidateEvaluation{Cost: 4, ConstraintViolation: 1},
			incumbent: CandidateEvaluation{Cost: 3, ConstraintViolation: 1},
			want:      false,
		},
		{
			name:      "equal violation and equal cost is not better",
			candidate: CandidateEvaluation{Cost: 3, ConstraintViolation: 1},
			incumbent: CandidateEvaluation{Cost: 3, ConstraintViolation: 1},
			want:      false,
		},
		{
			name:      "a negative violation ranks as feasible",
			candidate: CandidateEvaluation{Cost: 5, ConstraintViolation: -1},
			incumbent: CandidateEvaluation{Cost: 0, ConstraintViolation: 1},
			want:      true,
		},
		{
			name:      "a NaN violation loses to a finite one",
			candidate: CandidateEvaluation{Cost: 5, ConstraintViolation: math.NaN()},
			incumbent: CandidateEvaluation{Cost: 5, ConstraintViolation: 1},
			want:      false,
		},
		{
			name:      "a finite violation beats NaN",
			candidate: CandidateEvaluation{Cost: 5, ConstraintViolation: 1},
			incumbent: CandidateEvaluation{Cost: 5, ConstraintViolation: math.NaN()},
			want:      true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := BetterConstrainedCandidate(
				test.candidate, test.incumbent, config,
			); got != test.want {
				t.Errorf("BetterConstrainedCandidate() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestBetterConstrainedCandidateNilConfig(t *testing.T) {
	tests := []struct {
		name      string
		candidate CandidateEvaluation
		incumbent CandidateEvaluation
		want      bool
	}{
		{
			name:      "lower cost wins",
			candidate: CandidateEvaluation{Cost: 1},
			incumbent: CandidateEvaluation{Cost: 2},
			want:      true,
		},
		{
			name:      "violations are ignored without a config",
			candidate: CandidateEvaluation{Cost: 1, ConstraintViolation: 100},
			incumbent: CandidateEvaluation{Cost: 2},
			want:      true,
		},
		{
			name:      "equal costs are not better",
			candidate: CandidateEvaluation{Cost: 1},
			incumbent: CandidateEvaluation{Cost: 1},
			want:      false,
		},
		{
			name:      "a NaN cost loses to a real cost",
			candidate: CandidateEvaluation{Cost: math.NaN()},
			incumbent: CandidateEvaluation{Cost: 1},
			want:      false,
		},
		{
			name:      "a real cost beats a NaN cost",
			candidate: CandidateEvaluation{Cost: 1},
			incumbent: CandidateEvaluation{Cost: math.NaN()},
			want:      true,
		},
		{
			name:      "two NaN costs compare as equivalent",
			candidate: CandidateEvaluation{Cost: math.NaN()},
			incumbent: CandidateEvaluation{Cost: math.NaN()},
			want:      false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := BetterConstrainedCandidate(
				test.candidate, test.incumbent, nil,
			); got != test.want {
				t.Errorf("BetterConstrainedCandidate() = %v, want %v", got, test.want)
			}
		})
	}
}

// TestBetterConstrainedCandidateIsStrictWeakOrdering pins the property
// sort.SliceStable relies on: irreflexivity, asymmetry, and transitivity of
// both the ordering and the equivalence it induces.
func TestBetterConstrainedCandidateIsStrictWeakOrdering(t *testing.T) {
	evaluations := []CandidateEvaluation{
		{Cost: 5, ConstraintViolation: 1},
		{Cost: 5, ConstraintViolation: math.NaN()},
		{Cost: 5, ConstraintViolation: math.Inf(1)},
		{Cost: 5, ConstraintViolation: -1},
		{Cost: 5, ConstraintViolation: 0},
		{Cost: math.NaN(), ConstraintViolation: 0},
		{Cost: math.NaN(), ConstraintViolation: 1},
		{Cost: -1e9, ConstraintViolation: 1},
		{Cost: 1e9, ConstraintViolation: 0},
		{Cost: math.Inf(-1), ConstraintViolation: 2},
	}

	configs := map[string]*ConstraintConfig{
		"nil":         nil,
		"feasibility": {},
		"linear penalty": {
			Handling:      ConstraintHandlingPenalty,
			PenaltyMethod: PenaltyLinear,
			PenaltyFactor: 2,
		},
		"quadratic penalty": {
			Handling:      ConstraintHandlingPenalty,
			PenaltyMethod: PenaltyQuadratic,
			PenaltyFactor: 1,
		},
		"zero penalty factor": {
			Handling: ConstraintHandlingPenalty,
		},
	}

	for name, config := range configs {
		t.Run(name, func(t *testing.T) {
			less := func(left, right CandidateEvaluation) bool {
				return BetterConstrainedCandidate(left, right, config)
			}
			// Equivalence under a strict weak ordering: neither is less.
			equivalent := func(left, right CandidateEvaluation) bool {
				return !less(left, right) && !less(right, left)
			}

			for i, a := range evaluations {
				if less(a, a) {
					t.Errorf("less(%v, %v) is true, want irreflexive", a, a)
				}

				for j, b := range evaluations {
					if less(a, b) && less(b, a) {
						t.Errorf("less is not asymmetric for %v (%d) and %v (%d)", a, i, b, j)
					}

					for _, c := range evaluations {
						if less(a, b) && less(b, c) && !less(a, c) {
							t.Errorf("less is not transitive for %v, %v, %v", a, b, c)
						}

						if equivalent(a, b) && equivalent(b, c) && !equivalent(a, c) {
							t.Errorf("equivalence is not transitive for %v, %v, %v", a, b, c)
						}
					}
				}
			}
		})
	}
}

func TestPenalizedCost(t *testing.T) {
	tests := []struct {
		name      string
		method    PenaltyMethod
		cost      float64
		violation float64
		factor    float64
		want      float64
	}{
		{name: "linear", method: PenaltyLinear, cost: 3, violation: 2, factor: 4, want: 11},
		{name: "quadratic", method: PenaltyQuadratic, cost: 3, violation: 2, factor: 4, want: 19},
		{name: "empty defaults to quadratic", method: "", cost: 1, violation: 2, factor: 3, want: 13},
		{
			name: "unrecognized defaults to quadratic", method: "bogus",
			cost: 1, violation: 2, factor: 3, want: 13,
		},
		{name: "zero violation leaves the cost alone", method: PenaltyLinear, cost: 7, want: 7},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := PenalizedCost(test.cost, test.violation, test.factor, test.method)
			if got != test.want {
				t.Errorf("PenalizedCost() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestPenaltyRankingFallsBackToDebRulesOnExactTies(t *testing.T) {
	config := &ConstraintConfig{
		Handling:      ConstraintHandlingPenalty,
		PenaltyMethod: PenaltyLinear,
		PenaltyFactor: 1,
	}

	// Both score 10 under a linear penalty with factor one, so ranking falls
	// through to Deb's rules, which prefer the feasible candidate.
	feasible := CandidateEvaluation{Cost: 10}
	infeasible := CandidateEvaluation{Cost: 9, ConstraintViolation: 1}

	if !BetterConstrainedCandidate(feasible, infeasible, config) {
		t.Error("exact score tie did not fall back to Deb's feasibility rule")
	}

	if BetterConstrainedCandidate(infeasible, feasible, config) {
		t.Error("exact score tie preferred the infeasible candidate")
	}

	// Same scores, both infeasible: the smaller violation wins.
	small := CandidateEvaluation{Cost: 9.5, ConstraintViolation: 0.5}
	large := CandidateEvaluation{Cost: 9, ConstraintViolation: 1}

	if !BetterConstrainedCandidate(small, large, config) {
		t.Error("exact score tie did not prefer the smaller violation")
	}
}

// TestPenaltyRankingLetsCostOutweighFeasibility documents the trade-off named in
// BetterConstrainedCandidate's doc comment: under a penalty policy a large
// enough cost advantage beats feasibility, unlike Deb's rules.
func TestPenaltyRankingLetsCostOutweighFeasibility(t *testing.T) {
	infeasible := CandidateEvaluation{Cost: -1e9, ConstraintViolation: 1}
	feasible := CandidateEvaluation{Cost: 1e9}

	penalty := &ConstraintConfig{
		Handling:      ConstraintHandlingPenalty,
		PenaltyMethod: PenaltyLinear,
		PenaltyFactor: 1,
	}
	if !BetterConstrainedCandidate(infeasible, feasible, penalty) {
		t.Error("penalty ranking did not prefer the far cheaper infeasible candidate")
	}

	if BetterConstrainedCandidate(infeasible, feasible, &ConstraintConfig{}) {
		t.Error("Deb's rules preferred an infeasible candidate")
	}
}

func constant(value float64) ConstraintFunction {
	return func([]float64) float64 { return value }
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
