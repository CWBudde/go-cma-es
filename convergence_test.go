package cmaes

import (
	"math"
	"testing"
)

func convergenceTestState() *strategyState {
	return &strategyState{
		m:      []float64{0, 0},
		psigma: []float64{0, 0},
		pc:     []float64{0, 0},
		c:      identityMatrix(2),
		b:      identityMatrix(2),
		d:      []float64{1, 1},
		sigma:  1,
	}
}

func TestConvergenceCriteriaReportDistinctReasons(t *testing.T) {
	target := 1.0
	population := []candidate{{cost: 2, constraintViolation: 0}}

	tests := []struct {
		name      string
		config    ConvergenceConfig
		want      TerminationReason
		prepare   func(*convergenceTracker, *strategyState)
		best      Best
		iteration int
	}{
		{
			name: "target cost", config: ConvergenceConfig{TargetCost: &target},
			want: TerminationTargetCost, best: Best{Cost: 1}, iteration: 1,
		},
		{
			name: "TolX", config: ConvergenceConfig{TolX: 1e-6},
			want: TerminationTolX, best: Best{Cost: 2}, iteration: 1,
			prepare: func(_ *convergenceTracker, state *strategyState) {
				state.sigma = 1e-7
			},
		},
		{
			name: "TolFun", config: ConvergenceConfig{TolFun: 1e-6},
			want: TerminationTolFun, best: Best{Cost: 2}, iteration: 1,
			prepare: func(tracker *convergenceTracker, _ *strategyState) {
				tracker.historyLimit = 1
			},
		},
		{
			name: "TolXUp", config: ConvergenceConfig{TolXUp: 10},
			want: TerminationTolXUp, best: Best{Cost: 2}, iteration: 1,
			prepare: func(_ *convergenceTracker, state *strategyState) {
				state.sigma = 11
			},
		},
		{
			name: "condition", config: ConvergenceConfig{ConditionCov: 100},
			want: TerminationConditionNumber, best: Best{Cost: 2}, iteration: 1,
			prepare: func(_ *convergenceTracker, state *strategyState) {
				state.d = []float64{1, 11}
			},
		},
		{
			name: "no effect axis", config: ConvergenceConfig{NoEffectAxis: true},
			want: TerminationNoEffectAxis, best: Best{Cost: 2}, iteration: 1,
			prepare: func(_ *convergenceTracker, state *strategyState) {
				state.m = []float64{math.MaxFloat64 / 2, math.MaxFloat64 / 2}
				state.sigma = math.SmallestNonzeroFloat64
			},
		},
		{
			name: "no effect coord", config: ConvergenceConfig{NoEffectCoord: true},
			want: TerminationNoEffectCoord, best: Best{Cost: 2}, iteration: 1,
			prepare: func(_ *convergenceTracker, state *strategyState) {
				state.m = []float64{math.MaxFloat64 / 2, 0}
				state.sigma = math.SmallestNonzeroFloat64
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := convergenceTestState()

			tracker := newConvergenceTracker(&test.config, nil, 1, 2, 6)
			if test.prepare != nil {
				test.prepare(tracker, state)
			}

			reason, stop := tracker.observe(test.iteration, test.best, state, population)
			if !stop || reason != test.want {
				t.Errorf("observe() = (%q, %v), want (%q, true)", reason, stop, test.want)
			}
		})
	}
}

func TestConvergenceStagnationAndMinimumIteration(t *testing.T) {
	config := &ConvergenceConfig{
		MinImprovement:       0.5,
		StagnationIterations: 2,
		MinIterations:        4,
	}
	tracker := newConvergenceTracker(config, nil, 1, 2, 6)
	state := convergenceTestState()
	population := []candidate{{cost: 10}}

	for iteration := 1; iteration < 4; iteration++ {
		reason, stop := tracker.observe(iteration, Best{Cost: 10}, state, population)
		if stop {
			t.Fatalf("iteration %d stopped early with %q", iteration, reason)
		}
	}

	reason, stop := tracker.observe(4, Best{Cost: 10}, state, population)
	if !stop || reason != TerminationStagnation {
		t.Errorf("iteration 4 = (%q, %v), want (%q, true)",
			reason, stop, TerminationStagnation)
	}
}

func TestTargetCostRequiresFeasibleBest(t *testing.T) {
	target := 0.0
	tracker := newConvergenceTracker(
		&ConvergenceConfig{TargetCost: &target},
		&ConstraintConfig{},
		1,
		2,
		6,
	)

	reason, stop := tracker.observe(
		1,
		Best{Cost: -1, ConstraintViolation: 1},
		convergenceTestState(),
		[]candidate{{cost: -1, constraintViolation: 1}},
	)
	if stop || reason != "" {
		t.Errorf("infeasible target = (%q, %v), want no stop", reason, stop)
	}
}

func TestCovarianceConditionNumberUsesSquaredAxisScales(t *testing.T) {
	if got := covarianceConditionNumber([]float64{2, 10}); got != 25 {
		t.Errorf("condition = %v, want 25", got)
	}
}
