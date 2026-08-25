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

// hansenConvergence returns the published default criteria set (Hansen,
// arXiv:1604.00772, App. B): TolX = 1e-12 relative to sigma^(0),
// TolFun = 1e-12, TolXUp = 1e4, ConditionCov = 1e14, both no-effect criteria
// active. The values are written out here rather than read from
// NewDefaultConfig so these tests pin behavior instead of tracking whatever
// the default constants happen to be. Stagnation is this library's own
// criterion and is left off; the cases that need it switch it on.
func hansenConvergence() ConvergenceConfig {
	return ConvergenceConfig{
		TolX:          1e-12,
		TolFun:        1e-12,
		TolXUp:        1e4,
		ConditionCov:  1e14,
		NoEffectAxis:  true,
		NoEffectCoord: true,
	}
}

// flatPopulation builds a population whose candidates carry the given costs
// and no constraint violation, which is what the TolFun window consumes.
func flatPopulation(costs ...float64) []candidate {
	population := make([]candidate, 0, len(costs))
	for _, cost := range costs {
		population = append(population, candidate{cost: cost})
	}

	return population
}

func TestConvergenceCriteriaReportDistinctReasons(t *testing.T) {
	target := 1.0
	population := flatPopulation(2)

	tests := []struct {
		name      string
		config    ConvergenceConfig
		want      TerminationReason
		prepare   func(*strategyState)
		best      Best
		iteration int
		wantStop  bool
	}{
		{
			// The published criterion is f(x_best) <= target; equality counts.
			name: "target cost", config: ConvergenceConfig{TargetCost: &target},
			want: TerminationTargetCost, wantStop: true, best: Best{Cost: 1}, iteration: 1,
		},
		{
			// TolX at its published default: stop once the distribution has
			// shrunk below 1e-12 * sigma^(0). With sigma^(0) = 1, D = I and
			// p_c = 0 the measured extent is sigma itself, so sigma = 1e-13
			// is one decade below the threshold.
			name: "TolX", config: ConvergenceConfig{TolX: 1e-12},
			want: TerminationTolX, wantStop: true, best: Best{Cost: 2}, iteration: 1,
			prepare: func(state *strategyState) {
				state.sigma = 1e-13
			},
		},
		{
			// One decade of extent above the threshold: no stop.
			name: "TolX not reached", config: ConvergenceConfig{TolX: 1e-12},
			want: "", wantStop: false, best: Best{Cost: 2}, iteration: 1,
			prepare: func(state *strategyState) {
				state.sigma = 1e-11
			},
		},
		{
			// D holds standard deviations, so cond(C) = (1e8/1)^2 = 1e16,
			// two decades past the published ConditionCov = 1e14.
			name: "condition", config: ConvergenceConfig{ConditionCov: 1e14},
			want: TerminationConditionNumber, wantStop: true, best: Best{Cost: 2}, iteration: 1,
			prepare: func(state *strategyState) {
				state.d = []float64{1, 1e8}
			},
		},
		{
			// cond(C) = (1e6)^2 = 1e12, two decades below the threshold.
			name: "condition not reached", config: ConvergenceConfig{ConditionCov: 1e14},
			want: "", wantStop: false, best: Best{Cost: 2}, iteration: 1,
			prepare: func(state *strategyState) {
				state.d = []float64{1, 1e6}
			},
		},
		{
			// Axis (iteration-1) mod n = 0. With B = I the tested step is
			// 0.1*sigma*D_0 in coordinate 0 and zero elsewhere; against a mean
			// of ~9e307 (ulp ~2e292) a step of 0.1 is lost in the rounding.
			name: "no effect axis", config: ConvergenceConfig{NoEffectAxis: true},
			want: TerminationNoEffectAxis, wantStop: true, best: Best{Cost: 2}, iteration: 1,
			prepare: func(state *strategyState) {
				state.m = []float64{math.MaxFloat64 / 2, math.MaxFloat64 / 2}
			},
		},
		{
			// A mean of 1 absorbs no 0.1-sized step, so the axis still moves.
			name: "no effect axis not reached", config: ConvergenceConfig{NoEffectAxis: true},
			want: "", wantStop: false, best: Best{Cost: 2}, iteration: 1,
			prepare: func(state *strategyState) {
				state.m = []float64{1, 1}
			},
		},
		{
			// The coordinate criterion is an "any coordinate" quantifier, so a
			// single unmovable coordinate is enough; the step is
			// 0.2*sigma*sqrt(C_00) = 0.2 against a mean of ~9e307.
			name: "no effect coord", config: ConvergenceConfig{NoEffectCoord: true},
			want: TerminationNoEffectCoord, wantStop: true, best: Best{Cost: 2}, iteration: 1,
			prepare: func(state *strategyState) {
				state.m = []float64{math.MaxFloat64 / 2, 0}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := convergenceTestState()

			tracker := newConvergenceTracker(&test.config, nil, 1, 2, 6)
			if test.prepare != nil {
				test.prepare(state)
			}

			reason, stop := tracker.observe(test.iteration, test.best, state, population)
			if stop != test.wantStop || reason != test.want {
				t.Errorf("observe() = (%q, %v), want (%q, %v)",
					reason, stop, test.want, test.wantStop)
			}
		})
	}
}

// TestConvergencePrecedenceOrder pins the order in which observe() reports
// criteria. Every case drives a state in which several criteria hold at once
// under the full published criteria set; `order` lists the reasons expected as
// the winner is successively switched off, so a reordering of the blocks in
// observe() fails here. The cases chain across every adjacent pair of blocks.
func TestConvergencePrecedenceOrder(t *testing.T) {
	target := 0.0
	population := flatPopulation(5)
	best := Best{Cost: 5}

	// fillWindow runs historyLimit-1 flat generations so that one further
	// observation completes the TolFun window.
	fillWindow := func(t *testing.T, tracker *convergenceTracker, state *strategyState) {
		t.Helper()

		for iteration := 1; iteration < tracker.historyLimit; iteration++ {
			if reason, stop := tracker.observe(iteration, best, state, population); stop {
				t.Fatalf("iteration %d stopped early with %q", iteration, reason)
			}
		}
	}

	tests := []struct {
		name  string
		tweak func(*ConvergenceConfig)
		run   func(*testing.T, *convergenceTracker) (TerminationReason, bool)
		order []TerminationReason
	}{
		{
			// Target reached while the distribution has already collapsed
			// below TolX.
			name:  "target cost outranks TolX",
			tweak: func(config *ConvergenceConfig) { config.TargetCost = &target },
			order: []TerminationReason{TerminationTargetCost, TerminationTolX},
			run: func(_ *testing.T, tracker *convergenceTracker) (TerminationReason, bool) {
				state := convergenceTestState()
				state.sigma = 1e-13

				return tracker.observe(1, Best{Cost: -1}, state, population)
			},
		},
		{
			// A full, flat TolFun window reached at the same generation in
			// which sigma collapses below TolX.
			name:  "TolX outranks TolFun",
			order: []TerminationReason{TerminationTolX, TerminationTolFun},
			run: func(t *testing.T, tracker *convergenceTracker) (TerminationReason, bool) {
				t.Helper()

				state := convergenceTestState()
				fillWindow(t, tracker, state)
				state.sigma = 1e-13

				return tracker.observe(tracker.historyLimit, best, state, population)
			},
		},
		{
			// The same full, flat window, but with sigma blown up past TolXUp.
			name:  "TolFun outranks TolXUp",
			order: []TerminationReason{TerminationTolFun, TerminationTolXUp},
			run: func(t *testing.T, tracker *convergenceTracker) (TerminationReason, bool) {
				t.Helper()

				state := convergenceTestState()
				fillWindow(t, tracker, state)
				state.sigma = 1e5

				return tracker.observe(tracker.historyLimit, best, state, population)
			},
		},
		{
			// max(D) = 1e9 both blows the axis extent past 1e4*sigma^(0) and
			// drives cond(C) = 1e18 past ConditionCov.
			name:  "TolXUp outranks the condition number",
			order: []TerminationReason{TerminationTolXUp, TerminationConditionNumber},
			run: func(t *testing.T, tracker *convergenceTracker) (TerminationReason, bool) {
				t.Helper()

				state := convergenceTestState()
				if reason, stop := tracker.observe(1, best, state, population); stop {
					t.Fatalf("iteration 1 stopped early with %q", reason)
				}

				state.d = []float64{1, 1e9}

				return tracker.observe(2, best, state, population)
			},
		},
		{
			// cond(C) = (1e-4/1e-12)^2 = 1e16 exceeds ConditionCov while both
			// no-effect criteria hold against a mean of ~9e307. The axis
			// extent stays at 1e-4, far from either TolX or TolXUp.
			name: "condition number outranks both no-effect criteria",
			order: []TerminationReason{
				TerminationConditionNumber, TerminationNoEffectAxis, TerminationNoEffectCoord,
			},
			run: func(t *testing.T, tracker *convergenceTracker) (TerminationReason, bool) {
				t.Helper()

				state := convergenceTestState()
				if reason, stop := tracker.observe(1, best, state, population); stop {
					t.Fatalf("iteration 1 stopped early with %q", reason)
				}

				state.d = []float64{1e-12, 1e-4}
				state.m = []float64{math.MaxFloat64 / 2, math.MaxFloat64 / 2}

				return tracker.observe(2, best, state, population)
			},
		},
		{
			// At iteration 4 the tested axis is (4-1) mod 2 = 1, whose step
			// still moves coordinate 1 (mean 0), so NoEffectAxis does not
			// hold; coordinate 0 is unmovable, so NoEffectCoord does. The
			// best cost never improves, so the stagnation counter has reached
			// its limit by then as well.
			name: "no effect coord outranks stagnation",
			tweak: func(config *ConvergenceConfig) {
				config.StagnationIterations = 2
				config.MinIterations = 4
			},
			order: []TerminationReason{TerminationNoEffectCoord, TerminationStagnation},
			run: func(t *testing.T, tracker *convergenceTracker) (TerminationReason, bool) {
				t.Helper()

				state := convergenceTestState()
				state.m = []float64{math.MaxFloat64 / 2, 0}

				for iteration := 1; iteration < 4; iteration++ {
					if reason, stop := tracker.observe(
						iteration, best, state, population,
					); stop {
						t.Fatalf("iteration %d stopped early with %q", iteration, reason)
					}
				}

				return tracker.observe(4, best, state, population)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := hansenConvergence()
			if test.tweak != nil {
				test.tweak(&config)
			}

			for position, want := range test.order {
				tracker := newConvergenceTracker(&config, nil, 1, 2, 6)

				reason, stop := test.run(t, tracker)
				if !stop || reason != want {
					t.Fatalf("with %d criteria disabled: observe() = (%q, %v), want (%q, true)",
						position, reason, stop, want)
				}

				disableCriterion(&config, want)
			}
		})
	}
}

// disableCriterion switches off the single criterion that reports the given
// reason, so the next-ranked criterion for the same state can be observed.
func disableCriterion(config *ConvergenceConfig, reason TerminationReason) {
	switch reason {
	case TerminationTargetCost:
		config.TargetCost = nil
	case TerminationTolX:
		config.TolX = 0
	case TerminationTolFun:
		config.TolFun = 0
	case TerminationTolXUp:
		config.TolXUp = 0
	case TerminationConditionNumber:
		config.ConditionCov = 0
	case TerminationNoEffectAxis:
		config.NoEffectAxis = false
	case TerminationNoEffectCoord:
		config.NoEffectCoord = false
	case TerminationStagnation:
		config.StagnationIterations = 0
	default:
		config.MinIterations = 0
	}
}

// TestTolFunHistoryLimitMatchesHansen pins the published window length,
// 10 + ceil(30n/lambda) generations.
func TestTolFunHistoryLimitMatchesHansen(t *testing.T) {
	tests := []struct {
		dimension int
		lambda    int
		want      int
	}{
		{dimension: 2, lambda: 6, want: 20},   // 10 + ceil(60/6)  = 10 + 10
		{dimension: 10, lambda: 10, want: 40}, // 10 + ceil(300/10) = 10 + 30
		{dimension: 5, lambda: 8, want: 29},   // 10 + ceil(150/8)  = 10 + 19
	}

	for _, test := range tests {
		config := hansenConvergence()

		tracker := newConvergenceTracker(&config, nil, 1, test.dimension, test.lambda)
		if tracker.historyLimit != test.want {
			t.Errorf("historyLimit(n=%d, lambda=%d) = %d, want %d",
				test.dimension, test.lambda, tracker.historyLimit, test.want)
		}
	}
}

// TestTolFunUsesBestOfGenerationWindow drives real generations through the
// tracker. Hansen's criterion ranges over the best value of each of the last
// 10 + ceil(30n/lambda) generations together with the current generation's
// full spread, so historical intra-generation spread must not keep the run
// alive once the best-of-generation series has gone flat.
func TestTolFunUsesBestOfGenerationWindow(t *testing.T) {
	const tolFun = 1e-9

	newTracker := func() (*convergenceTracker, *strategyState) {
		config := ConvergenceConfig{TolFun: tolFun}

		return newConvergenceTracker(&config, nil, 1, 2, 6), convergenceTestState()
	}

	t.Run("window not yet full", func(t *testing.T) {
		tracker, state := newTracker()

		for iteration := 1; iteration < tracker.historyLimit; iteration++ {
			reason, stop := tracker.observe(
				iteration, Best{Cost: 0}, state, flatPopulation(0, 0),
			)
			if stop {
				t.Fatalf("iteration %d stopped with %q before the window was full",
					iteration, reason)
			}
		}
	})

	t.Run("best of generation still varying", func(t *testing.T) {
		tracker, state := newTracker()

		// Each generation improves by 1e-3, far more than TolFun, so the
		// range over the window stays at 1e-3 even though every single
		// generation is internally flat.
		for iteration := 1; iteration <= 2*tracker.historyLimit; iteration++ {
			best := -1e-3 * float64(iteration)

			reason, stop := tracker.observe(
				iteration, Best{Cost: best}, state, flatPopulation(best, best),
			)
			if stop {
				t.Fatalf("iteration %d stopped with %q while the best value still moved",
					iteration, reason)
			}
		}
	})

	t.Run("flat best of generation stops despite an earlier wide spread", func(t *testing.T) {
		tracker, state := newTracker()

		// Every generation but the last is internally spread across [0, 100]
		// while its best value stays 0. Only the current generation's spread
		// counts, so once that collapses the criterion must fire.
		for iteration := 1; iteration < tracker.historyLimit; iteration++ {
			reason, stop := tracker.observe(
				iteration, Best{Cost: 0}, state, flatPopulation(0, 100),
			)
			if stop {
				t.Fatalf("iteration %d stopped with %q before the window was full",
					iteration, reason)
			}
		}

		reason, stop := tracker.observe(
			tracker.historyLimit, Best{Cost: 0}, state, flatPopulation(0, 0),
		)
		if !stop || reason != TerminationTolFun {
			t.Errorf("full flat window = (%q, %v), want (%q, true)",
				reason, stop, TerminationTolFun)
		}
	})
}

// TestTolFunSkipsNonFiniteScores documents the recovery rule: a non-finite
// score is skipped, never allowed to discard an accumulated window.
func TestTolFunSkipsNonFiniteScores(t *testing.T) {
	newTracker := func() (*convergenceTracker, *strategyState) {
		config := ConvergenceConfig{TolFun: 1e-9}

		return newConvergenceTracker(&config, nil, 1, 2, 6), convergenceTestState()
	}

	fill := func(t *testing.T, tracker *convergenceTracker, state *strategyState) {
		t.Helper()

		for iteration := 1; iteration < tracker.historyLimit; iteration++ {
			if reason, stop := tracker.observe(
				iteration, Best{Cost: 0}, state, flatPopulation(0, 0),
			); stop {
				t.Fatalf("iteration %d stopped early with %q", iteration, reason)
			}
		}
	}

	t.Run("one bad candidate among finite ones", func(t *testing.T) {
		tracker, state := newTracker()
		fill(t, tracker, state)

		reason, stop := tracker.observe(
			tracker.historyLimit, Best{Cost: 0}, state, flatPopulation(0, math.NaN()),
		)
		if !stop || reason != TerminationTolFun {
			t.Errorf("generation with one NaN = (%q, %v), want (%q, true)",
				reason, stop, TerminationTolFun)
		}
	})

	t.Run("whole generation unusable", func(t *testing.T) {
		tracker, state := newTracker()
		fill(t, tracker, state)

		population := flatPopulation(math.NaN(), math.Inf(1))
		if reason, stop := tracker.observe(
			tracker.historyLimit, Best{Cost: 0}, state, population,
		); stop {
			t.Fatalf("unusable generation stopped with %q", reason)
		}

		// The window kept its historyLimit-1 entries, so the very next usable
		// generation completes it instead of restarting the count.
		reason, stop := tracker.observe(
			tracker.historyLimit+1, Best{Cost: 0}, state, flatPopulation(0, 0),
		)
		if !stop || reason != TerminationTolFun {
			t.Errorf("generation after an unusable one = (%q, %v), want (%q, true)",
				reason, stop, TerminationTolFun)
		}
	})
}

// TestTolXIsRelativeToInitialSigma covers Hansen's TolX = 1e-12 * sigma^(0).
// A fine-tuning run started at sigma^(0) = 1e-12 must not be declared
// converged before it has taken a single step.
func TestTolXIsRelativeToInitialSigma(t *testing.T) {
	const initialSigma = 1e-12

	config := ConvergenceConfig{TolX: 1e-12}
	tracker := newConvergenceTracker(&config, nil, initialSigma, 2, 6)
	state := convergenceTestState()
	state.sigma = initialSigma
	population := flatPopulation(1)

	if reason, stop := tracker.observe(1, Best{Cost: 1}, state, population); stop {
		t.Fatalf("run at its initial sigma stopped with %q", reason)
	}

	// The threshold is 1e-12 * 1e-12 = 1e-24; 1e-25 is below it.
	state.sigma = 1e-25

	reason, stop := tracker.observe(2, Best{Cost: 1}, state, population)
	if !stop || reason != TerminationTolX {
		t.Errorf("collapsed distribution = (%q, %v), want (%q, true)",
			reason, stop, TerminationTolX)
	}
}

// TestTolXUpUsesTheInitialDistributionExtent covers "sigma * max(D) increased
// by more than 1e4 over its initial value". Both the initial step size and the
// initial axis scaling take part in that reference value.
func TestTolXUpUsesTheInitialDistributionExtent(t *testing.T) {
	population := flatPopulation(1)
	best := Best{Cost: 1}

	t.Run("non-default initial sigma", func(t *testing.T) {
		config := ConvergenceConfig{TolXUp: 1e4}
		tracker := newConvergenceTracker(&config, nil, 2, 2, 6)
		state := convergenceTestState()
		state.sigma = 2

		if reason, stop := tracker.observe(1, best, state, population); stop {
			t.Fatalf("iteration 1 stopped with %q", reason)
		}

		// The threshold is 2 * 1 * 1e4 = 2e4, so 1.5e4 is still below it.
		state.sigma = 1.5e4
		if reason, stop := tracker.observe(2, best, state, population); stop {
			t.Fatalf("sigma below the threshold stopped with %q", reason)
		}

		state.sigma = 2.5e4

		reason, stop := tracker.observe(3, best, state, population)
		if !stop || reason != TerminationTolXUp {
			t.Errorf("sigma above the threshold = (%q, %v), want (%q, true)",
				reason, stop, TerminationTolXUp)
		}
	})

	t.Run("non-identity initial axis scaling", func(t *testing.T) {
		config := ConvergenceConfig{TolXUp: 1e4}
		tracker := newConvergenceTracker(&config, nil, 1, 2, 6)
		state := convergenceTestState()
		state.d = []float64{1, 100}

		if reason, stop := tracker.observe(1, best, state, population); stop {
			t.Fatalf("iteration 1 stopped with %q", reason)
		}

		// The initial extent is sigma^(0) * max(D^(0)) = 100, so the
		// threshold is 1e6. A tenfold growth to 1e5 is not a 1e4 growth.
		state.sigma = 1e3
		if reason, stop := tracker.observe(2, best, state, population); stop {
			t.Fatalf("extent 1e5 stopped with %q, want no stop below 1e6", reason)
		}

		state.sigma = 1e4

		reason, stop := tracker.observe(3, best, state, population)
		if !stop || reason != TerminationTolXUp {
			t.Errorf("extent 1e6 = (%q, %v), want (%q, true)", reason, stop, TerminationTolXUp)
		}
	})
}

// TestMinIterationsGatesEveryCriterion pins that MinIterations delays all
// criteria uniformly, not just stagnation.
func TestMinIterationsGatesEveryCriterion(t *testing.T) {
	config := hansenConvergence()
	config.MinIterations = 5

	tracker := newConvergenceTracker(&config, nil, 1, 2, 6)
	state := convergenceTestState()
	state.sigma = 1e-13
	population := flatPopulation(1)

	for iteration := 1; iteration < 5; iteration++ {
		if reason, stop := tracker.observe(iteration, Best{Cost: 1}, state, population); stop {
			t.Fatalf("iteration %d stopped with %q before MinIterations", iteration, reason)
		}
	}

	reason, stop := tracker.observe(5, Best{Cost: 1}, state, population)
	if !stop || reason != TerminationTolX {
		t.Errorf("iteration 5 = (%q, %v), want (%q, true)", reason, stop, TerminationTolX)
	}
}

// TestStagnationCountsImprovementsBelowMinImprovement pins that an improvement
// smaller than MinImprovement does not reset the stagnation counter, while one
// above it does.
func TestStagnationCountsImprovementsBelowMinImprovement(t *testing.T) {
	state := convergenceTestState()
	population := flatPopulation(1)

	newTracker := func() *convergenceTracker {
		config := &ConvergenceConfig{MinImprovement: 0.5, StagnationIterations: 3}

		return newConvergenceTracker(config, nil, 1, 2, 6)
	}

	t.Run("improvements below the threshold stagnate", func(t *testing.T) {
		tracker := newTracker()

		// Iteration 1 adopts the first candidate as the reference. Each later
		// iteration improves by 0.1 < 0.5, so the counter climbs to 3 by
		// iteration 4.
		for iteration := 1; iteration < 4; iteration++ {
			cost := 10 - 0.1*float64(iteration)
			if reason, stop := tracker.observe(
				iteration, Best{Cost: cost}, state, population,
			); stop {
				t.Fatalf("iteration %d stopped early with %q", iteration, reason)
			}
		}

		reason, stop := tracker.observe(4, Best{Cost: 9.6}, state, population)
		if !stop || reason != TerminationStagnation {
			t.Errorf("iteration 4 = (%q, %v), want (%q, true)",
				reason, stop, TerminationStagnation)
		}
	})

	t.Run("improvements above the threshold reset the counter", func(t *testing.T) {
		tracker := newTracker()

		for iteration := 1; iteration <= 8; iteration++ {
			cost := 10 - float64(iteration)
			if reason, stop := tracker.observe(
				iteration, Best{Cost: cost}, state, population,
			); stop {
				t.Fatalf("iteration %d stopped with %q while still improving", iteration, reason)
			}
		}
	})
}

func TestConvergenceStagnationAndMinimumIteration(t *testing.T) {
	config := &ConvergenceConfig{
		MinImprovement:       0.5,
		StagnationIterations: 2,
		MinIterations:        4,
	}
	tracker := newConvergenceTracker(config, nil, 1, 2, 6)
	state := convergenceTestState()
	population := flatPopulation(10)

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
	tests := []struct {
		name       string
		axisScales []float64
		want       float64
	}{
		// D holds standard deviations: cond(C) = (10/2)^2.
		{name: "spread", axisScales: []float64{2, 10}, want: 25},
		{name: "isotropic", axisScales: []float64{3, 3, 3}, want: 1},
		{name: "empty", axisScales: nil, want: 0},
		{name: "singular", axisScales: []float64{0, 1}, want: math.Inf(1)},
		{name: "not a number", axisScales: []float64{math.NaN(), 1}, want: math.Inf(1)},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := covarianceConditionNumber(test.axisScales); got != test.want {
				t.Errorf("covarianceConditionNumber(%v) = %v, want %v",
					test.axisScales, got, test.want)
			}
		})
	}
}

// TestCovarianceConditionNumberDoesNotAllocate guards the hot path: the
// condition number is computed two to three times per iteration.
func TestCovarianceConditionNumberDoesNotAllocate(t *testing.T) {
	axisScales := []float64{1, 2, 3, 4}

	allocations := testing.AllocsPerRun(100, func() {
		if covarianceConditionNumber(axisScales) <= 0 {
			t.Error("condition number must be positive")
		}
	})
	if allocations != 0 {
		t.Errorf("covarianceConditionNumber allocated %v times per call, want 0", allocations)
	}
}
