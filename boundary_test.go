package cmaes

import (
	"math"
	"testing"
)

// nonUniformConfig returns a box whose three coordinates disagree in width, in
// sign and in whether they straddle the origin, so that any code path that
// still broadcasts a single scalar bound is visibly wrong.
func nonUniformConfig(method BoundaryMethod) *Config {
	return &Config{
		BoundaryMethod: method,
		LowerBounds:    []float64{-10, -10, 0.1},
		UpperBounds:    []float64{10, 10, 5},
		ProblemSize:    3,
	}
}

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

// TestBoundaryRepairsRespectPerCoordinateBounds checks clamp and reflect in a
// box whose third coordinate is [0.1, 5]. The reflected values are computed by
// hand: reflecting -12 across the lower bound -10 gives -8, and reflecting -1
// across the lower bound 0.1 gives 0.1 + (0.1 - -1) = 1.2.
func TestBoundaryRepairsRespectPerCoordinateBounds(t *testing.T) {
	tests := []struct {
		name   string
		method BoundaryMethod
		want   []float64
	}{
		{name: "clamp", method: BoundaryClamp, want: []float64{-10, 3, 0.1}},
		{name: "reflect", method: BoundaryReflect, want: []float64{-8, 3, 1.2}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := &strategyState{m: []float64{0, 0, 0}, sigma: 1}
			population := []candidate{{
				x: []float64{-12, 3, -1},
				y: make([]float64, 3),
			}}

			applyBoundaryHandling(population, state, nonUniformConfig(test.method))
			assertVectorClose(t, population[0].x, test.want, 1e-15)
		})
	}
}

// TestLinearQuadraticShoulderMargins pins Hansen's al and au. cma.py's
// BoxConstraintsLinQuadTransformation uses min((ub-lb)/2, (1+|b|)/20) for each
// bound separately, so the box [-1, 1] gets al = au = 2/20 = 0.1, the box
// [0.1, 5] gets al = 1.1/20 = 0.055 and au = 6/20 = 0.3, and the narrow box
// [-0.05, 0.05] is capped at half its width because 1.05/20 exceeds it.
func TestLinearQuadraticShoulderMargins(t *testing.T) {
	tests := []struct {
		name      string
		lower     float64
		upper     float64
		wantLower float64
		wantUpper float64
	}{
		{name: "symmetric unit box", lower: -1, upper: 1, wantLower: 0.1, wantUpper: 0.1},
		{name: "asymmetric box", lower: 0.1, upper: 5, wantLower: 0.055, wantUpper: 0.3},
		{name: "half width caps", lower: -0.05, upper: 0.05, wantLower: 0.05, wantUpper: 0.05},
		{name: "bound at origin", lower: 0, upper: 20, wantLower: 0.05, wantUpper: 1.05},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			lowerMargin, upperMargin := shoulderMargins(test.lower, test.upper)
			if math.Abs(lowerMargin-test.wantLower) > 1e-15 ||
				math.Abs(upperMargin-test.wantUpper) > 1e-15 {
				t.Errorf("shoulderMargins(%v, %v) = (%v, %v), want (%v, %v)",
					test.lower, test.upper, lowerMargin, upperMargin,
					test.wantLower, test.wantUpper)
			}
		})
	}
}

// TestHansenLinearQuadraticTransformation checks the transformation on the box
// [-1, 1], where Hansen's margins are al = au = (1 + 1)/20 = 0.1. The quadratic
// shoulder is x -> lb + (x - (lb - al))^2 / (4 al), so the bound itself maps to
// lb + al^2/(4 al) = lb + al/4, and lb - al maps exactly to lb. The upper side
// is the mirror image.
func TestHansenLinearQuadraticTransformation(t *testing.T) {
	const margin = 0.1

	tests := []struct {
		name      string
		value     float64
		wantValue float64
	}{
		{name: "identity", value: 0, wantValue: 0},
		{name: "identity inside shoulder", value: 0.5, wantValue: 0.5},
		{name: "lower shoulder", value: -1, wantValue: -1 + margin/4},
		{name: "lower bound", value: -1 - margin, wantValue: -1},
		{name: "upper shoulder", value: 1, wantValue: 1 - margin/4},
		{name: "upper bound", value: 1 + margin, wantValue: 1},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			value := transformBounded(test.value, -1, 1)
			if math.Abs(value-test.wantValue) > 1e-15 {
				t.Errorf("transformBounded(%v) = %v, want %v",
					test.value, value, test.wantValue)
			}
		})
	}

	for _, value := range []float64{-1e6, -17, -1.2, 1.2, 19, 1e6} {
		transformed := transformBounded(value, -1, 1)
		if transformed < -1 || transformed > 1 {
			t.Errorf("transformBounded(%v) = %v, outside [-1, 1]", value, transformed)
		}
	}
}

// TestTransformBoundedIsPerCoordinate checks the asymmetric box [0.1, 5], where
// al = 0.055 and au = 0.3. The lower bound maps to 0.1 + al/4 = 0.11375 and the
// upper bound to 5 - au/4 = 4.925; a coordinate handled with a shared scalar
// margin could not produce both.
func TestTransformBoundedIsPerCoordinate(t *testing.T) {
	tests := []struct {
		name      string
		value     float64
		wantValue float64
	}{
		{name: "lower bound", value: 0.1, wantValue: 0.1 + 0.055/4},
		{name: "lower shoulder start", value: 0.1 - 0.055, wantValue: 0.1},
		{name: "interior", value: 2.5, wantValue: 2.5},
		{name: "upper bound", value: 5, wantValue: 5 - 0.3/4},
		{name: "upper shoulder start", value: 5 + 0.3, wantValue: 5},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			value := transformBounded(test.value, 0.1, 5)
			if math.Abs(value-test.wantValue) > 1e-15 {
				t.Errorf("transformBounded(%v, 0.1, 5) = %v, want %v",
					test.value, value, test.wantValue)
			}
		})
	}

	for _, value := range []float64{-1e4, -3, 0, 5.5, 41, 1e4} {
		transformed := transformBounded(value, 0.1, 5)
		if transformed < 0.1 || transformed > 5 {
			t.Errorf("transformBounded(%v, 0.1, 5) = %v, outside [0.1, 5]", value, transformed)
		}
	}
}

// TestPenaltyBoundaryUsesPerCoordinateBounds pushes one candidate past the
// upper bound of the narrow third coordinate only, where au = 0.3. Its value
// 5.6 is ub + 2 au, one shoulder width past the end of the shoulder, so the
// recorded squared deviation is 0.3^2 = 0.09. The transformation reflects it to
// 5.3 - 0.3 = 5.0 and the shoulder then maps that to 5 - 0.3/4 = 4.925. Under a
// shared scalar box of [-10, 10] the same value would be interior.
func TestPenaltyBoundaryUsesPerCoordinateBounds(t *testing.T) {
	state := &strategyState{m: []float64{0, 0, 2}, sigma: 1}
	population := []candidate{{x: []float64{1, -2, 5.6}, y: make([]float64, 3)}}

	applyBoundaryHandling(population, state, nonUniformConfig(BoundaryPenalty))

	assertVectorClose(t, population[0].evaluatedPosition(), []float64{1, -2, 4.925}, 1e-15)

	if math.Abs(population[0].boundaryDistance-0.09) > 1e-15 {
		t.Errorf("boundaryDistance = %v, want 0.09", population[0].boundaryDistance)
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
	config := &Config{
		BoundaryMethod: BoundaryPenalty,
		LowerBound:     -10,
		UpperBound:     10,
		ProblemSize:    2,
		Lambda:         6,
		Mu:             3,
	}

	applyBoundaryHandling(population, state, config)
	population[0].cost = 7
	newBoundaryPenaltyState(config).assign(population, state, config)

	assertVectorClose(t, population[0].x, wantX, 0)
	assertVectorClose(t, population[0].y, wantY, 0)
	assertVectorClose(t, population[0].evaluatedPosition(), wantX, 0)

	if population[0].boundaryDistance != 0 || population[0].boundaryPenalty != 0 {
		t.Errorf("interior penalty = (%v, %v), want zero",
			population[0].boundaryDistance, population[0].boundaryPenalty)
	}
}

// penaltyFixture builds a two-dimensional generation on the box [-1, 1] with an
// identity covariance and unit step size, so that sigma^2 C_ii = 1 in both
// coordinates. Hansen's shoulder width here is al = au = 0.1, so the penalty is
// measured from the interval [-1.1, 1.1]. The mean sits at 1.2, one shoulder
// width past its end, which is what lets the weights initialize at all. Exactly
// one candidate is out there too, also at 1.2, giving it a deviation of 0.1 in
// its first coordinate and none elsewhere.
func penaltyFixture(costScale float64) (*Config, *strategyState, []candidate) {
	config := &Config{
		BoundaryMethod: BoundaryPenalty,
		LowerBound:     -1,
		UpperBound:     1,
		ProblemSize:    2,
		Lambda:         6,
		Mu:             3,
	}
	state := &strategyState{
		m:     []float64{1.2, 0},
		c:     identityMatrix(2),
		d:     []float64{1, 1},
		b:     identityMatrix(2),
		sigma: 1,
	}
	positions := [][]float64{{1.2, 0}, {0.5, 0}, {0, 0.25}, {-0.5, 0}}
	costs := []float64{1, 2, 5, 9}

	population := make([]candidate, len(positions))
	for index, position := range positions {
		population[index] = candidate{
			x:    append([]float64(nil), position...),
			y:    make([]float64, 2),
			cost: costScale * costs[index],
		}
	}

	applyBoundaryHandling(population, state, config)

	return config, state, population
}

// TestBoundaryPenaltySeedsWeightsFromNormalizedFitnessSpread pins the published
// initialization. The four costs 1, 2, 5 and 9 give the reference interquartile
// indices (4+1)/4 = 1 and 3(4+1)/4 = 3, an interquartile range of 9 - 2 = 7.
// Both coordinate variances are sigma^2 C_ii = 1, so the normalized spread is
// also 7 and each weight is seeded at gamma = 2 * 7 = 14. The out-of-bounds
// candidate deviates by 0.1 in one of two coordinates, so its penalty is
// 14 * 0.1^2 / 2 = 0.07.
func TestBoundaryPenaltySeedsWeightsFromNormalizedFitnessSpread(t *testing.T) {
	config, state, population := penaltyFixture(1)
	penalty := newBoundaryPenaltyState(config)

	penalty.assign(population, state, config)

	assertVectorClose(t, penalty.gamma, []float64{14, 14}, 1e-12)

	if math.Abs(population[0].boundaryPenalty-0.07) > 1e-12 {
		t.Errorf("out-of-bounds penalty = %v, want 0.07", population[0].boundaryPenalty)
	}

	for index := 1; index < len(population); index++ {
		if population[index].boundaryPenalty != 0 {
			t.Errorf("interior penalty[%d] = %v, want 0",
				index, population[index].boundaryPenalty)
		}
	}
}

// TestBoundaryPenaltyIsInvariantToObjectiveScaling exercises the property the
// scheme exists for: multiplying the objective by a positive constant must
// multiply the penalty by the same constant, so that cost + penalty keeps its
// ranking. The two runs differ only in the scale of the costs.
func TestBoundaryPenaltyIsInvariantToObjectiveScaling(t *testing.T) {
	const scale = 1e-9

	baseConfig, baseState, basePopulation := penaltyFixture(1)
	newBoundaryPenaltyState(baseConfig).assign(basePopulation, baseState, baseConfig)

	scaledConfig, scaledState, scaledPopulation := penaltyFixture(scale)
	newBoundaryPenaltyState(scaledConfig).assign(scaledPopulation, scaledState, scaledConfig)

	for index := range basePopulation {
		want := scale * basePopulation[index].boundaryPenalty
		got := scaledPopulation[index].boundaryPenalty

		if math.Abs(got-want) > 1e-12*math.Abs(want)+1e-300 {
			t.Errorf("penalty[%d] = %v, want %v", index, got, want)
		}
	}
}

// TestBoundaryPenaltyWeightsAreScaleFreeWithoutFitnessSpread covers the
// degenerate generations: a flat one, and one whose costs are all non-finite.
// Neither carries any information about the scale of the objective, so no
// constant in objective units would be defensible; the weights must stay where
// they were. Before anything has been learned that means no penalty at all, and
// once weights exist a flat generation must leave them alone.
func TestBoundaryPenaltyWeightsAreScaleFreeWithoutFitnessSpread(t *testing.T) {
	degenerate := []struct {
		name  string
		costs []float64
	}{
		{name: "flat", costs: []float64{3, 3, 3, 3}},
		{name: "all non-finite", costs: []float64{
			math.NaN(), math.Inf(1), math.Inf(1), math.NaN(),
		}},
	}

	for _, test := range degenerate {
		t.Run(test.name+" leaves fresh weights at zero", func(t *testing.T) {
			config, state, population := penaltyFixture(1)
			for index := range population {
				population[index].cost = test.costs[index]
			}

			penalty := newBoundaryPenaltyState(config)
			penalty.assign(population, state, config)

			assertVectorClose(t, penalty.gamma, []float64{0, 0}, 0)

			for index := range population {
				if population[index].boundaryPenalty != 0 {
					t.Errorf("penalty[%d] = %v, want 0",
						index, population[index].boundaryPenalty)
				}
			}
		})

		t.Run(test.name+" preserves learned weights", func(t *testing.T) {
			config, state, population := penaltyFixture(1)
			penalty := newBoundaryPenaltyState(config)
			penalty.assign(population, state, config)
			learned := append([]float64(nil), penalty.gamma...)

			_, _, flat := penaltyFixture(1)
			for index := range flat {
				flat[index].cost = test.costs[index]
			}

			penalty.assign(flat, state, config)

			assertVectorClose(t, penalty.gamma, learned, 0)

			if math.Abs(flat[0].boundaryPenalty-population[0].boundaryPenalty) > 1e-15 {
				t.Errorf("penalty on a degenerate generation = %v, want %v",
					flat[0].boundaryPenalty, population[0].boundaryPenalty)
			}
		})
	}
}

// TestBoundaryPenaltyWeightsAdaptToAnOutOfBoundsMean is the cross-generation
// behavior the scheme is built around. While the distribution mean stays far
// enough outside the box, the weight of the offending coordinate must keep
// growing generation over generation; a coordinate whose mean is inside must be
// left alone, and once the mean returns the growth must stop.
func TestBoundaryPenaltyWeightsAdaptToAnOutOfBoundsMean(t *testing.T) {
	config, state, population := penaltyFixture(1)
	penalty := newBoundaryPenaltyState(config)

	// The mean is 40 standard deviations past the upper bound of the first
	// coordinate, well beyond the three-sigma threshold, and inside the box in
	// the second.
	state.m = []float64{41, 0}

	penalty.assign(population, state, config)
	seeded := append([]float64(nil), penalty.gamma...)

	previous := seeded[0]

	for generation := range 5 {
		_, _, next := penaltyFixture(1)
		penalty.assign(next, state, config)

		if penalty.gamma[0] <= previous {
			t.Fatalf("generation %d: gamma[0] = %v, want growth beyond %v",
				generation, penalty.gamma[0], previous)
		}

		previous = penalty.gamma[0]
	}

	if penalty.gamma[1] != seeded[1] {
		t.Errorf("gamma[1] = %v, want the seeded %v; an in-bounds coordinate must not adapt",
			penalty.gamma[1], seeded[1])
	}

	// With the mean back inside the box the weights must hold steady.
	state.m = []float64{0.5, 0}

	held := append([]float64(nil), penalty.gamma...)

	_, _, settled := penaltyFixture(1)
	penalty.assign(settled, state, config)

	assertVectorClose(t, penalty.gamma, held, 0)
}

// TestBoundaryPenaltyShrinksWithTheDistribution checks the sigma^2 C_ii
// normalization. Contracting the distribution by a factor of ten while the
// objective keeps the same spread must raise the weights by a hundred, because
// the weights are measured per squared search-space unit and the units have
// shrunk. Without that the penalty would fade away exactly when the strategy
// becomes able to resolve it.
func TestBoundaryPenaltyShrinksWithTheDistribution(t *testing.T) {
	wide, wideState, widePopulation := penaltyFixture(1)
	widePenalty := newBoundaryPenaltyState(wide)
	widePenalty.assign(widePopulation, wideState, wide)

	narrow, narrowState, narrowPopulation := penaltyFixture(1)
	narrowState.sigma = 0.1
	narrowPenalty := newBoundaryPenaltyState(narrow)
	narrowPenalty.assign(narrowPopulation, narrowState, narrow)

	for coordinate := range widePenalty.gamma {
		want := 100 * widePenalty.gamma[coordinate]
		if math.Abs(narrowPenalty.gamma[coordinate]-want) > 1e-9*want {
			t.Errorf("gamma[%d] at sigma 0.1 = %v, want %v",
				coordinate, narrowPenalty.gamma[coordinate], want)
		}
	}
}

// TestPenaltyBoundaryDoesNotDistortSigmaOnInteriorProblem optimizes a sphere
// whose optimum sits at the center of the box with an initial step size five
// times the box's half width, so most of the early samples land outside it and
// the boundary handling does real work. The run must still converge onto the
// interior optimum: it has to reach the optimum, stop on a convergence
// criterion rather than exhausting its budget, and contract its step size by
// orders of magnitude instead of being held open by a penalty that keeps
// growing.
func TestPenaltyBoundaryDoesNotDistortSigmaOnInteriorProblem(t *testing.T) {
	methods := []BoundaryMethod{BoundaryClamp, BoundaryReflect, BoundaryPenalty}

	for _, method := range methods {
		t.Run(string(method), func(t *testing.T) {
			config := optimizationConfig(4, 402, sphere)
			config.LowerBound = -1
			config.UpperBound = 1
			config.InitialSigma = 5
			config.MaxIterations = 2000
			config.BoundaryMethod = method

			result, err := Optimize(config)
			if err != nil {
				t.Fatalf("Optimize: %v", err)
			}

			if result.GlobalBest.Cost > 1e-12 {
				t.Errorf("interior cost = %g, want <= 1e-12", result.GlobalBest.Cost)
			}

			if result.TerminationReason == TerminationMaxIterations {
				t.Errorf("termination = %v, want a convergence criterion",
					result.TerminationReason)
			}

			finalSigma := result.SigmaHistory[len(result.SigmaHistory)-1]
			if finalSigma > 1e-2*config.InitialSigma {
				t.Errorf("final sigma = %g, want <= %g; the step size did not contract",
					finalSigma, 1e-2*config.InitialSigma)
			}
		})
	}
}

// TestOptimizeFindsBoundActiveOptimumForEveryBoundaryMethod drives the optimum
// outside a non-uniform box, so the constrained optimum is the corner
// (1, 1, 5) and its cost is (2-1)^2 + (2-1)^2 + (10-5)^2 = 27.
//
// It runs the shipped defaults, active CMA included. Repairing a genotype and
// then subtracting variance along the repaired direction used to send reflect
// away from every bound-active optimum, so this is the regression guard for
// the genotype/phenotype split that adaptationStep draws.
func TestOptimizeFindsBoundActiveOptimumForEveryBoundaryMethod(t *testing.T) {
	target := []float64{2, 2, 10}
	lower := []float64{-1, -1, 0.1}
	upper := []float64{1, 1, 5}
	methods := []BoundaryMethod{BoundaryClamp, BoundaryReflect, BoundaryPenalty}

	objective := func(position []float64) float64 {
		var cost float64

		for coordinate, value := range position {
			delta := value - target[coordinate]
			cost += delta * delta
		}

		return cost
	}

	for _, method := range methods {
		t.Run(string(method), func(t *testing.T) {
			for seed := int64(401); seed < 421; seed++ {
				config := optimizationConfig(3, seed, objective)
				config.LowerBounds = lower
				config.UpperBounds = upper
				config.InitialMean = []float64{0, 0, 2.5}
				config.InitialSigma = 0.8
				config.MaxIterations = 500
				config.BoundaryMethod = method

				result, err := Optimize(config)
				if err != nil {
					t.Fatalf("seed %d: Optimize: %v", seed, err)
				}

				if result.GlobalBest.Cost > 27.001 {
					t.Errorf("seed %d: bound-active cost = %g, want <= 27.001",
						seed, result.GlobalBest.Cost)
				}
			}
		})
	}
}

// TestRepairKeepsSampledStepForAdaptation checks the genotype/phenotype split
// the covariance update depends on: after a repair, y describes the step to the
// position the objective is evaluated at, while adaptationStep still describes
// the step the distribution drew.
func TestRepairKeepsSampledStepForAdaptation(t *testing.T) {
	tests := []struct {
		name      string
		method    BoundaryMethod
		wantX     []float64
		wantY     []float64
		repairing bool
	}{
		{
			name:      "clamp",
			method:    BoundaryClamp,
			wantX:     []float64{2, 1, 5},
			wantY:     []float64{1, 0, 1.5},
			repairing: true,
		},
		{
			name:      "reflect",
			method:    BoundaryReflect,
			wantX:     []float64{2, 1, 3},
			wantY:     []float64{1, 0, 0.5},
			repairing: true,
		},
		{
			name:      "penalty",
			method:    BoundaryPenalty,
			wantX:     []float64{2, 5, 7},
			wantY:     []float64{1, 2, 2.5},
			repairing: false,
		},
	}

	for _, test := range tests {
		t.Run(string(test.method), func(t *testing.T) {
			state := &strategyState{m: []float64{0, 1, 2}, sigma: 2}
			sampledY := []float64{1, 2, 2.5}
			population := []candidate{{
				x:        []float64{2, 5, 7},
				y:        append([]float64(nil), sampledY...),
				sampledY: nil,
			}}
			population[0].sampledY = population[0].y
			config := &Config{
				BoundaryMethod: test.method,
				LowerBounds:    []float64{-1, -1, 1},
				UpperBounds:    []float64{3, 1, 5},
				ProblemSize:    3,
			}

			applyBoundaryHandling(population, state, config)

			assertVectorClose(t, population[0].x, test.wantX, 1e-15)
			assertVectorClose(t, population[0].y, test.wantY, 1e-15)
			assertVectorClose(t, population[0].adaptationStep(), sampledY, 0)

			repaired := population[0].adaptationStep()
			if test.repairing == vectorsEqual(repaired, population[0].y) {
				t.Errorf("repaired = %v, adaptation step = %v; repairing = %v",
					population[0].y, repaired, test.repairing)
			}
		})
	}
}

// TestInitialPopulationAdaptsToTheSuppliedStep checks the one case where the
// repaired step is the sampled step: a caller-supplied position replaces the
// draw outright, so adaptation must learn the step towards it rather than the
// discarded draw.
func TestInitialPopulationAdaptsToTheSuppliedStep(t *testing.T) {
	state := &strategyState{m: []float64{1, 2}, sigma: 0.5}
	population := []candidate{{
		x: []float64{9, 9},
		y: []float64{16, 14},
	}}
	population[0].sampledY = population[0].y

	applyInitialPopulation(population, state, [][]float64{{2, 4}})

	assertVectorClose(t, population[0].x, []float64{2, 4}, 0)
	assertVectorClose(t, population[0].y, []float64{2, 4}, 0)
	assertVectorClose(t, population[0].adaptationStep(), []float64{2, 4}, 0)
}

func vectorsEqual(left, right []float64) bool {
	if len(left) != len(right) {
		return false
	}

	for index, value := range left {
		if value != right[index] {
			return false
		}
	}

	return true
}
