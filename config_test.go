package cmaes

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func testConfig(problemSize int) *Config {
	config := NewDefaultConfig(problemSize)
	config.ObjectiveFunc = func(position []float64) float64 {
		return position[0]
	}
	config.LowerBound = -5
	config.UpperBound = 5

	return config
}

func TestNewDefaultConfig(t *testing.T) {
	config := NewDefaultConfig(10)

	if config.ProblemSize != 10 || len(config.InitialMean) != 10 {
		t.Fatalf("problem shape = (%d, %d), want (10, 10)",
			config.ProblemSize, len(config.InitialMean))
	}

	if config.Lambda != 10 || config.Mu != 5 {
		t.Errorf("population = (%d, %d), want (10, 5)", config.Lambda, config.Mu)
	}

	if config.InitialSigma != defaultInitialSigma {
		t.Errorf("InitialSigma = %v, want %v", config.InitialSigma, defaultInitialSigma)
	}

	if config.BoundaryMethod != BoundaryPenalty || config.CovarianceMode != CovarianceFull {
		t.Errorf("modes = (%q, %q), want (%q, %q)", config.BoundaryMethod,
			config.CovarianceMode, BoundaryPenalty, CovarianceFull)
	}

	if config.MaxIterations != defaultMaxIterations || config.MaxEvaluations != 0 {
		t.Errorf("caps = (%d, %d), want (%d, 0)", config.MaxIterations,
			config.MaxEvaluations, defaultMaxIterations)
	}

	if config.MaxWorkers != runtime.NumCPU() {
		t.Errorf("MaxWorkers = %d, want %d", config.MaxWorkers, runtime.NumCPU())
	}

	if !config.ActiveCMA {
		t.Error("ActiveCMA = false, want true")
	}

	wantConvergence := NewDefaultConvergenceConfig()
	if !reflect.DeepEqual(config.Convergence, wantConvergence) {
		t.Errorf("Convergence = %+v, want %+v", config.Convergence, wantConvergence)
	}

	for index, value := range config.InitialMean {
		if value != 0 {
			t.Errorf("InitialMean[%d] = %v, want 0", index, value)
		}
	}
}

func TestConstructorsReturnIndependentConfigs(t *testing.T) {
	first := NewDefaultConfig(3)
	second := NewDefaultConfig(3)
	first.InitialMean[0] = 42
	first.Convergence.TolX = 42

	if second.InitialMean[0] != 0 {
		t.Fatalf("constructors share InitialMean storage: second[0] = %v", second.InitialMean[0])
	}

	if second.Convergence.TolX != defaultTolX {
		t.Fatalf("constructors share Convergence storage: second TolX = %v", second.Convergence.TolX)
	}
}

func TestConstructorAcceptsInvalidDimensionWithoutPanicking(t *testing.T) {
	for _, problemSize := range []int{0, -1} {
		config := NewDefaultConfig(problemSize)
		config.ObjectiveFunc = func([]float64) float64 { return 0 }

		err := config.Validate()
		if err == nil || !strings.Contains(err.Error(), "problem_size") {
			t.Errorf("NewDefaultConfig(%d).Validate() = %v, want problem_size error",
				problemSize, err)
		}
	}
}

func TestConfigurationPresets(t *testing.T) {
	tests := []struct {
		name           string
		newConfig      func(int) *Config
		wantMode       CovarianceMode
		wantIterations int
	}{
		{"default", NewDefaultConfig, CovarianceFull, defaultMaxIterations},
		{"separable", NewSeparableConfig, CovarianceSeparable, defaultMaxIterations},
		{"high dimensional", NewHighDimensionalConfig, CovarianceSeparable, highDimensionalIterations},
		{"fast convergence", NewFastConvergenceConfig, CovarianceFull, fastConvergenceIterations},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := test.newConfig(10)
			if config.CovarianceMode != test.wantMode {
				t.Errorf("CovarianceMode = %q, want %q", config.CovarianceMode, test.wantMode)
			}

			if config.MaxIterations != test.wantIterations {
				t.Errorf("MaxIterations = %d, want %d", config.MaxIterations, test.wantIterations)
			}

			if config.Lambda != 10 || config.Mu != 5 || config.InitialSigma != defaultInitialSigma {
				t.Errorf("preset changed core defaults: lambda=%d mu=%d sigma=%v",
					config.Lambda, config.Mu, config.InitialSigma)
			}
		})
	}
}

// TestDerivedStrategyParametersGolden pins the output of
// deriveStrategyParameters against future drift. The expected values are NOT
// published constants: Hansen's tutorial (arXiv:1604.00772) gives formulas, not
// a table of evaluated parameters, so these rows are this implementation's own
// verified output, recomputed independently from the formulas in Table 1
// ("Default Strategy Parameters") before being pinned here:
//
//	lambda   = 4 + floor(3 ln n)
//	w'_i     = ln(max(mu, lambda/2) + 1/2) - ln i, w_i = w'_i / sum_j w'_j
//	mu_eff   = 1 / sum_i w_i^2
//	c_sigma  = (mu_eff + 2) / (n + mu_eff + 5)
//	d_sigma  = 1 + c_sigma + 2 max(0, sqrt((mu_eff - 1)/(n + 1)) - 1)
//	c_c      = (4 + mu_eff/n) / (n + 4 + 2 mu_eff/n)
//	c_1      = 2 / ((n + 1.3)^2 + mu_eff)
//	c_mu     = min(1 - c_1, 2 (mu_eff - 2 + 1/mu_eff) / ((n + 2)^2 + mu_eff))
//
// A change here is a change to the search trajectory and must be deliberate.
// The defining invariants of those formulas are checked separately, and
// formula-derived rather than pinned, in TestStrategyParameterInvariants.
func TestDerivedStrategyParametersGolden(t *testing.T) {
	tests := []struct {
		name    string
		weights []float64
		n       int
		lambda  int
		mu      int
		muEff   float64
		cSigma  float64
		dSigma  float64
		cc      float64
		c1      float64
		cmu     float64
	}{
		{
			name: "n=2", n: 2, lambda: 6, mu: 3,
			weights: []float64{0.637042571241217, 0.284570257438033, 0.0783871713207503},
			muEff:   2.02861146461006, cSigma: 0.446204987378317, dSigma: 1.44620498737832,
			cc: 0.624554539026826, c1: 0.154815399896414, cmu: 0.0578590850719163,
		},
		{
			name: "n=10", n: 10, lambda: 10, mu: 5,
			weights: []float64{
				0.456272646903406, 0.270753097001785, 0.16223111715867,
				0.0852335471001645, 0.0255095918359748,
			},
			muEff: 3.1672992814107, cSigma: 0.284428587946367, dSigma: 1.28442858794637,
			cc: 0.294990383035622, c1: 0.0152838245247517, cmu: 0.0201542827612084,
		},
		{
			name: "n=56", n: 56, lambda: 16, mu: 8,
			weights: []float64{
				0.328436208515219, 0.222058828315815, 0.159832049974207,
				0.115681448116411, 0.0814355807697086, 0.0534546697748028,
				0.0297971466168295, 0.00930406791700748,
			},
			muEff: 4.84091450090117, cSigma: 0.103900660444313, dSigma: 1.10390066044431,
			cc: 0.0679117276092178, c1: 0.000608248288162692, cmu: 0.00180921991708185,
		},
		{
			name: "n=100", n: 100, lambda: 17, mu: 8,
			weights: []float64{
				0.315095875267623, 0.215694193800631, 0.157547937633811,
				0.116292512333639, 0.0842923183903696, 0.0581462561668194,
				0.0360400755404608, 0.0168908308666469,
			},
			muEff: 5.09618887861017, cSigma: 0.0644544461610228, dSigma: 1.06445444616102,
			cc: 0.0389134200578418, c1: 0.000194802926953566, cmu: 0.00063260323183747,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := NewDefaultConfig(test.n)
			if config.Lambda != test.lambda || config.Mu != test.mu {
				t.Fatalf("population = (%d, %d), want (%d, %d)",
					config.Lambda, config.Mu, test.lambda, test.mu)
			}

			got := deriveStrategyParameters(config)
			assertFloatClose(t, "muEff", got.muEff, test.muEff)
			assertFloatClose(t, "cSigma", got.cSigma, test.cSigma)
			assertFloatClose(t, "dSigma", got.dSigma, test.dSigma)
			assertFloatClose(t, "cc", got.cc, test.cc)
			assertFloatClose(t, "c1", got.c1, test.c1)
			assertFloatClose(t, "cmu", got.cmu, test.cmu)

			if len(got.weights) != len(test.weights) {
				t.Fatalf("len(weights) = %d, want %d", len(got.weights), len(test.weights))
			}

			for index := range test.weights {
				assertFloatClose(t, "weight", got.weights[index], test.weights[index])
			}
		})
	}
}

// assertFloatClose compares got and want with a relative tolerance and a small
// absolute floor. A single absolute tolerance would be a few ulp for muEff and
// millions of ulp for c1, holding rows of the same table to wildly different
// strictness; the relative form is roughly 45 ulp everywhere.
func assertFloatClose(t *testing.T, name string, got, want float64) {
	t.Helper()

	const (
		relativeTolerance = 1e-14
		absoluteFloor     = 1e-300
	)

	limit := math.Max(absoluteFloor, relativeTolerance*math.Abs(want))
	if math.Abs(got-want) > limit {
		t.Errorf("%s = %.17g, want %.17g (tolerance %g relative)", name, got, want, relativeTolerance)
	}
}

// rawCmu is Hansen's unclamped rank-mu learning rate, restated here so tests can
// check the clamp min(1 - c1, ...) that deriveStrategyParameters applies.
func rawCmu(n, muEff float64) float64 {
	return 2 * (muEff - 2 + 1/muEff) / ((n+2)*(n+2) + muEff)
}

// TestStrategyParameterInvariants checks the properties the formulas must have
// for any dimension, which a table of pinned values cannot catch: a broken
// normalisation, a sign flip, or a learning-rate sum above one all keep the
// table shape intact.
func TestStrategyParameterInvariants(t *testing.T) {
	for _, n := range []int{1, 2, 3, 5, 10, 17, 30, 56, 100, 250} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			config := NewDefaultConfig(n)

			// lambda = 4 + floor(3 ln n) (Hansen, Table 1).
			wantLambda := 4 + int(math.Floor(3*math.Log(float64(n))))
			if config.Lambda != wantLambda {
				t.Fatalf("Lambda = %d, want 4 + floor(3 ln %d) = %d", config.Lambda, n, wantLambda)
			}

			got := deriveStrategyParameters(config)
			assertWeightInvariants(t, got.weights, config.Mu)

			// muEff = 1 / sum(w_i^2), and 1 <= muEff <= mu because the weights
			// are a probability vector.
			squareSum := 0.0
			for _, weight := range got.weights {
				squareSum += weight * weight
			}

			assertFloatClose(t, "muEff", got.muEff, 1/squareSum)

			if got.muEff < 1 || got.muEff > float64(config.Mu) {
				t.Errorf("muEff = %v, want within [1, mu=%d]", got.muEff, config.Mu)
			}

			// The covariance update must remain a convex combination.
			if got.c1+got.cmu > 1 {
				t.Errorf("c1 + cmu = %v, want at most 1", got.c1+got.cmu)
			}

			if got.c1 <= 0 || got.cmu <= 0 {
				t.Errorf("learning rates = (%v, %v), want both positive", got.c1, got.cmu)
			}
		})
	}
}

// assertWeightInvariants checks that weights form a strictly decreasing
// probability vector of length mu, which is what recombination assumes.
func assertWeightInvariants(t *testing.T, weights []float64, mu int) {
	t.Helper()

	if len(weights) != mu {
		t.Fatalf("len(weights) = %d, want mu = %d", len(weights), mu)
	}

	sum := 0.0
	for _, weight := range weights {
		sum += weight
	}

	assertFloatClose(t, "sum(weights)", sum, 1)

	for index, weight := range weights {
		if weight <= 0 {
			t.Errorf("weights[%d] = %v, want strictly positive", index, weight)
		}

		if index > 0 && weight >= weights[index-1] {
			t.Errorf("weights[%d] = %v, want strictly below weights[%d] = %v",
				index, weight, index-1, weights[index-1])
		}
	}
}

// TestDerivedStrategyParameterBranches covers the paths NewDefaultConfig never
// reaches: a caller-raised Mu, the separable rank-mu rescaling, the c_mu clamp
// actually binding, and a one-dimensional problem.
func TestDerivedStrategyParameterBranches(t *testing.T) {
	t.Run("mu above half of lambda", func(t *testing.T) {
		config := testConfig(10)
		config.Lambda = 10
		config.Mu = 8

		got := deriveStrategyParameters(config)
		assertWeightInvariants(t, got.weights, config.Mu)

		// With Mu > Lambda/2 the offset is ln(Mu + 1/2) (purecmaes.m), so the
		// unnormalised weights are ln(8.5) - ln(i). Normalisation cancels in a
		// ratio, which makes this independent of the sum.
		base := math.Log(8.5)
		wantRatio := (base - math.Log(8)) / (base - math.Log(1))
		assertFloatClose(t, "weights[7]/weights[0]", got.weights[7]/got.weights[0], wantRatio)
	})

	t.Run("separable rank-mu rescaling", func(t *testing.T) {
		full := testConfig(10)
		separable := testConfig(10)
		separable.CovarianceMode = CovarianceSeparable

		gotFull := deriveStrategyParameters(full)
		gotSeparable := deriveStrategyParameters(separable)

		// sep-CMA-ES multiplies c_mu by (n + 2)/3, still clamped by 1 - c_1.
		want := math.Min(1-gotFull.c1, gotFull.cmu*(10+2)/3)
		assertFloatClose(t, "separable cmu", gotSeparable.cmu, want)

		if gotSeparable.cmu <= gotFull.cmu {
			t.Errorf("separable cmu = %v, want above full cmu = %v", gotSeparable.cmu, gotFull.cmu)
		}
	})

	t.Run("cmu clamp binds", func(t *testing.T) {
		// A large population on a one-dimensional problem drives the raw rank-mu
		// rate towards 2, well above the 1 - c_1 ceiling.
		config := testConfig(1)
		config.InitialMean = []float64{0}
		config.Lambda = 100
		config.Mu = 50

		got := deriveStrategyParameters(config)
		if rawCmu(1, got.muEff) <= 1-got.c1 {
			t.Fatalf("raw cmu = %v does not exceed 1 - c1 = %v, so the clamp is untested",
				rawCmu(1, got.muEff), 1-got.c1)
		}

		assertFloatClose(t, "clamped cmu", got.cmu, 1-got.c1)
	})

	t.Run("one dimension", func(t *testing.T) {
		config := NewDefaultConfig(1)
		if config.Lambda != 4 || config.Mu != 2 {
			t.Fatalf("population = (%d, %d), want (4, 2)", config.Lambda, config.Mu)
		}

		got := deriveStrategyParameters(config)
		assertWeightInvariants(t, got.weights, config.Mu)

		for name, value := range map[string]float64{
			"muEff": got.muEff, "cSigma": got.cSigma, "dSigma": got.dSigma,
			"cc": got.cc, "c1": got.c1, "cmu": got.cmu,
		} {
			if !isFinite(value) {
				t.Errorf("%s = %v, want a finite value", name, value)
			}
		}
	})
}

func TestValidateAcceptsCompleteConfiguration(t *testing.T) {
	target := 0.0
	config := testConfig(3)
	config.InitialMean = []float64{-2, 0, 2}
	config.MaxEvaluations = 0
	config.Convergence = &ConvergenceConfig{
		TargetCost:           &target,
		TolX:                 1e-12,
		TolFun:               1e-12,
		TolXUp:               1e4,
		ConditionCov:         1e14,
		MinImprovement:       0.01,
		StagnationIterations: 10,
		MinIterations:        2,
		NoEffectAxis:         true,
		NoEffectCoord:        true,
	}
	config.Constraints = &ConstraintConfig{
		Handling:          ConstraintHandlingPenalty,
		PenaltyMethod:     PenaltyQuadratic,
		Inequalities:      []ConstraintFunction{func([]float64) float64 { return 0 }},
		Equalities:        []ConstraintFunction{func([]float64) float64 { return 0 }},
		PenaltyFactor:     2,
		EqualityTolerance: 1e-8,
	}

	err := config.Validate()
	if err != nil {
		t.Fatalf("Validate() = %v, want nil", err)
	}
}

func TestValidateRejectsInvalidConfiguration(t *testing.T) {
	tests := []struct {
		name   string
		want   string
		mutate func(*Config)
	}{
		{"missing objective", "ObjectiveFunc", func(c *Config) { c.ObjectiveFunc = nil }},
		{"zero problem size", "problem_size", func(c *Config) { c.ProblemSize = 0 }},
		{"non-finite lower bound", "finite", func(c *Config) { c.LowerBound = math.NaN() }},
		{"non-finite upper bound", "finite", func(c *Config) { c.UpperBound = math.Inf(1) }},
		{"equal bounds", "lower_bound", func(c *Config) { c.LowerBound, c.UpperBound = 1, 1 }},
		{"inverted bounds", "lower_bound", func(c *Config) { c.LowerBound, c.UpperBound = 2, 1 }},
		{"overflowing bound span", "must be finite", func(c *Config) {
			c.LowerBound, c.UpperBound = -math.MaxFloat64, math.MaxFloat64
		}},
		{"zero sigma", "initial_sigma", func(c *Config) { c.InitialSigma = 0 }},
		{"infinite sigma", "initial_sigma", func(c *Config) { c.InitialSigma = math.Inf(1) }},
		{"mean length", "initial_mean", func(c *Config) { c.InitialMean = []float64{0} }},
		{"non-finite mean", "initial_mean", func(c *Config) { c.InitialMean[1] = math.NaN() }},
		{"mean below the box", "initial_mean", func(c *Config) { c.InitialMean[0] = -20 }},
		{"mean above the box", "initial_mean", func(c *Config) { c.InitialMean[2] = 20 }},
		{"mean outside a per-dimension box", "initial_mean", func(c *Config) {
			c.UpperBounds = []float64{5, 5, 0.5}
			c.InitialMean[2] = 1
		}},
		{"short lower bounds", "lower_bounds", func(c *Config) {
			c.LowerBounds = []float64{-1, -1}
		}},
		{"long upper bounds", "upper_bounds", func(c *Config) {
			c.UpperBounds = []float64{1, 1, 1, 1}
		}},
		{"non-finite per-dimension bound", "lower_bounds", func(c *Config) {
			c.LowerBounds = []float64{-1, math.Inf(-1), -1}
		}},
		{"inverted per-dimension bounds", "lower_bound[1]", func(c *Config) {
			c.LowerBounds = []float64{-1, 3, -1}
			c.UpperBounds = []float64{1, 2, 1}
		}},
		{"evaluation budget below one generation", "max_evaluations", func(c *Config) {
			c.MaxEvaluations = c.Lambda - 1
		}},
		{"small lambda", "lambda", func(c *Config) { c.Lambda = 1 }},
		{"zero mu", "mu", func(c *Config) { c.Mu = 0 }},
		{"mu above lambda", "mu", func(c *Config) { c.Mu = c.Lambda + 1 }},
		{"zero iterations", "max_iterations", func(c *Config) { c.MaxIterations = 0 }},
		{"negative evaluations", "max_evaluations", func(c *Config) { c.MaxEvaluations = -1 }},
		{"negative workers", "max_workers", func(c *Config) { c.MaxWorkers = -1 }},
		{"seed and rand", "mutually exclusive", func(c *Config) {
			seed := int64(7)
			c.Seed = &seed
			c.Rand = rand.New(rand.NewSource(seed))
		}},
		{"unknown boundary", "boundary_method", func(c *Config) {
			c.BoundaryMethod = BoundaryMethod("teleport")
		}},
		{"unknown covariance", "covariance_mode", func(c *Config) {
			c.CovarianceMode = CovarianceMode("sparse")
		}},
		{"non-finite target", "target_cost", func(c *Config) {
			target := math.NaN()
			c.Convergence = &ConvergenceConfig{TargetCost: &target}
		}},
		{"negative TolX", "tol_x", func(c *Config) {
			c.Convergence = &ConvergenceConfig{TolX: -1}
		}},
		{"infinite TolFun", "tol_fun", func(c *Config) {
			c.Convergence = &ConvergenceConfig{TolFun: math.Inf(1)}
		}},
		{"negative TolXUp", "tol_x_up", func(c *Config) {
			c.Convergence = &ConvergenceConfig{TolXUp: -1}
		}},
		{"infinite condition", "condition_cov", func(c *Config) {
			c.Convergence = &ConvergenceConfig{ConditionCov: math.Inf(1)}
		}},
		{"negative improvement", "min_improvement", func(c *Config) {
			c.Convergence = &ConvergenceConfig{MinImprovement: -1}
		}},
		{"infinite improvement", "min_improvement", func(c *Config) {
			c.Convergence = &ConvergenceConfig{MinImprovement: math.Inf(1)}
		}},
		{"negative stagnation", "stagnation_iterations", func(c *Config) {
			c.Convergence = &ConvergenceConfig{StagnationIterations: -1}
		}},
		{"negative minimum iterations", "min_iterations", func(c *Config) {
			c.Convergence = &ConvergenceConfig{MinIterations: -1}
		}},
		{"minimum iterations beyond cap", "min_iterations", func(c *Config) {
			c.Convergence = &ConvergenceConfig{MinIterations: c.MaxIterations + 1}
		}},
		{"unknown constraint handling", "handling", func(c *Config) {
			c.Constraints = &ConstraintConfig{Handling: ConstraintHandlingMethod("rank")}
		}},
		{"unknown penalty method", "penalty_method", func(c *Config) {
			c.Constraints = &ConstraintConfig{PenaltyMethod: PenaltyMethod("cubic")}
		}},
		{"negative penalty factor", "penalty_factor", func(c *Config) {
			c.Constraints = &ConstraintConfig{PenaltyFactor: -1}
		}},
		{"infinite penalty factor", "penalty_factor", func(c *Config) {
			c.Constraints = &ConstraintConfig{PenaltyFactor: math.Inf(1)}
		}},
		{"negative equality tolerance", "equality_tolerance", func(c *Config) {
			c.Constraints = &ConstraintConfig{EqualityTolerance: -1}
		}},
		{"infinite equality tolerance", "equality_tolerance", func(c *Config) {
			c.Constraints = &ConstraintConfig{EqualityTolerance: math.Inf(1)}
		}},
		{"nil inequality", "inequality constraint", func(c *Config) {
			c.Constraints = &ConstraintConfig{Inequalities: []ConstraintFunction{nil}}
		}},
		{"nil equality", "equality constraint", func(c *Config) {
			c.Constraints = &ConstraintConfig{Equalities: []ConstraintFunction{nil}}
		}},
		{"zero penalty factor", "positive", func(c *Config) {
			c.Constraints = &ConstraintConfig{Handling: ConstraintHandlingPenalty}
		}},
		{"configured constraints without functions", "no constraint functions", func(c *Config) {
			c.Constraints = &ConstraintConfig{
				Handling:      ConstraintHandlingPenalty,
				PenaltyMethod: PenaltyQuadratic,
				PenaltyFactor: 9,
			}
		}},
		{"feasibility handling without functions", "no constraint functions", func(c *Config) {
			c.Constraints = &ConstraintConfig{Handling: ConstraintHandlingFeasibility}
		}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := testConfig(3)
			test.mutate(config)

			err := config.Validate()
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Validate() = %v, want error containing %q", err, test.want)
			}
		})
	}
}

func TestCoordinateBoundsBroadcastAndPrecedence(t *testing.T) {
	tests := []struct {
		name       string
		lowers     []float64
		uppers     []float64
		wantLowers []float64
		wantUppers []float64
	}{
		{
			name:       "scalars broadcast",
			wantLowers: []float64{-5, -5, -5}, wantUppers: []float64{5, 5, 5},
		},
		{
			name:   "slices win on both sides",
			lowers: []float64{-1, -2, -3}, uppers: []float64{1, 2, 3},
			wantLowers: []float64{-1, -2, -3}, wantUppers: []float64{1, 2, 3},
		},
		{
			name:   "sides resolve independently",
			lowers: []float64{0, 0, 0},
			// The upper side keeps the scalar because UpperBounds stays nil.
			wantLowers: []float64{0, 0, 0}, wantUppers: []float64{5, 5, 5},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := testConfig(3)
			config.LowerBounds = test.lowers
			config.UpperBounds = test.uppers

			for coordinate := range 3 {
				lower, upper := coordinateBounds(config, coordinate)
				if lower != test.wantLowers[coordinate] || upper != test.wantUppers[coordinate] {
					t.Errorf("coordinateBounds(%d) = (%v, %v), want (%v, %v)", coordinate,
						lower, upper, test.wantLowers[coordinate], test.wantUppers[coordinate])
				}

				width := coordinateBoxWidth(config, coordinate)
				wantWidth := test.wantUppers[coordinate] - test.wantLowers[coordinate]

				if width != wantWidth {
					t.Errorf("coordinateBoxWidth(%d) = %v, want %v", coordinate, width, wantWidth)
				}
			}

			err := config.Validate()
			if err != nil {
				t.Fatalf("Validate() = %v, want nil", err)
			}
		})
	}
}

func TestCoordinateBoundsFallsBackOutsideConfiguredSlices(t *testing.T) {
	config := testConfig(3)
	config.LowerBounds = []float64{-1, -1, -1}

	// A coordinate outside the slice, which validation would reject, must not
	// panic: coordinateBounds is called before Validate by construction.
	lower, upper := coordinateBounds(config, 7)
	if lower != config.LowerBound || upper != config.UpperBound {
		t.Errorf("coordinateBounds(7) = (%v, %v), want the scalars (%v, %v)",
			lower, upper, config.LowerBound, config.UpperBound)
	}
}

func TestValidateAcceptsEvaluationBudgetOfExactlyOneGeneration(t *testing.T) {
	config := testConfig(3)
	config.MaxEvaluations = config.Lambda

	err := config.Validate()
	if err != nil {
		t.Fatalf("Validate() = %v, want nil", err)
	}
}

func TestValidateAcceptsZeroConstraintConfig(t *testing.T) {
	config := testConfig(3)
	config.Constraints = &ConstraintConfig{}

	err := config.Validate()
	if err != nil {
		t.Fatalf("Validate() with an unconfigured ConstraintConfig = %v, want nil", err)
	}
}

func TestValidateNilConfig(t *testing.T) {
	var config *Config

	err := config.Validate()
	if err == nil {
		t.Fatal("nil Config.Validate() = nil, want an error")
	}
}

func TestTerminationReasonValues(t *testing.T) {
	reasons := map[TerminationReason]string{
		TerminationMaxIterations:   "maximum_iterations",
		TerminationMaxEvaluations:  "maximum_evaluations",
		TerminationTargetCost:      "target_cost",
		TerminationStagnation:      "stagnation",
		TerminationTolX:            "tol_x",
		TerminationTolFun:          "tol_fun",
		TerminationTolXUp:          "tol_x_up",
		TerminationConditionNumber: "condition_number",
		TerminationNoEffectAxis:    "no_effect_axis",
		TerminationNoEffectCoord:   "no_effect_coord",
		TerminationCancelled:       "cancelled", //nolint:misspell // Public value follows PLAN.md.
	}

	for reason, want := range reasons {
		if string(reason) != want {
			t.Errorf("reason = %q, want %q", reason, want)
		}
	}
}
