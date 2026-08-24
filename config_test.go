package cmaes

import (
	"math"
	"math/rand"
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

	if second.InitialMean[0] != 0 {
		t.Fatalf("constructors share InitialMean storage: second[0] = %v", second.InitialMean[0])
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

func TestDerivedStrategyParameters(t *testing.T) {
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

func assertFloatClose(t *testing.T, name string, got, want float64) {
	t.Helper()

	const tolerance = 1e-14
	if math.Abs(got-want) > tolerance {
		t.Errorf("%s = %.17g, want %.17g (tolerance %g)", name, got, want, tolerance)
	}
}

func TestValidateAcceptsCompleteConfiguration(t *testing.T) {
	target := 0.0
	config := testConfig(3)
	config.InitialMean = []float64{-20, 0, 20}
	config.MaxEvaluations = 0
	config.Convergence = &ConvergenceConfig{
		TargetCost:           &target,
		MinImprovement:       0.01,
		StagnationIterations: 10,
		MinIterations:        2,
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
		{"zero sigma", "initial_sigma", func(c *Config) { c.InitialSigma = 0 }},
		{"infinite sigma", "initial_sigma", func(c *Config) { c.InitialSigma = math.Inf(1) }},
		{"mean length", "initial_mean", func(c *Config) { c.InitialMean = []float64{0} }},
		{"non-finite mean", "initial_mean", func(c *Config) { c.InitialMean[1] = math.NaN() }},
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
		TerminationConditionNumber: "condition_number",
		TerminationNoEffectAxis:    "no_effect_axis",
		TerminationNoEffectCoord:   "no_effect_coord",
	}

	for reason, want := range reasons {
		if string(reason) != want {
			t.Errorf("reason = %q, want %q", reason, want)
		}
	}
}
