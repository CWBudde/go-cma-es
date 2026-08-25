package cmaes

import (
	"errors"
	"fmt"
	"math"
	"runtime"
)

const (
	defaultInitialSigma       = 0.3
	defaultMaxIterations      = 1000
	defaultTolX               = 1e-11
	defaultTolFun             = 1e-12
	defaultTolXUp             = 1e4
	defaultConditionCov       = 1e14
	highDimensionalIterations = 3000
	fastConvergenceIterations = 300
)

// strategyParameters are Hansen's dimension- and population-dependent
// parameters. They are deliberately derived in one place so later algorithm
// phases cannot silently use a different formula.
type strategyParameters struct {
	weights         []float64
	negativeWeights []float64
	muEff           float64
	cSigma          float64
	dSigma          float64
	cc              float64
	c1              float64
	cmu             float64
}

// NewDefaultConfig creates the default full-covariance CMA-ES configuration.
// The initial mean is the origin and the initial step size is 0.3. Callers must
// set ObjectiveFunc, LowerBound, and UpperBound before validating or optimizing.
func NewDefaultConfig(problemSize int) *Config {
	lambda := defaultPopulationSize(problemSize)
	mu := lambda / 2

	var initialMean []float64
	if problemSize > 0 {
		initialMean = make([]float64, problemSize)
	}

	return &Config{
		Convergence:    NewDefaultConvergenceConfig(),
		InitialMean:    initialMean,
		BoundaryMethod: BoundaryPenalty,
		CovarianceMode: CovarianceFull,
		InitialSigma:   defaultInitialSigma,
		ProblemSize:    problemSize,
		Lambda:         lambda,
		Mu:             mu,
		MaxIterations:  defaultMaxIterations,
		MaxWorkers:     runtime.NumCPU(),
		ActiveCMA:      true,
	}
}

// NewDefaultConvergenceConfig returns Hansen's distribution-derived stopping
// criteria. Target-cost and stagnation termination remain opt-in.
func NewDefaultConvergenceConfig() *ConvergenceConfig {
	return &ConvergenceConfig{
		TolX:          defaultTolX,
		TolFun:        defaultTolFun,
		TolXUp:        defaultTolXUp,
		ConditionCov:  defaultConditionCov,
		NoEffectAxis:  true,
		NoEffectCoord: true,
	}
}

// NewSeparableConfig creates a default configuration using diagonal covariance.
func NewSeparableConfig(problemSize int) *Config {
	config := NewDefaultConfig(problemSize)
	config.CovarianceMode = CovarianceSeparable

	return config
}

// NewHighDimensionalConfig creates a separable configuration with a longer run
// cap for problems where a dense covariance matrix is impractical.
func NewHighDimensionalConfig(problemSize int) *Config {
	config := NewSeparableConfig(problemSize)
	config.MaxIterations = highDimensionalIterations

	return config
}

// NewFastConvergenceConfig creates a full-covariance configuration with a
// shorter run cap for cheap or well-behaved objectives.
func NewFastConvergenceConfig(problemSize int) *Config {
	config := NewDefaultConfig(problemSize)
	config.MaxIterations = fastConvergenceIterations

	return config
}

func defaultPopulationSize(problemSize int) int {
	if problemSize <= 0 {
		return 0
	}

	return 4 + int(math.Floor(3*math.Log(float64(problemSize))))
}

func deriveStrategyParameters(config *Config) strategyParameters {
	weights := make([]float64, config.Mu)
	weightSum := 0.0
	weightBase := math.Log(math.Max(float64(config.Mu), float64(config.Lambda)/2) + 0.5)

	for i := range weights {
		weights[i] = weightBase - math.Log(float64(i+1))
		weightSum += weights[i]
	}

	weightSquareSum := 0.0

	for i := range weights {
		weights[i] /= weightSum
		weightSquareSum += weights[i] * weights[i]
	}

	n := float64(config.ProblemSize)
	muEff := 1 / weightSquareSum
	cSigma := (muEff + 2) / (n + muEff + 5)
	dSigma := 1 + cSigma + 2*math.Max(0, math.Sqrt((muEff-1)/(n+1))-1)
	cc := (4 + muEff/n) / (n + 4 + 2*muEff/n)
	c1 := 2 / ((n+1.3)*(n+1.3) + muEff)

	cmu := math.Min(1-c1, 2*(muEff-2+1/muEff)/((n+2)*(n+2)+muEff))
	if config.CovarianceMode == CovarianceSeparable {
		cmu = math.Min(1-c1, cmu*(n+2)/3)
	}

	var negativeWeights []float64
	if config.ActiveCMA {
		negativeWeights = deriveNegativeWeights(config, weights, muEff, c1, cmu, weightBase)
	}

	return strategyParameters{
		weights:         weights,
		negativeWeights: negativeWeights,
		muEff:           muEff,
		cSigma:          cSigma,
		dSigma:          dSigma,
		cc:              cc,
		c1:              c1,
		cmu:             cmu,
	}
}

// Validate checks whether config is complete and internally consistent.
func (config *Config) Validate() error {
	if config == nil {
		return errors.New("config must not be nil")
	}

	if config.ObjectiveFunc == nil {
		return errors.New("ObjectiveFunc must not be nil")
	}

	if config.ProblemSize <= 0 {
		return fmt.Errorf("problem_size must be positive (got %d)", config.ProblemSize)
	}

	err := validateBounds(config)
	if err != nil {
		return err
	}

	err = validateInitialDistribution(config)
	if err != nil {
		return err
	}

	if config.Lambda < 2 {
		return fmt.Errorf("lambda must be at least 2 (got %d)", config.Lambda)
	}

	if config.Mu < 1 || config.Mu > config.Lambda {
		return fmt.Errorf("mu must be in [1, lambda] (got mu=%d, lambda=%d)", config.Mu, config.Lambda)
	}

	if config.MaxIterations <= 0 {
		return fmt.Errorf("max_iterations must be positive (got %d)", config.MaxIterations)
	}

	if config.MaxEvaluations < 0 {
		return fmt.Errorf("max_evaluations must be non-negative (got %d)", config.MaxEvaluations)
	}

	if config.MaxWorkers < 0 {
		return fmt.Errorf("max_workers must be non-negative (got %d)", config.MaxWorkers)
	}

	if config.Seed != nil && config.Rand != nil {
		return errors.New("seed and Rand are mutually exclusive")
	}

	err = validateModes(config)
	if err != nil {
		return err
	}

	err = validateConvergenceConfig(config.Convergence, config.MaxIterations)
	if err != nil {
		return fmt.Errorf("invalid convergence config: %w", err)
	}

	err = validateConstraintConfig(config.Constraints)
	if err != nil {
		return fmt.Errorf("invalid constraint config: %w", err)
	}

	return nil
}

func validateBounds(config *Config) error {
	if !isFinite(config.LowerBound) || !isFinite(config.UpperBound) {
		return fmt.Errorf("lower_bound and upper_bound must be finite (got %v, %v)",
			config.LowerBound, config.UpperBound)
	}

	if config.LowerBound >= config.UpperBound {
		return fmt.Errorf("lower_bound (%v) must be less than upper_bound (%v)",
			config.LowerBound, config.UpperBound)
	}

	if !isFinite(config.UpperBound - config.LowerBound) {
		return fmt.Errorf("upper_bound - lower_bound must be finite (got %v - %v)",
			config.UpperBound, config.LowerBound)
	}

	return nil
}

func validateInitialDistribution(config *Config) error {
	if !isFinite(config.InitialSigma) || config.InitialSigma <= 0 {
		return fmt.Errorf("initial_sigma must be finite and positive (got %v)", config.InitialSigma)
	}

	if len(config.InitialMean) != config.ProblemSize {
		return fmt.Errorf("initial_mean has length %d, want problem_size %d",
			len(config.InitialMean), config.ProblemSize)
	}

	for index, value := range config.InitialMean {
		if !isFinite(value) {
			return fmt.Errorf("initial_mean[%d] must be finite (got %v)", index, value)
		}
	}

	return nil
}

func validateModes(config *Config) error {
	switch config.BoundaryMethod {
	case BoundaryPenalty, BoundaryClamp, BoundaryReflect:
	default:
		return fmt.Errorf("unknown boundary_method %q", config.BoundaryMethod)
	}

	switch config.CovarianceMode {
	case CovarianceFull, CovarianceSeparable:
	default:
		return fmt.Errorf("unknown covariance_mode %q", config.CovarianceMode)
	}

	return nil
}

func validateConvergenceConfig(config *ConvergenceConfig, maxIterations int) error {
	if config == nil {
		return nil
	}

	if config.TargetCost != nil && !isFinite(*config.TargetCost) {
		return fmt.Errorf("target_cost must be finite (got %v)", *config.TargetCost)
	}

	tolerances := []struct {
		name  string
		value float64
	}{
		{"tol_x", config.TolX},
		{"tol_fun", config.TolFun},
		{"tol_x_up", config.TolXUp},
		{"condition_cov", config.ConditionCov},
	}

	for _, tolerance := range tolerances {
		if !isFinite(tolerance.value) || tolerance.value < 0 {
			return fmt.Errorf("%s must be finite and non-negative (got %v)",
				tolerance.name, tolerance.value)
		}
	}

	if !isFinite(config.MinImprovement) || config.MinImprovement < 0 {
		return fmt.Errorf("min_improvement must be finite and non-negative (got %v)",
			config.MinImprovement)
	}

	if config.StagnationIterations < 0 {
		return fmt.Errorf("stagnation_iterations must be non-negative (got %d)",
			config.StagnationIterations)
	}

	if config.MinIterations < 0 || config.MinIterations > maxIterations {
		return fmt.Errorf("min_iterations must be in [0, max_iterations] (got %d)",
			config.MinIterations)
	}

	return nil
}

func validateConstraintConfig(config *ConstraintConfig) error {
	if config == nil {
		return nil
	}

	switch config.Handling {
	case "", ConstraintHandlingFeasibility, ConstraintHandlingPenalty:
	default:
		return fmt.Errorf("unknown handling method %q", config.Handling)
	}

	switch config.PenaltyMethod {
	case "", PenaltyLinear, PenaltyQuadratic:
	default:
		return fmt.Errorf("unknown penalty_method %q", config.PenaltyMethod)
	}

	if !isFinite(config.PenaltyFactor) || config.PenaltyFactor < 0 {
		return fmt.Errorf("penalty_factor must be finite and non-negative (got %v)",
			config.PenaltyFactor)
	}

	if !isFinite(config.EqualityTolerance) || config.EqualityTolerance < 0 {
		return fmt.Errorf("equality_tolerance must be finite and non-negative (got %v)",
			config.EqualityTolerance)
	}

	for index, constraint := range config.Inequalities {
		if constraint == nil {
			return fmt.Errorf("inequality constraint %d is nil", index)
		}
	}

	for index, constraint := range config.Equalities {
		if constraint == nil {
			return fmt.Errorf("equality constraint %d is nil", index)
		}
	}

	if config.Handling == ConstraintHandlingPenalty && config.PenaltyFactor == 0 {
		return errors.New("penalty_factor must be positive with penalty handling")
	}

	return nil
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
