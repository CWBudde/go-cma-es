package cmaes

import (
	"errors"
	"fmt"
	"math"
	"runtime"
)

const (
	defaultInitialSigma  = 0.3
	defaultMaxIterations = 1000
	// defaultTolX is Hansen's TolX, which is relative to the initial step size:
	// the run stops below 1e-12 * sigma^(0).
	defaultTolX               = 1e-12
	defaultTolFun             = 1e-12
	defaultTolXUp             = 1e4
	defaultConditionCov       = 1e14
	highDimensionalIterations = 3000
	fastConvergenceIterations = 300
)

// errConstraintFunctionsMissing reports a ConstraintConfig that is configured
// but carries no callbacks, which is what a JSON round trip leaves behind. It
// is a sentinel because LoadConfig tolerates exactly this one condition: the
// callbacks cannot be serialized, so a loaded configuration is invalid to run
// until the caller reattaches them, and Validate says so.
var errConstraintFunctionsMissing = errors.New(
	"constraint handling is configured but no constraint functions are set; " +
		"they are not restored by LoadConfig and must be reattached")

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
// set ObjectiveFunc and the search box - either the LowerBound and UpperBound
// scalars or the per-dimension LowerBounds and UpperBounds slices - before
// validating or optimizing.
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

// NewBlockDiagonalConfig creates a default configuration whose covariance is
// split into consecutive blocks of blockSize coordinates. The final block may
// be shorter. Set BlockGroups to an explicit partition for non-contiguous
// parameter groups.
func NewBlockDiagonalConfig(problemSize, blockSize int) *Config {
	config := NewDefaultConfig(problemSize)
	config.CovarianceMode = CovarianceBlock
	config.BlockSize = blockSize

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

// maxRankMuShare bounds the rank-mu learning rate away from the convexity
// boundary 1-c1, as a share of it.
//
// The covariance update is a convex combination whose decay factor is
// 1 - c1 - cmu*sum(w). Clamping cmu to exactly 1-c1 drives that decay to
// exactly zero: C is then rebuilt from the current generation alone, retaining
// nothing from the ones before it. That is not an aggressive CMA-ES, it is a
// memoryless one, and learning a metric across generations is the property this
// library exists to provide.
//
// Hansen's positive-definiteness guard on the active weights,
// alpha_posdef = (1 - c1 - cmu)/(n*cmu), is the same expression, so at that
// boundary it is exactly zero too: every negative weight is scaled to nothing
// and ActiveCMA becomes a silent no-op. Reserving a share of the budget fixes
// both, because both are the same slack.
//
// The bound binds only where cmu would otherwise approach the boundary, which
// needs a muEff far above Hansen's default -- either a very large Lambda, or
// the separable and block corrections multiplying an already large dense rate
// by (n+2)/(blockDimension+2). Configurations in the published regime are
// unchanged; TestRankMuBoundLeavesPublishedRegimeUnchanged pins that.
//
// The specific share is a judgment call rather than a published constant: it is
// the smallest reserve that keeps a comfortably conditioned active budget in
// the regimes this library has been measured in. Ros and Hansen leave the
// saturated case unspecified because their factor was not intended to reach it.
const maxRankMuShare = 0.9

func deriveStrategyParameters(config *Config) strategyParameters {
	weights := make([]float64, config.Mu)
	weightSum := 0.0

	// weightBase is the offset of the log-decreasing recombination weights
	// w'_i = weightBase - ln(i). It is a hybrid of two published forms: Hansen's
	// tutorial (arXiv:1604.00772, Table 1) uses ln((lambda + 1)/2), while
	// purecmaes.m uses ln(mu + 1/2). They agree at the default mu = lambda/2.
	// Taking the larger of mu and lambda/2 keeps the tutorial form for that
	// default and switches to the purecmaes form once a caller raises Mu above
	// lambda/2, where the tutorial form would drive the trailing raw weights
	// negative and leave recombination with something other than a probability
	// vector.
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

	cmu := 2 * (muEff - 2 + 1/muEff) / ((n+2)*(n+2) + muEff)
	blockDimension := covarianceBlockDimension(config)

	if blockDimension < config.ProblemSize {
		// Ros and Hansen's separable correction is the block-size-one end
		// point of this factor. A bounded block has only blockDimension
		// covariance directions to learn, so it need not retain the dense
		// n-dimensional rank-mu rate.
		cmu *= (n + 2) / float64(blockDimension+2)
	}

	cmu = math.Min(maxRankMuShare*(1-c1), cmu)

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

	// Zero means "no cap"; any positive budget must cover one whole generation,
	// because a run that cannot finish its first generation returns nothing.
	if config.MaxEvaluations > 0 && config.MaxEvaluations < config.Lambda {
		return fmt.Errorf("max_evaluations (%d) must be zero or at least lambda (%d)",
			config.MaxEvaluations, config.Lambda)
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

// coordinateBounds returns the search interval for one coordinate,
// broadcasting the scalar bounds when no per-dimension slice is set.
//
// The two sides are resolved independently: LowerBounds[coordinate] wins over
// LowerBound whenever LowerBounds covers that coordinate, and likewise for the
// upper side. An index outside a configured slice falls back to the scalar, so
// the helper is safe to call before Validate has checked the slice lengths.
//
//nolint:nonamedreturns // Two same-typed results: the names say which is which.
func coordinateBounds(config *Config, coordinate int) (lower, upper float64) {
	lower = config.LowerBound
	if coordinate >= 0 && coordinate < len(config.LowerBounds) {
		lower = config.LowerBounds[coordinate]
	}

	upper = config.UpperBound
	if coordinate >= 0 && coordinate < len(config.UpperBounds) {
		upper = config.UpperBounds[coordinate]
	}

	return lower, upper
}

// coordinateBoxWidth returns the width of one coordinate's search interval,
// upper - lower, using the same broadcasting rule as coordinateBounds. Validate
// guarantees the width is positive and finite for a valid configuration.
func coordinateBoxWidth(config *Config, coordinate int) float64 {
	lower, upper := coordinateBounds(config, coordinate)

	return upper - lower
}

func validateBounds(config *Config) error {
	err := validateBoundSlice("lower_bounds", config.LowerBounds, config.ProblemSize)
	if err != nil {
		return err
	}

	err = validateBoundSlice("upper_bounds", config.UpperBounds, config.ProblemSize)
	if err != nil {
		return err
	}

	if (config.LowerBounds == nil && !isFinite(config.LowerBound)) ||
		(config.UpperBounds == nil && !isFinite(config.UpperBound)) {
		return fmt.Errorf("lower_bound and upper_bound must be finite (got %v, %v)",
			config.LowerBound, config.UpperBound)
	}

	for coordinate := range config.ProblemSize {
		lower, upper := coordinateBounds(config, coordinate)

		if lower >= upper {
			return fmt.Errorf("lower_bound[%d] (%v) must be less than upper_bound[%d] (%v)",
				coordinate, lower, coordinate, upper)
		}

		if !isFinite(upper - lower) {
			return fmt.Errorf("upper_bound - lower_bound must be finite for coordinate %d (got %v - %v)",
				coordinate, upper, lower)
		}
	}

	return nil
}

// validateBoundSlice checks an optional per-dimension bound vector. A nil slice
// is the documented "broadcast the scalar" case and is always accepted.
func validateBoundSlice(name string, bounds []float64, problemSize int) error {
	if bounds == nil {
		return nil
	}

	if len(bounds) != problemSize {
		return fmt.Errorf("%s has length %d, want problem_size %d", name, len(bounds), problemSize)
	}

	for index, value := range bounds {
		if !isFinite(value) {
			return fmt.Errorf("%s[%d] must be finite (got %v)", name, index, value)
		}
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

		// The mean is copied into the distribution state unprojected, so an
		// out-of-box mean would start the run outside the feasible region.
		lower, upper := coordinateBounds(config, index)
		if value < lower || value > upper {
			return fmt.Errorf("initial_mean[%d] (%v) must lie within [%v, %v]",
				index, value, lower, upper)
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
		return nil
	case CovarianceBlock:
		return validateBlockConfiguration(config)
	default:
		return fmt.Errorf("unknown covariance_mode %q", config.CovarianceMode)
	}
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

	// The constraint callbacks are not serializable, so a configuration that
	// survives a save/load cycle keeps its handling settings but loses every
	// function. Rejecting a configured-but-empty ConstraintConfig turns that
	// silent loss into the same loud failure a missing ObjectiveFunc produces.
	// A wholly zero ConstraintConfig carries no intent and stays legal.
	if len(config.Inequalities) == 0 && len(config.Equalities) == 0 && !isZeroConstraintConfig(config) {
		return errConstraintFunctionsMissing
	}

	return nil
}

// isZeroConstraintConfig reports whether config carries no configured intent,
// which is the one case where an empty constraint set is meaningful.
func isZeroConstraintConfig(config *ConstraintConfig) bool {
	return config.Handling == "" &&
		config.PenaltyMethod == "" &&
		config.PenaltyFactor == 0 &&
		config.EqualityTolerance == 0
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
