//go:build js && wasm

package main

import (
	"context"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"strings"
	"syscall/js"

	"github.com/CWBudde/go-cma-es"
)

const (
	defaultIterations = 120
	defaultSeed       = 42
	defaultGrid       = 150
	maxIterations     = 320
	maxLambda         = 96
	maxGrid           = 256
)

type runRequest struct {
	spec        landscapeSpec
	mode        cmaes.CovarianceMode
	seed        int64
	sigma       float64
	lambda      int
	iterations  int
	active      bool
	fixedBudget bool
}

type runHistory struct {
	population    []float32
	mean          []float32
	ellipse       []float32
	bestTrail     []float32
	bestCost      []float32
	iterationBest []float32
	sigma         []float32
	condition     []float32
	lambda        int
}

func jsInfo(js.Value) any {
	items := make([]any, 0, len(landscapes))
	for _, spec := range landscapes {
		items = append(items, map[string]any{
			"key":     spec.key,
			"name":    spec.name,
			"note":    spec.note,
			"lower":   spec.lower,
			"upper":   spec.upper,
			"sigma":   spec.sigma,
			"initial": []any{spec.initial[0], spec.initial[1]},
			"minima":  pointsToJS(spec.minima),
		})
	}

	return map[string]any{
		"version":    cmaes.Version,
		"goVersion":  runtime.Version(),
		"goos":       runtime.GOOS,
		"goarch":     runtime.GOARCH,
		"landscapes": items,
	}
}

func jsLandscape(opts js.Value) any {
	spec, ok := lookupLandscape(readString(opts, "landscape", "rosenbrock"))
	if !ok {
		return errorResult("landscape: unknown landscape")
	}

	width := clampInt(readInt(opts, "width", defaultGrid), 32, maxGrid)
	height := clampInt(readInt(opts, "height", defaultGrid), 32, maxGrid)
	values := make([]float32, 0, width*height)
	position := make([]float64, 2)

	for row := range height {
		position[1] = spec.upper - (float64(row)+0.5)/float64(height)*(spec.upper-spec.lower)

		for column := range width {
			position[0] = spec.lower + (float64(column)+0.5)/float64(width)*(spec.upper-spec.lower)
			values = append(values, float32(spec.objective(position)))
		}
	}

	result := map[string]any{
		"key": spec.key, "name": spec.name, "note": spec.note,
		"width": width, "height": height, "lower": spec.lower, "upper": spec.upper,
		"minima": pointsToJS(spec.minima),
	}
	putFloats(result, opts.Get("out"), "values", rankNormalize(values))

	return result
}

func jsRun(opts js.Value) any {
	request, errResult := parseRunRequest(opts, "rosenbrock")
	if errResult != nil {
		return errResult
	}

	history, result, err := runCMA(request)
	if err != nil {
		return errorResult("run: %v", err)
	}

	response := runResponse(history, result, "")
	response["landscape"] = request.spec.key
	response["mode"] = string(request.mode)
	response["active"] = request.active
	writeRunArrays(response, opts.Get("out"), history, "")

	return response
}

func parseRunRequest(opts js.Value, fallbackLandscape string) (runRequest, map[string]any) {
	spec, ok := lookupLandscape(readString(opts, "landscape", fallbackLandscape))
	if !ok {
		return runRequest{}, errorResult("run: unknown landscape")
	}

	mode := cmaes.CovarianceFull
	if readString(opts, "mode", "full") == "separable" {
		mode = cmaes.CovarianceSeparable
	}

	sigma := readFloat(opts, "sigma", spec.sigma)
	if !(sigma > 0) {
		return runRequest{}, errorResult("run: sigma must be positive")
	}

	return runRequest{
		spec:       spec,
		mode:       mode,
		seed:       int64(readFloat(opts, "seed", defaultSeed)),
		sigma:      sigma,
		lambda:     clampInt(readInt(opts, "lambda", 8), 4, maxLambda),
		iterations: clampInt(readInt(opts, "iterations", defaultIterations), 1, maxIterations),
		active:     readBool(opts, "active", true),
	}, nil
}

func runCMA(request runRequest) (*runHistory, *cmaes.Result, error) {
	config := cmaes.NewDefaultConfig(2)
	config.ObjectiveFunc = request.spec.objective
	config.LowerBound = request.spec.lower
	config.UpperBound = request.spec.upper
	config.InitialMean = []float64{request.spec.initial[0], request.spec.initial[1]}
	config.InitialSigma = request.sigma
	config.Lambda = request.lambda
	config.Mu = request.lambda / 2
	config.MaxIterations = request.iterations
	config.MaxWorkers = 1
	config.EnableParallel = false
	config.CovarianceMode = request.mode
	config.ActiveCMA = request.active
	config.Seed = &request.seed
	if request.fixedBudget {
		config.Convergence = nil
	}

	history := &runHistory{lambda: request.lambda}
	result, err := cmaes.OptimizeContext(
		context.Background(),
		config,
		cmaes.WithPopulationObserver(history.recordPopulation),
		cmaes.WithDistributionObserver(history.recordDistribution),
	)
	if err != nil {
		return nil, nil, err
	}

	history.bestCost = narrow(result.ConvergenceCurve)
	history.iterationBest = narrow(result.IterationBestHistory)
	history.sigma = narrow(result.SigmaHistory)
	history.condition = narrow(result.ConditionNumberHistory)

	return history, result, nil
}

func (history *runHistory) recordPopulation(snapshot cmaes.PopulationSnapshot) {
	for _, candidate := range snapshot.Population {
		history.population = append(history.population,
			float32(candidate.Position[0]), float32(candidate.Position[1]))
	}

	history.bestTrail = append(history.bestTrail,
		float32(snapshot.Best.Position[0]), float32(snapshot.Best.Position[1]))
}

func (history *runHistory) recordDistribution(snapshot cmaes.DistributionSnapshot) {
	history.mean = append(history.mean, float32(snapshot.Mean[0]), float32(snapshot.Mean[1]))

	// A point on the drawn ellipse is m + A*[cos(t),sin(t)], where
	// A = 2*sigma*B*D. The four entries below are A in row-major order.
	scale := 2 * snapshot.Sigma
	history.ellipse = append(history.ellipse,
		float32(scale*snapshot.Eigenvectors[0][0]*snapshot.Eigenvalues[0]),
		float32(scale*snapshot.Eigenvectors[0][1]*snapshot.Eigenvalues[1]),
		float32(scale*snapshot.Eigenvectors[1][0]*snapshot.Eigenvalues[0]),
		float32(scale*snapshot.Eigenvectors[1][1]*snapshot.Eigenvalues[1]),
	)
}

func runResponse(history *runHistory, result *cmaes.Result, prefix string) map[string]any {
	return map[string]any{
		prefixed(prefix, "frames"):      result.IterationCount,
		prefixed(prefix, "lambda"):      history.lambda,
		prefixed(prefix, "evaluations"): result.FuncEvalCount,
		prefixed(prefix, "termination"): string(result.TerminationReason),
		prefixed(prefix, "best"):        finiteNumber(result.GlobalBest.Cost),
		prefixed(prefix, "bestPosition"): []any{
			finiteNumber(result.GlobalBest.Position[0]),
			finiteNumber(result.GlobalBest.Position[1]),
		},
	}
}

func writeRunArrays(response map[string]any, out js.Value, history *runHistory, prefix string) {
	putFloats(response, out, prefixed(prefix, "population"), history.population)
	putFloats(response, out, prefixed(prefix, "mean"), history.mean)
	putFloats(response, out, prefixed(prefix, "ellipse"), history.ellipse)
	putFloats(response, out, prefixed(prefix, "bestTrail"), history.bestTrail)
	putFloats(response, out, prefixed(prefix, "bestCost"), history.bestCost)
	putFloats(response, out, prefixed(prefix, "iterationBest"), history.iterationBest)
	putFloats(response, out, prefixed(prefix, "sigma"), history.sigma)
	putFloats(response, out, prefixed(prefix, "condition"), history.condition)
}

func prefixed(prefix, key string) string {
	if prefix == "" {
		return key
	}

	return prefix + strings.ToUpper(key[:1]) + key[1:]
}

func jsCompare(opts js.Value) any {
	request, errResult := parseRunRequest(opts, "ellipsoid")
	if errResult != nil {
		return errResult
	}

	// This page is deliberately fixed to the hard comparison landscape.
	request.spec, _ = lookupLandscape("ellipsoid")
	request.mode = cmaes.CovarianceFull
	request.active = true
	request.fixedBudget = true

	history, result, err := runCMA(request)
	if err != nil {
		return errorResult("compare: %v", err)
	}

	baseline := runIsotropic(request)
	response := runResponse(history, result, "cma")
	response["landscape"] = request.spec.key
	response["seed"] = float64(request.seed)
	response["budget"] = request.lambda * request.iterations
	response["isoFrames"] = request.iterations
	response["isoLambda"] = request.lambda
	response["isoEvaluations"] = request.lambda * request.iterations
	response["isoTermination"] = "fixed_budget"
	response["isoBest"] = finiteNumber(baseline.best)
	response["isoBestPosition"] = []any{baseline.bestPosition[0], baseline.bestPosition[1]}

	out := opts.Get("out")
	writeRunArrays(response, out, history, "cma")
	putFloats(response, out, "isoPopulation", baseline.population)
	putFloats(response, out, "isoMean", baseline.mean)
	putFloats(response, out, "isoBestTrail", baseline.bestTrail)
	putFloats(response, out, "isoBestCost", baseline.bestCost)

	return response
}

type isotropicHistory struct {
	population   []float32
	mean         []float32
	bestTrail    []float32
	bestCost     []float32
	bestPosition [2]float64
	best         float64
}

type isotropicCandidate struct {
	position [2]float64
	cost     float64
}

func runIsotropic(request runRequest) isotropicHistory {
	rng := rand.New(rand.NewSource(request.seed))
	mean := request.spec.initial
	history := isotropicHistory{best: math.Inf(1)}
	mu := request.lambda / 2
	weights := isotropicWeights(request.lambda, mu)

	for range request.iterations {
		population := make([]isotropicCandidate, request.lambda)
		for index := range population {
			position := [2]float64{
				mean[0] + request.sigma*rng.NormFloat64(),
				mean[1] + request.sigma*rng.NormFloat64(),
			}
			position[0] = min(request.spec.upper, max(request.spec.lower, position[0]))
			position[1] = min(request.spec.upper, max(request.spec.lower, position[1]))
			population[index] = isotropicCandidate{
				position: position,
				cost:     request.spec.objective(position[:]),
			}
		}

		sort.SliceStable(population, func(left, right int) bool {
			return population[left].cost < population[right].cost
		})

		mean = [2]float64{}
		for index := range mu {
			mean[0] += weights[index] * population[index].position[0]
			mean[1] += weights[index] * population[index].position[1]
		}

		if population[0].cost < history.best {
			history.best = population[0].cost
			history.bestPosition = population[0].position
		}

		for _, candidate := range population {
			history.population = append(history.population,
				float32(candidate.position[0]), float32(candidate.position[1]))
		}
		history.mean = append(history.mean, float32(mean[0]), float32(mean[1]))
		history.bestTrail = append(history.bestTrail,
			float32(history.bestPosition[0]), float32(history.bestPosition[1]))
		history.bestCost = append(history.bestCost, float32(history.best))
	}

	return history
}

func isotropicWeights(lambda, mu int) []float64 {
	weights := make([]float64, mu)
	sum := 0.0
	for index := range weights {
		weights[index] = math.Log(float64(lambda)/2+0.5) - math.Log(float64(index+1))
		sum += weights[index]
	}

	for index := range weights {
		weights[index] /= sum
	}

	return weights
}

func jsRestart(opts js.Value) any {
	spec, _ := lookupLandscape("rastrigin")
	baseSeed := int64(readFloat(opts, "seed", defaultSeed))
	restarts := clampInt(readInt(opts, "restarts", 5), 2, 6)
	iterations := clampInt(readInt(opts, "iterations", 55), 10, 100)
	baseLambda := clampInt(readInt(opts, "lambda", 6), 4, 12)
	budget := baseLambda * ((1 << restarts) - 1) * iterations

	config := cmaes.NewDefaultConfig(2)
	config.ObjectiveFunc = spec.objective
	config.LowerBound = spec.lower
	config.UpperBound = spec.upper
	config.InitialMean = []float64{spec.initial[0], spec.initial[1]}
	config.InitialSigma = 1.25
	config.Lambda = baseLambda
	config.Mu = baseLambda / 2
	config.MaxIterations = iterations
	config.MaxEvaluations = budget
	config.MaxWorkers = 1
	config.EnableParallel = false
	config.ActiveCMA = true
	config.Convergence = nil
	config.Seed = &baseSeed

	result, err := cmaes.OptimizeWithRestarts(config, cmaes.RestartIPOP)
	if err != nil {
		return errorResult("restart schedule: %v", err)
	}

	records := make([]any, 0, len(result.Restarts))
	markers := make([]float32, 0, len(result.Restarts)*4)
	for _, record := range result.Restarts {
		position := record.Best.Position
		basinX, basinY := math.Round(position[0]), math.Round(position[1])
		markers = append(markers,
			float32(position[0]), float32(position[1]), float32(basinX), float32(basinY))
		records = append(records, map[string]any{
			"restart": record.Restart + 1, "lambda": record.Lambda, "seed": float64(record.Seed),
			"evaluations": record.Evaluations, "iterations": record.Iterations,
			"termination": string(record.TerminationReason), "best": finiteNumber(record.Best.Cost),
			"start":    []any{record.InitialMean[0], record.InitialMean[1]},
			"position": []any{position[0], position[1]},
			"basin":    []any{basinX, basinY},
		})
	}

	response := map[string]any{
		"records": records, "restarts": len(result.Restarts), "baseLambda": baseLambda,
		"totalEvaluations": result.FuncEvalCount, "best": finiteNumber(result.GlobalBest.Cost),
		"bestPosition": []any{result.GlobalBest.Position[0], result.GlobalBest.Position[1]},
	}
	putFloats(response, opts.Get("out"), "markers", markers)

	return response
}

func pointsToJS(points [][2]float64) []any {
	items := make([]any, len(points))
	for index, point := range points {
		items[index] = []any{point[0], point[1]}
	}

	return items
}
