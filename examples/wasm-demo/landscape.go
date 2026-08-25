package main

import (
	"math"
	"sort"

	"github.com/CWBudde/go-cma-es"
)

type landscapeSpec struct {
	objective cmaes.ObjectiveFunction
	key       string
	name      string
	note      string
	initial   [2]float64
	minima    [][2]float64
	lower     float64
	upper     float64
	sigma     float64
	// optimum is the objective value at every entry of minima, and tolerance
	// is how far a declared minimum may sit from it. Neither is zero for
	// Michalewicz (whose optimum is negative) or Schwefel (whose published
	// minimizer and constant are both rounded).
	optimum   float64
	tolerance float64
}

var landscapes = []landscapeSpec{
	{
		key: "rosenbrock", name: "Rosenbrock valley",
		note:  "A narrow curved valley: the ellipse must rotate and contract while the mean follows the bend.",
		lower: -2, upper: 2, initial: [2]float64{-1.3, 1.45}, sigma: 0.42,
		minima: [][2]float64{{1, 1}}, tolerance: 1e-10, objective: cmaes.Rosenbrock,
	},
	{
		key: "ellipsoid", name: "Rotated ellipsoid · condition 10⁶",
		note:  "The thesis case for CMA-ES: covariance learns a diagonal valley whose two curvatures differ by one million.",
		lower: -3, upper: 3, initial: [2]float64{2.1, -1.5}, sigma: 0.62,
		minima: [][2]float64{{0, 0}}, tolerance: 1e-10, objective: conditionedEllipsoid,
	},
	{
		key: "rastrigin", name: "Rastrigin",
		note:  "Many regular local basins expose why a single local run can need population restarts.",
		lower: -5.12, upper: 5.12, initial: [2]float64{3.4, -3.1}, sigma: 1.25,
		minima: [][2]float64{{0, 0}}, tolerance: 1e-10, objective: cmaes.Rastrigin,
	},
	{
		key: "himmelblau", name: "Himmelblau",
		note:  "Four equal minima show how the initial distribution and seed choose a basin.",
		lower: -5, upper: 5, initial: [2]float64{0, 0}, sigma: 1.45,
		minima:    [][2]float64{{3, 2}, {-2.805118, 3.131312}, {-3.77931, -3.283186}, {3.584428, -1.848126}},
		tolerance: 1e-10,
		objective: cmaes.Himmelblau,
	},
	{
		key: "sphere", name: "Sphere",
		note:  "The control case: there is no curvature to learn, so the ellipse stays circular and only step-size adaptation shrinks it.",
		lower: -5, upper: 5, initial: [2]float64{-3.5, 3.2}, sigma: 1.2,
		minima: [][2]float64{{0, 0}}, tolerance: 1e-10, objective: cmaes.Sphere,
	},
	{
		key: "ackley", name: "Ackley",
		note:  "Drawn on ±15 instead of the usual ±32.768 so the central funnel is more than one pixel wide; the ellipse rides over the bumpy rim before collapsing down the funnel.",
		lower: -15, upper: 15, initial: [2]float64{-9.5, 8}, sigma: 3.5,
		minima: [][2]float64{{0, 0}}, tolerance: 1e-10, objective: cmaes.Ackley,
	},
	{
		key: "schwefel", name: "Schwefel",
		note:  "Deceptive: the local slope points away from the corner optimum, so only a σ wide enough to sample a distant basin lets the mean cross to it.",
		lower: -500, upper: 500, initial: [2]float64{-120, 60}, sigma: 150,
		minima: [][2]float64{{420.9687, 420.9687}}, optimum: 2.5455675e-05, tolerance: 1e-9,
		objective: cmaes.Schwefel,
	},
	{
		key: "michalewicz", name: "Michalewicz",
		note:  "Flat plateaus give the ranking almost nothing to order until a sample drops into one of the steep valleys, after which the ellipse stretches along it.",
		lower: 0, upper: math.Pi, initial: [2]float64{2.6, 0.6}, sigma: 0.5,
		minima: [][2]float64{{2.202906, 1.570796}}, optimum: -1.8013034, tolerance: 1e-6,
		objective: cmaes.Michalewicz,
	},
	{
		key: "zakharov", name: "Zakharov",
		note:  "Unimodal but strongly coupled: a clean view of the ellipse rotating onto the shared axis before it contracts along it.",
		lower: -5, upper: 10, initial: [2]float64{-3.5, 7.5}, sigma: 1.5,
		minima: [][2]float64{{0, 0}}, tolerance: 1e-10, objective: cmaes.Zakharov,
	},
	{
		key: "schaffer", name: "Expanded Schaffer F6",
		note:  "Concentric rings, drawn on ±10 so the grid resolves them: the mean tunnels inward ring by ring and can stall on one ridge short of the center.",
		lower: -10, upper: 10, initial: [2]float64{4, -4}, sigma: 2,
		minima: [][2]float64{{0, 0}}, tolerance: 1e-10, objective: cmaes.ExpandedSchafferF6,
	},
}

func lookupLandscape(key string) (landscapeSpec, bool) {
	for _, spec := range landscapes {
		if spec.key == key {
			return spec, true
		}
	}

	return landscapeSpec{}, false
}

func conditionedEllipsoid(position []float64) float64 {
	const inverseSqrtTwo = 0.70710678118654752440

	along := (position[0] + position[1]) * inverseSqrtTwo
	across := (position[0] - position[1]) * inverseSqrtTwo

	return along*along + 1e6*across*across
}

func rankNormalize(values []float32) []float32 {
	order := make([]int, len(values))
	for index := range order {
		order[index] = index
	}

	sort.SliceStable(order, func(left, right int) bool {
		return values[order[left]] < values[order[right]]
	})

	normalized := make([]float32, len(values))
	denominator := float32(max(1, len(values)-1))
	for rank, index := range order {
		normalized[index] = float32(rank) / denominator
	}

	return normalized
}
