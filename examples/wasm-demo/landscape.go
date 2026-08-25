package main

import (
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
}

var landscapes = []landscapeSpec{
	{
		key: "rosenbrock", name: "Rosenbrock valley",
		note:  "A narrow curved valley: the ellipse must rotate and contract while the mean follows the bend.",
		lower: -2, upper: 2, initial: [2]float64{-1.3, 1.45}, sigma: 0.42,
		minima: [][2]float64{{1, 1}}, objective: cmaes.Rosenbrock,
	},
	{
		key: "ellipsoid", name: "Rotated ellipsoid · condition 10⁶",
		note:  "The thesis case for CMA-ES: covariance learns a diagonal valley whose two curvatures differ by one million.",
		lower: -3, upper: 3, initial: [2]float64{2.1, -1.5}, sigma: 0.62,
		minima: [][2]float64{{0, 0}}, objective: conditionedEllipsoid,
	},
	{
		key: "rastrigin", name: "Rastrigin",
		note:  "Many regular local basins expose why a single local run can need population restarts.",
		lower: -5.12, upper: 5.12, initial: [2]float64{3.4, -3.1}, sigma: 1.25,
		minima: [][2]float64{{0, 0}}, objective: cmaes.Rastrigin,
	},
	{
		key: "himmelblau", name: "Himmelblau",
		note:  "Four equal minima show how the initial distribution and seed choose a basin.",
		lower: -5, upper: 5, initial: [2]float64{0, 0}, sigma: 1.45,
		minima:    [][2]float64{{3, 2}, {-2.805118, 3.131312}, {-3.77931, -3.283186}, {3.584428, -1.848126}},
		objective: cmaes.Himmelblau,
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
