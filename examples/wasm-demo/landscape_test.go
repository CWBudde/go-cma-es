package main

import (
	"math"
	"testing"
)

func TestLandscapeMinima(t *testing.T) {
	t.Parallel()

	for _, spec := range landscapes {
		spec := spec
		t.Run(spec.key, func(t *testing.T) {
			t.Parallel()

			if len(spec.minima) == 0 {
				t.Fatal("landscape has no documented minimum")
			}

			if spec.tolerance <= 0 {
				t.Fatal("landscape declares no tolerance for its optimum")
			}

			for _, minimum := range spec.minima {
				position := []float64{minimum[0], minimum[1]}
				if cost := spec.objective(position); math.Abs(cost-spec.optimum) > spec.tolerance {
					t.Fatalf("objective(%v) = %g, want %g within %g",
						position, cost, spec.optimum, spec.tolerance)
				}
			}
		})
	}
}

func TestConditionedEllipsoidHasPublishedCondition(t *testing.T) {
	t.Parallel()

	const inverseSqrtTwo = 0.70710678118654752440
	along := conditionedEllipsoid([]float64{inverseSqrtTwo, inverseSqrtTwo})
	across := conditionedEllipsoid([]float64{inverseSqrtTwo, -inverseSqrtTwo})

	if ratio := across / along; math.Abs(ratio-1e6) > 1e-6 {
		t.Fatalf("curvature ratio = %g, want 1e6", ratio)
	}
}

func TestRankNormalize(t *testing.T) {
	t.Parallel()

	got := rankNormalize([]float32{30, 10, 20})
	want := []float32{1, 0, 0.5}

	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("normalized[%d] = %g, want %g", index, got[index], want[index])
		}
	}
}

func TestLatticeBasin(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		position []float64
		want     [2]float64
	}{
		{"already integral", []float64{3, -2}, [2]float64{3, -2}},
		{"rounds down below the half step", []float64{2.49, -2.49}, [2]float64{2, -2}},
		{"rounds away from zero at the half step", []float64{2.5, -2.5}, [2]float64{3, -3}},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			if got := latticeBasin(testCase.position); got != testCase.want {
				t.Fatalf("latticeBasin(%v) = %v, want %v", testCase.position, got, testCase.want)
			}
		})
	}
}

func TestRingBasin(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		position   []float64
		wantRadius float64
	}{
		{"origin collapses to the center", []float64{0, 0}, 0},
		{"inside the first ridge collapses to the center", []float64{1, 1}, 0},
		{"just outside the first ridge snaps to ring one", []float64{2.3, 2.3}, math.Pi},
		{"a point on ring two stays there", []float64{0, 2 * math.Pi}, 2 * math.Pi},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			got := ringBasin(testCase.position)
			if radius := math.Hypot(got[0], got[1]); math.Abs(radius-testCase.wantRadius) > 1e-12 {
				t.Fatalf("|ringBasin(%v)| = %g, want %g", testCase.position, radius, testCase.wantRadius)
			}
		})
	}
}

// TestSchafferRidgeDwarfsItsReward pins the reason the Schaffer landscape needs
// restarts: the barrier between the first ripple ring and the global optimum is
// roughly a hundred times larger than the improvement it guards (the measured
// ratio is 99.6, so the assertion leaves a margin below it).
func TestSchafferRidgeDwarfsItsReward(t *testing.T) {
	t.Parallel()

	spec, ok := lookupLandscape("schaffer")
	if !ok {
		t.Fatal("the schaffer landscape is missing")
	}

	center := spec.objective([]float64{0, 0})
	ring := spec.objective([]float64{math.Pi, 0})
	ridge := spec.objective([]float64{1.5 * math.Pi, 0})

	reward := ring - center
	if barrier := ridge - ring; barrier < 90*reward {
		t.Fatalf("ridge barrier %g is not ~100x the reward %g", barrier, reward)
	}
}
