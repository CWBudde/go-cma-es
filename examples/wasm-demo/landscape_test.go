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

			for _, minimum := range spec.minima {
				position := []float64{minimum[0], minimum[1]}
				if cost := spec.objective(position); math.Abs(cost) > 1e-10 {
					t.Fatalf("objective(%v) = %g, want zero", position, cost)
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
