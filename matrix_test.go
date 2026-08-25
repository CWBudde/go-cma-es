package cmaes

import (
	"math"
	"testing"
)

func TestSymmetricRankOneUpdate(t *testing.T) {
	matrix := [][]float64{{1, 2}, {2, 4}}
	symmetricRankOneUpdate(matrix, 0.5, []float64{2, -1})

	want := [][]float64{{3, 1}, {1, 4.5}}
	for row := range want {
		for column := range want {
			if matrix[row][column] != want[row][column] {
				t.Fatalf("matrix[%d][%d] = %g, want %g", row, column, matrix[row][column], want[row][column])
			}
		}
	}
}

// TestSymmetricRankOneUpdateNegativeScale covers the active-CMA case: a
// negative rank-one weight, which is what drives an intermediate covariance
// indefinite. Starting from the identity, subtracting v*v^T for the unit
// vector v = (1,0) leaves diag(0,1); subtracting a further 1.5*v*v^T for
// v = (0,1) leaves diag(0,-0.5), a matrix with a genuinely negative
// eigenvalue. All entries are exact in binary floating point.
func TestSymmetricRankOneUpdateNegativeScale(t *testing.T) {
	matrix := identityMatrix(2)
	symmetricRankOneUpdate(matrix, -1, []float64{1, 0})
	symmetricRankOneUpdate(matrix, -1.5, []float64{0, 1})

	want := [][]float64{{0, 0}, {0, -0.5}}
	for row := range want {
		for column := range want {
			if matrix[row][column] != want[row][column] {
				t.Fatalf("matrix[%d][%d] = %g, want %g", row, column, matrix[row][column], want[row][column])
			}
		}
	}
}

// TestSymmetricRankOneUpdateStaysSymmetric pins the invariant the
// implementation exists for: after any sequence of updates, of either sign and
// with vectors that are not axis aligned, the two triangles are bit-equal, not
// merely close. Recombination relies on that, because the eigendecomposition
// symmetrizes its input and would otherwise silently absorb the drift.
func TestSymmetricRankOneUpdateStaysSymmetric(t *testing.T) {
	const size = 5

	matrix := identityMatrix(size)
	vectors := [][]float64{
		{0.1, -0.7, 3.3, 1e-8, 42},
		{-2.5, 0.03, -0.9, 7, 1e6},
		{1e-3, 1e3, -1e-3, -1e3, 0.5},
	}
	scales := []float64{1.7, -0.4, -3.25}

	for index, vector := range vectors {
		symmetricRankOneUpdate(matrix, scales[index], vector)

		for row := range size {
			for column := range size {
				if matrix[row][column] != matrix[column][row] {
					t.Fatalf(
						"after update %d: matrix[%d][%d] = %.17g, matrix[%d][%d] = %.17g",
						index, row, column, matrix[row][column],
						column, row, matrix[column][row],
					)
				}
			}
		}
	}
}

func TestMatrixVectorProduct(t *testing.T) {
	matrix := [][]float64{{1, 2, 3}, {4, 5, 6}}
	got := matrixVectorProduct(matrix, []float64{2, -1, 0.5})
	want := []float64{1.5, 6}

	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("product[%d] = %g, want %g", index, got[index], want[index])
		}
	}
}

func TestMatrixVectorProductRejectsShortVector(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected a panic for a vector shorter than a matrix row")
		}
	}()

	matrixVectorProduct([][]float64{{1, 2, 3}}, []float64{1, 2})
}

func TestConditionNumber(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		want   float64
	}{
		{name: "empty", values: nil, want: 0},
		{name: "identity", values: []float64{1, 1, 1}, want: 1},
		{name: "unordered", values: []float64{4, 0.5, 2}, want: 8},
		{name: "singular", values: []float64{0, 1}, want: math.Inf(1)},
		{name: "negative", values: []float64{-1, 2}, want: math.Inf(1)},
		{name: "not a number", values: []float64{math.NaN()}, want: math.Inf(1)},
		{name: "infinite", values: []float64{math.Inf(1)}, want: math.Inf(1)},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := conditionNumber(test.values)
			if got != test.want {
				t.Fatalf("conditionNumber(%v) = %g, want %g", test.values, got, test.want)
			}
		})
	}
}

func TestMatrixConstructors(t *testing.T) {
	matrix := identityMatrix(3)
	matrix[0][1] = 7

	if matrix[0][0] != 1 || matrix[1][1] != 1 || matrix[2][2] != 1 {
		t.Fatalf("diagonal is not identity: %v", matrix)
	}

	if matrix[1][0] != 0 {
		t.Fatal("matrix rows unexpectedly alias")
	}
}
