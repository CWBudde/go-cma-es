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
