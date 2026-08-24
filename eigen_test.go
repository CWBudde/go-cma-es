package cmaes

import (
	"fmt"
	"math"
	"testing"
)

func TestSymmetricEigendecompositionKnownAnswers(t *testing.T) {
	tests := []struct {
		name   string
		matrix [][]float64
		want   []float64
	}{
		{
			name: "diagonal",
			matrix: [][]float64{
				{4, 0, 0},
				{0, 1, 0},
				{0, 0, 2},
			},
			want: []float64{1, 2, 4},
		},
		{
			name: "two by two rotation",
			matrix: [][]float64{
				{1.75, -1.299038105676658},
				{-1.299038105676658, 3.25},
			},
			want: []float64{1, 4},
		},
		{
			name:   "Hilbert six",
			matrix: hilbertMatrix(6),
			want: []float64{
				1.0827994845655498e-7,
				1.2570757122625195e-5,
				6.157483541826577e-4,
				1.6321521319875822e-2,
				2.4236087057520955e-1,
				1.6188998589243391,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			values, vectors := symmetricEigendecomposition(test.matrix)
			for index, want := range test.want {
				assertClose(t, values[index], want, 1e-12)
			}

			assertReconstruction(t, test.matrix, values, vectors)
			assertOrthonormal(t, vectors)
		})
	}
}

func TestSymmetricEigendecompositionIllConditioned(t *testing.T) {
	const small = 1e-16

	matrix := [][]float64{
		{0.5 + small/2, 0.5 - small/2, 0},
		{0.5 - small/2, 0.5 + small/2, 0},
		{0, 0, 1e8},
	}

	values, vectors := symmetricEigendecomposition(matrix)

	if conditionNumber(values) < 1e20 {
		t.Fatalf("condition number = %g, want at least 1e20", conditionNumber(values))
	}

	assertReconstruction(t, matrix, values, vectors)
	assertOrthonormal(t, vectors)
}

func TestSymmetricEigendecompositionOneDimension(t *testing.T) {
	values, vectors := symmetricEigendecomposition([][]float64{{3.5}})

	assertClose(t, values[0], 3.5, 0)
	assertClose(t, vectors[0][0], 1, 0)
}

func TestSymmetricEigendecompositionDenseCovariance(t *testing.T) {
	matrix := benchmarkCovariance(56)
	values, vectors := symmetricEigendecomposition(matrix)

	assertReconstruction(t, matrix, values, vectors)
	assertOrthonormal(t, vectors)
}

func TestSymmetricEigendecompositionGuards(t *testing.T) {
	matrix := [][]float64{
		{-1e-15, 1e-14},
		{-1e-14, 3},
	}
	values, vectors := symmetricEigendecomposition(matrix)

	if values[0] != 0 {
		t.Fatalf("negative eigenvalue was not floored: %g", values[0])
	}

	symmetrized := [][]float64{{-1e-15, 0}, {0, 3}}
	assertReconstruction(t, symmetrized, values, vectors)
}

func TestSymmetricEigendecompositionRejectsNonSquareMatrix(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected a panic for a non-square matrix")
		}
	}()

	symmetricEigendecomposition([][]float64{{1, 2}, {2}})
}

func BenchmarkEigenDecomposition(b *testing.B) {
	for _, size := range []int{56, 200, 1000} {
		matrix := benchmarkCovariance(size)

		b.Run(fmt.Sprintf("n=%d", size), func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				values, _ := symmetricEigendecomposition(matrix)
				if len(values) != size {
					b.Fatalf("got %d eigenvalues, want %d", len(values), size)
				}
			}
		})
	}
}

func hilbertMatrix(size int) [][]float64 {
	matrix := makeSquareMatrix(size)
	for row := range size {
		for column := range size {
			matrix[row][column] = 1 / float64(row+column+1)
		}
	}

	return matrix
}

func benchmarkCovariance(size int) [][]float64 {
	matrix := makeSquareMatrix(size)
	for row := range size {
		for column := range size {
			distance := math.Abs(float64(row - column))
			matrix[row][column] = math.Pow(0.9, distance)
		}
	}

	return matrix
}

func assertReconstruction(
	t *testing.T,
	matrix [][]float64,
	values []float64,
	vectors [][]float64,
) {
	t.Helper()

	for row := range matrix {
		for column := range matrix {
			got := 0.0
			for index, value := range values {
				got += vectors[row][index] * value * vectors[column][index]
			}

			assertClose(t, got, matrix[row][column], 1e-12)
		}
	}
}

func assertOrthonormal(t *testing.T, vectors [][]float64) {
	t.Helper()

	for left := range vectors {
		for right := range vectors {
			got := 0.0
			for row := range vectors {
				got += vectors[row][left] * vectors[row][right]
			}

			want := 0.0
			if left == right {
				want = 1
			}

			assertClose(t, got, want, 1e-12)
		}
	}
}

func assertClose(t *testing.T, got, want, tolerance float64) {
	t.Helper()

	scale := math.Max(1, math.Abs(want))
	if math.Abs(got-want) > tolerance*scale {
		t.Fatalf("got %.17g, want %.17g (tolerance %g)", got, want, tolerance)
	}
}
