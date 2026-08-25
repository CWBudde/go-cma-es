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
			// The reference spectrum is independently checkable: its sum
			// reproduces the exact trace 1 + 1/3 + 1/5 + 1/7 + 1/9 + 1/11 and
			// its product the exact determinant 1/186313420339200000. Both are
			// asserted for the computed spectrum in
			// TestSymmetricEigendecompositionHilbertInvariants.
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

			// A symmetric eigensolver is backward stable, so the absolute
			// error of every eigenvalue is bounded by about eps times the
			// largest eigenvalue. Relative to the smallest eigenvalue that is
			// cond(A) * eps, which is 9e-16 for the two well-conditioned cases
			// and 3e-9 for Hilbert-6 (cond 1.5e7). Asserting that bound keeps
			// the large eigenvalues pinned to full precision while allowing
			// the small ones exactly the slack the conditioning demands.
			tolerance := math.Max(machineEpsilon, conditionNumber(test.want)*machineEpsilon)
			for index, want := range test.want {
				assertClose(t, values[index], want, tolerance)
			}

			assertReconstruction(t, test.matrix, values, vectors)
			assertOrthonormal(t, vectors)
		})
	}
}

// TestSymmetricEigendecompositionHilbertInvariants pins the tiny end of an
// ill-conditioned spectrum against two exact closed forms rather than against
// a single computed number: the trace and the determinant of the Hilbert
// matrix. The trace pins the large end to full double precision, while the
// determinant is the product of all six eigenvalues and is therefore only as
// accurate as the smallest one, 1.08e-7, allows.
func TestSymmetricEigendecompositionHilbertInvariants(t *testing.T) {
	values, _ := symmetricEigendecomposition(hilbertMatrix(6))

	sum := 0.0
	product := 1.0

	for _, value := range values {
		sum += value
		product *= value
	}

	trace := 1 + 1.0/3 + 1.0/5 + 1.0/7 + 1.0/9 + 1.0/11
	determinant := 1 / 186313420339200000.0

	// The sum is dominated by the largest eigenvalue and is accurate to
	// roundoff. The product inherits the relative error of every factor, so it
	// is bounded by the number of factors times cond(A) * eps, about 2e-8 for
	// Hilbert-6.
	assertClose(t, sum, trace, 4*machineEpsilon)
	assertClose(t, product, determinant, float64(len(values))*1.5e7*machineEpsilon)
}

// TestSymmetricEigendecompositionKeepsGenuineIllConditioning is the guard
// against repairing eigenvalues that need no repair. [[a,b],[b,a]] has the
// exact eigenvalues a-b and a+b; with a = (1+2^-50)/2 and b = (1-2^-50)/2,
// both of which are exact in binary floating point, that is 2^-50 and 1, a
// condition number of 2^50 = 1.13e15. The matrix is positive definite, so the
// spectrum must come back untouched and its condition number must still
// exceed the default ConditionCov, which is what lets the condition_number
// criterion fire on a genuinely degenerate run.
func TestSymmetricEigendecompositionKeepsGenuineIllConditioning(t *testing.T) {
	smallest := math.Ldexp(1, -50)

	upper := (1 + smallest) / 2
	lower := (1 - smallest) / 2

	matrix := [][]float64{
		{upper, lower},
		{lower, upper},
	}

	values, vectors := symmetricEigendecomposition(matrix)

	assertClose(t, values[0], smallest, machineEpsilon)
	assertClose(t, values[1], 1, machineEpsilon)

	if values[0] >= eigenvalueRepairRatio*values[1] {
		t.Fatalf("smallest eigenvalue = %g, want it below the repair value %g",
			values[0], eigenvalueRepairRatio*values[1])
	}

	condition := conditionNumber(values)
	if math.IsInf(condition, 0) || condition <= defaultConditionCov {
		t.Fatalf("condition number = %g, want finite and above %g", condition, defaultConditionCov)
	}

	assertReconstruction(t, matrix, values, vectors)
	assertOrthonormal(t, vectors)
}

// TestSymmetricEigendecompositionRepairsBelowSmallestPositive covers the case
// where the repair value would exceed a genuinely positive eigenvalue.
// diag(0, 2^-50, 1) is already diagonal, so its exact spectrum is its own
// diagonal; the repaired zero must not overtake 2^-50 and break the ascending
// order the callers rely on.
func TestSymmetricEigendecompositionRepairsBelowSmallestPositive(t *testing.T) {
	smallest := math.Ldexp(1, -50)

	values, _ := symmetricEigendecomposition([][]float64{
		{0, 0, 0},
		{0, smallest, 0},
		{0, 0, 1},
	})

	assertClose(t, values[0], smallest, 0)
	assertClose(t, values[1], smallest, 0)
	assertClose(t, values[2], 1, 0)
}

// TestSymmetricEigendecompositionSingular decomposes a genuinely singular
// matrix. C = [[1,1,0],[1,1,0],[0,0,1]] is the direct sum of [[1,1],[1,1]],
// whose eigenvalues are 0 and 2, and the 1x1 block 1, so the exact spectrum is
// {0, 1, 2}.
func TestSymmetricEigendecompositionSingular(t *testing.T) {
	matrix := [][]float64{
		{1, 1, 0},
		{1, 1, 0},
		{0, 0, 1},
	}

	values, vectors := symmetricEigendecomposition(matrix)

	for index, value := range values {
		if value <= 0 {
			t.Fatalf("eigenvalue[%d] = %g, want strictly positive", index, value)
		}
	}

	assertClose(t, values[0], 2*eigenvalueRepairRatio, 1e-15)
	assertClose(t, values[1], 1, 1e-14)
	assertClose(t, values[2], 2, 1e-14)

	condition := conditionNumber(values)
	if math.IsInf(condition, 0) || condition > defaultConditionCov {
		t.Fatalf("condition number = %g, want finite and at most %g", condition, defaultConditionCov)
	}

	assertOrthonormal(t, vectors)
	assertReconstructionWithin(t, matrix, values, vectors, 1e-11)
}

// TestSymmetricEigendecompositionIndefinite decomposes [[0,1],[1,0]], whose
// exact eigenvalues are -1 and +1 with eigenvectors (1,-1)/sqrt(2) and
// (1,1)/sqrt(2). Active CMA can make an intermediate covariance indefinite
// like this; the negative eigenvalue must come back as a small positive number
// rather than as zero, and the basis must still be the exact one.
func TestSymmetricEigendecompositionIndefinite(t *testing.T) {
	matrix := [][]float64{
		{0, 1},
		{1, 0},
	}

	values, vectors := symmetricEigendecomposition(matrix)

	assertClose(t, values[0], eigenvalueRepairRatio, 1e-15)
	assertClose(t, values[1], 1, 1e-14)

	if values[0] <= 0 {
		t.Fatalf("clamped eigenvalue = %g, want strictly positive", values[0])
	}

	condition := conditionNumber(values)
	if math.IsInf(condition, 0) || condition > defaultConditionCov {
		t.Fatalf("condition number = %g, want finite and at most %g", condition, defaultConditionCov)
	}

	assertOrthonormal(t, vectors)

	// The eigenvector for +1 is (1,1)/sqrt(2) up to an overall sign.
	sign := math.Copysign(1, vectors[0][1])
	assertClose(t, sign*vectors[0][1], math.Sqrt2/2, 1e-15)
	assertClose(t, sign*vectors[1][1], math.Sqrt2/2, 1e-15)
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

	// The working copy is symmetrized to diag(-1e-15, 3), so the exact
	// spectrum is {-1e-15, 3} and the negative eigenvalue must be lifted to
	// the floor instead of to zero.
	if values[0] <= 0 {
		t.Fatalf("negative eigenvalue was not lifted above zero: %g", values[0])
	}

	assertClose(t, values[0], 3*eigenvalueRepairRatio, 1e-15)
	assertClose(t, values[1], 3, 1e-15)
	assertOrthonormal(t, vectors)
}

// TestSymmetricEigendecompositionNegativeSemidefinite covers the degenerate
// case where no eigenvalue is positive and there is therefore no scale for the
// floor to be relative to. diag(-2, -1) has exactly that spectrum.
func TestSymmetricEigendecompositionNegativeSemidefinite(t *testing.T) {
	values, vectors := symmetricEigendecomposition([][]float64{
		{-2, 0},
		{0, -1},
	})

	for index, value := range values {
		assertClose(t, value, 1, 0)

		if value <= 0 {
			t.Fatalf("eigenvalue[%d] = %g, want strictly positive", index, value)
		}
	}

	assertOrthonormal(t, vectors)
}

// TestSymmetricEigendecompositionSweepBudget pins the convergence signal that
// symmetricEigendecomposition turns into a panic. A matrix that needs at least
// one rotation cannot be diagonal after zero sweeps, while the same matrix
// converges within the production budget.
func TestSymmetricEigendecompositionSweepBudget(t *testing.T) {
	matrix := [][]float64{
		{1.75, -1.299038105676658},
		{-1.299038105676658, 3.25},
	}

	if _, _, converged := symmetricEigendecompositionWithStatus(matrix, 0); converged {
		t.Fatal("zero sweeps reported convergence for a non-diagonal matrix")
	}

	values, _, converged := symmetricEigendecompositionWithStatus(matrix, maxJacobiSweeps)
	if !converged {
		t.Fatal("the production sweep budget did not converge")
	}

	assertClose(t, values[0], 1, 1e-14)
	assertClose(t, values[1], 4, 1e-14)

	// An already diagonal matrix is converged before the first sweep, so the
	// budget is irrelevant to it.
	if _, _, converged := symmetricEigendecompositionWithStatus([][]float64{{2, 0}, {0, 5}}, 0); !converged {
		t.Fatal("a diagonal matrix was reported as not converged")
	}
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

// machineEpsilon is the double precision unit roundoff, 2^-52.
const machineEpsilon = 2.220446049250313e-16

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

// assertReconstruction checks B*diag(values)*B^T against the original matrix
// with an absolute tolerance scaled by the largest entry of that matrix, which
// is the scale a backward-stable decomposition can be held to.
func assertReconstruction(
	t *testing.T,
	matrix [][]float64,
	values []float64,
	vectors [][]float64,
) {
	t.Helper()

	scale := 0.0

	for row := range matrix {
		for column := range matrix[row] {
			scale = math.Max(scale, math.Abs(matrix[row][column]))
		}
	}

	assertReconstructionWithin(t, matrix, values, vectors, 1e-12*math.Max(1, scale))
}

func assertReconstructionWithin(
	t *testing.T,
	matrix [][]float64,
	values []float64,
	vectors [][]float64,
	tolerance float64,
) {
	t.Helper()

	for row := range matrix {
		for column := range matrix {
			got := 0.0
			for index, value := range values {
				got += vectors[row][index] * value * vectors[column][index]
			}

			if math.Abs(got-matrix[row][column]) > tolerance {
				t.Fatalf(
					"reconstruction[%d][%d] = %.17g, want %.17g (tolerance %g)",
					row, column, got, matrix[row][column], tolerance,
				)
			}
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

			// The accumulated rotations drift by about n*eps from exact
			// orthonormality, which is 1e-14 at the n=56 covariance below.
			if math.Abs(got-want) > 1e-13 {
				t.Fatalf("vectors[:,%d].vectors[:,%d] = %.17g, want %.17g", left, right, got, want)
			}
		}
	}
}

// assertClose compares got and want with a relative tolerance, so that an
// assertion on a tiny eigenvalue is exactly as strict as one on a large
// eigenvalue. A want of zero degenerates to an absolute comparison, and a
// tolerance of zero still means bit equality.
func assertClose(t *testing.T, got, want, tolerance float64) {
	t.Helper()

	scale := math.Abs(want)
	if scale == 0 {
		scale = 1
	}

	if math.Abs(got-want) > tolerance*scale {
		t.Fatalf("got %.17g, want %.17g (relative tolerance %g)", got, want, tolerance)
	}
}
