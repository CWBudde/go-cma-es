package cmaes

import (
	"fmt"
	"math"
	"sort"
)

const (
	jacobiTolerance = 1e-14
	// maxJacobiSweeps is a hard cap, not an expectation: random symmetric
	// positive definite covariance matrices converge in 1 sweep at n=2, 7 at
	// n=20, 9 at n=56 and 11 at n=200, so the cap is only reached by a
	// genuinely pathological input. Reaching it is reported rather than
	// silently accepted; see symmetricEigendecomposition.
	maxJacobiSweeps = 60
	// eigenvalueRepairRatio positions an eigenvalue that has to be repaired,
	// relative to the largest eigenvalue of the same matrix. Roundoff, and the
	// negative rank-mu weights of active CMA, can drive an eigenvalue to zero
	// or below. Flooring such an eigenvalue to exactly zero gives its axis a
	// step size of exactly zero, so B*diag(D)*z never moves along it again and
	// the dimension is dead for the rest of the run; conditionNumber also turns
	// infinite and trips the condition_number criterion immediately. Hansen's
	// cma.py instead nudges the spectrum just positive, and only when an
	// eigenvalue is actually non-positive, so a merely small positive
	// eigenvalue keeps its value and a genuinely ill-conditioned covariance
	// keeps its condition number. 1e-13 is about a thousand times the roundoff
	// level of the largest eigenvalue, so the repaired axis is well above
	// noise while contributing a negligible step.
	eigenvalueRepairRatio = 1e-13
)

// symmetricEigendecomposition computes the eigenvalues and eigenvectors of a
// real symmetric matrix with cyclic Jacobi sweeps. The returned eigenvalues
// are ascending and the matching eigenvectors are the columns of vectors.
//
// The input is not modified. Roundoff can make a covariance matrix slightly
// asymmetric or give it a non-positive eigenvalue, so the working copy is
// explicitly symmetrized and any non-positive eigenvalue is repaired to a
// small positive value before callers form their square roots.
//
// The function panics when the sweeps fail to converge within maxJacobiSweeps.
// That keeps the signature callers depend on while making a pathological input
// loud instead of silently returning a half-diagonalized matrix.
func symmetricEigendecomposition(matrix [][]float64) ([]float64, [][]float64) {
	values, vectors, converged := symmetricEigendecompositionWithStatus(matrix, maxJacobiSweeps)
	if !converged {
		panic(fmt.Sprintf(
			"cmaes: Jacobi eigendecomposition of a %dx%d matrix did not converge in %d sweeps",
			len(matrix), len(matrix), maxJacobiSweeps,
		))
	}

	return values, vectors
}

// symmetricEigendecompositionWithStatus is symmetricEigendecomposition with an
// explicit sweep budget and a convergence flag, so the sweep loop can be tested
// without a pathological matrix.
func symmetricEigendecompositionWithStatus(
	matrix [][]float64,
	maxSweeps int,
) ([]float64, [][]float64, bool) {
	size := len(matrix)
	work := symmetricCopy(matrix)
	vectors := identityMatrix(size)

	converged := false

	for range maxSweeps {
		threshold, done := jacobiThreshold(work)
		if done {
			converged = true

			break
		}

		for pivot := range size {
			for other := pivot + 1; other < size; other++ {
				jacobiRotate(work, vectors, pivot, other, threshold)
			}
		}
	}

	if !converged {
		// The budget may have run out on the sweep that finished the job, and
		// a zero budget never enters the loop at all.
		_, converged = jacobiThreshold(work)
	}

	values := make([]float64, size)
	for index := range size {
		values[index] = work[index][index]
	}

	sortEigenpairs(values, vectors)
	repairEigenvalues(values)

	return values, vectors, converged
}

// repairEigenvalues replaces every non-positive entry of an ascending spectrum
// with a small positive value, leaving positive eigenvalues exactly as
// computed: a covariance that is genuinely ill-conditioned but positive
// definite must keep its condition number so that the condition_number
// criterion can still see it. The replacement is the smaller of
// eigenvalueRepairRatio times the largest eigenvalue and the smallest
// genuinely positive eigenvalue, which keeps the spectrum ascending and never
// invents an axis longer than one the matrix actually has.
//
// A matrix without a single finite positive eigenvalue carries no scale to be
// relative to, so its spectrum is replaced by the unit spectrum: sampling
// stays isotropic and the run can recover, which an all-zero spectrum would
// not allow.
func repairEigenvalues(values []float64) {
	if len(values) == 0 {
		return
	}

	lambdaMax := values[len(values)-1]

	// Negated so that a NaN, which compares false against everything, takes
	// the fallback as well.
	if !(lambdaMax > 0) || math.IsInf(lambdaMax, 1) {
		for index := range values {
			values[index] = 1
		}

		return
	}

	repaired := eigenvalueRepairRatio * lambdaMax

	for _, value := range values {
		if value > 0 {
			repaired = math.Min(repaired, value)

			break
		}
	}

	for index, value := range values {
		if !(value > 0) {
			values[index] = repaired
		}
	}
}

func symmetricCopy(matrix [][]float64) [][]float64 {
	size := len(matrix)
	for row := range size {
		if len(matrix[row]) != size {
			panic("cmaes: eigendecomposition requires a square matrix")
		}
	}

	copyMatrix := makeSquareMatrix(size)

	for row := range size {
		copyMatrix[row][row] = matrix[row][row]
		for column := row + 1; column < size; column++ {
			value := (matrix[row][column] + matrix[column][row]) / 2
			copyMatrix[row][column] = value
			copyMatrix[column][row] = value
		}
	}

	return copyMatrix
}

func jacobiThreshold(matrix [][]float64) (float64, bool) {
	maxDiagonal := 0.0
	maxOffDiagonal := 0.0

	for row := range matrix {
		maxDiagonal = math.Max(maxDiagonal, math.Abs(matrix[row][row]))
		for column := row + 1; column < len(matrix); column++ {
			maxOffDiagonal = math.Max(maxOffDiagonal, math.Abs(matrix[row][column]))
		}
	}

	threshold := jacobiTolerance * maxDiagonal

	return threshold, maxOffDiagonal <= threshold
}

// jacobiRotate annihilates matrix[pivot][other] with a numerically stable
// Jacobi rotation and accumulates the rotation into the eigenvector basis.
func jacobiRotate(matrix, vectors [][]float64, pivot, other int, threshold float64) {
	entry := matrix[pivot][other]
	if math.Abs(entry) <= threshold {
		return
	}

	pivotValue := matrix[pivot][pivot]
	otherValue := matrix[other][other]
	tau := (otherValue - pivotValue) / (2 * entry)

	tangent := 1 / (math.Abs(tau) + math.Hypot(1, tau))
	if tau < 0 {
		tangent = -tangent
	}

	cosine := 1 / math.Hypot(1, tangent)
	sine := tangent * cosine

	matrix[pivot][pivot] = pivotValue - tangent*entry
	matrix[other][other] = otherValue + tangent*entry
	matrix[pivot][other] = 0
	matrix[other][pivot] = 0

	for index := range matrix {
		if index == pivot || index == other {
			continue
		}

		toPivot := matrix[index][pivot]
		toOther := matrix[index][other]
		rotatedPivot := cosine*toPivot - sine*toOther
		rotatedOther := sine*toPivot + cosine*toOther
		matrix[index][pivot] = rotatedPivot
		matrix[pivot][index] = rotatedPivot
		matrix[index][other] = rotatedOther
		matrix[other][index] = rotatedOther
	}

	for row := range vectors {
		toPivot := vectors[row][pivot]
		toOther := vectors[row][other]
		vectors[row][pivot] = cosine*toPivot - sine*toOther
		vectors[row][other] = sine*toPivot + cosine*toOther
	}
}

// sortEigenpairs orders the spectrum ascending and moves the matching
// eigenvector columns with it. The permutation is applied by walking its
// cycles so that no second n*n matrix is allocated.
func sortEigenpairs(values []float64, vectors [][]float64) {
	size := len(values)

	indices := make([]int, size)
	for index := range indices {
		indices[index] = index
	}

	sort.SliceStable(indices, func(left, right int) bool {
		return values[indices[left]] < values[indices[right]]
	})

	column := make([]float64, size)
	moved := make([]bool, size)

	for start := range size {
		if moved[start] || indices[start] == start {
			moved[start] = true

			continue
		}

		heldValue := values[start]
		readColumn(vectors, start, column)

		destination := start
		for {
			source := indices[destination]
			moved[destination] = true

			if source == start {
				values[destination] = heldValue
				writeColumn(vectors, destination, column)

				break
			}

			values[destination] = values[source]
			copyColumn(vectors, destination, source)
			destination = source
		}
	}
}

func readColumn(vectors [][]float64, column int, destination []float64) {
	for row := range vectors {
		destination[row] = vectors[row][column]
	}
}

func writeColumn(vectors [][]float64, column int, source []float64) {
	for row := range vectors {
		vectors[row][column] = source[row]
	}
}

func copyColumn(vectors [][]float64, destination, source int) {
	for row := range vectors {
		vectors[row][destination] = vectors[row][source]
	}
}
