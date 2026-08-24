package cmaes

import (
	"math"
	"sort"
)

const (
	jacobiTolerance = 1e-14
	maxJacobiSweeps = 60
)

// symmetricEigendecomposition computes the eigenvalues and eigenvectors of a
// real symmetric matrix with cyclic Jacobi sweeps. The returned eigenvalues
// are ascending and the matching eigenvectors are the columns of vectors.
//
// The input is not modified. Roundoff can make a covariance matrix slightly
// asymmetric or give it a tiny negative eigenvalue, so the working copy is
// explicitly symmetrized and negative eigenvalues are clamped to zero before
// callers form their square roots.
func symmetricEigendecomposition(matrix [][]float64) ([]float64, [][]float64) {
	size := len(matrix)
	work := symmetricCopy(matrix)
	vectors := identityMatrix(size)

	for range maxJacobiSweeps {
		threshold, converged := jacobiThreshold(work)
		if converged {
			break
		}

		for pivot := range size {
			for other := pivot + 1; other < size; other++ {
				jacobiRotate(work, vectors, pivot, other, threshold)
			}
		}
	}

	values := make([]float64, size)
	for index := range size {
		values[index] = math.Max(0, work[index][index])
	}

	sortEigenpairs(values, vectors)

	return values, vectors
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

func sortEigenpairs(values []float64, vectors [][]float64) {
	indices := make([]int, len(values))
	for index := range indices {
		indices[index] = index
	}

	sort.SliceStable(indices, func(left, right int) bool {
		return values[indices[left]] < values[indices[right]]
	})

	sortedValues := append([]float64(nil), values...)
	sortedVectors := makeSquareMatrix(len(values))

	for destination, source := range indices {
		sortedValues[destination] = values[source]
		for row := range vectors {
			sortedVectors[row][destination] = vectors[row][source]
		}
	}

	copy(values, sortedValues)

	for row := range vectors {
		copy(vectors[row], sortedVectors[row])
	}
}
