package cmaes

import "math"

func makeSquareMatrix(size int) [][]float64 {
	matrix := make([][]float64, size)
	data := make([]float64, size*size)

	for row := range size {
		matrix[row] = data[row*size : (row+1)*size]
	}

	return matrix
}

func identityMatrix(size int) [][]float64 {
	matrix := makeSquareMatrix(size)
	for index := range size {
		matrix[index][index] = 1
	}

	return matrix
}

// symmetricRankOneUpdate applies matrix += scale * vector * vector^T while
// writing both triangles explicitly to keep the covariance exactly symmetric.
func symmetricRankOneUpdate(matrix [][]float64, scale float64, vector []float64) {
	for row := range vector {
		for column := row; column < len(vector); column++ {
			value := matrix[row][column] + scale*vector[row]*vector[column]
			matrix[row][column] = value
			matrix[column][row] = value
		}
	}
}

func matrixVectorProduct(matrix [][]float64, vector []float64) []float64 {
	product := make([]float64, len(matrix))

	for row := range matrix {
		for column, value := range matrix[row] {
			product[row] += value * vector[column]
		}
	}

	return product
}

// conditionNumber returns the spectral condition number represented by a set
// of non-negative eigenvalues. A singular or invalid spectrum has infinite
// condition number; an empty spectrum has condition number zero.
func conditionNumber(eigenvalues []float64) float64 {
	if len(eigenvalues) == 0 {
		return 0
	}

	minimum := math.Inf(1)
	maximum := 0.0

	for _, value := range eigenvalues {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
			return math.Inf(1)
		}

		minimum = math.Min(minimum, value)
		maximum = math.Max(maximum, value)
	}

	if minimum == 0 {
		return math.Inf(1)
	}

	return maximum / minimum
}
