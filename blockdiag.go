package cmaes

import (
	"fmt"
	"math"
)

// covarianceBlockDimension returns the largest independently adapted block.
// It is n for full covariance and one for separable covariance, making the
// block learning-rate correction meet both established modes exactly.
func covarianceBlockDimension(config *Config) int {
	switch config.CovarianceMode {
	case CovarianceSeparable:
		return 1
	case CovarianceBlock:
		if len(config.BlockGroups) > 0 {
			largest := 0
			for _, group := range config.BlockGroups {
				largest = max(largest, len(group))
			}

			return largest
		}

		return min(config.ProblemSize, config.BlockSize)
	default:
		return config.ProblemSize
	}
}

func validateBlockConfiguration(config *Config) error {
	if len(config.BlockGroups) == 0 {
		if config.BlockSize < 1 || config.BlockSize > config.ProblemSize {
			return fmt.Errorf("block_size must be in [1, problem_size] in block mode (got %d)",
				config.BlockSize)
		}

		return nil
	}

	seen := make([]bool, config.ProblemSize)
	for groupIndex, group := range config.BlockGroups {
		if len(group) == 0 {
			return fmt.Errorf("block_groups[%d] must not be empty", groupIndex)
		}

		for memberIndex, coordinate := range group {
			if coordinate < 0 || coordinate >= config.ProblemSize {
				return fmt.Errorf(
					"block_groups[%d][%d] coordinate %d is outside [0, problem_size)",
					groupIndex, memberIndex, coordinate,
				)
			}

			if seen[coordinate] {
				return fmt.Errorf("block_groups contains coordinate %d more than once", coordinate)
			}

			seen[coordinate] = true
		}
	}

	for coordinate, present := range seen {
		if !present {
			return fmt.Errorf("block_groups does not contain coordinate %d", coordinate)
		}
	}

	return nil
}

func covarianceBlockGroups(config *Config) [][]int {
	if len(config.BlockGroups) > 0 {
		return cloneIndexGroups(config.BlockGroups)
	}

	count := (config.ProblemSize + config.BlockSize - 1) / config.BlockSize
	groups := make([][]int, 0, count)

	for start := 0; start < config.ProblemSize; start += config.BlockSize {
		end := min(config.ProblemSize, start+config.BlockSize)

		group := make([]int, end-start)
		for index := range group {
			group[index] = start + index
		}

		groups = append(groups, group)
	}

	return groups
}

func cloneIndexGroups(groups [][]int) [][]int {
	cloned := make([][]int, len(groups))
	for index, group := range groups {
		cloned[index] = append([]int(nil), group...)
	}

	return cloned
}

func newBlockStrategyState(config *Config) *strategyState {
	groups := covarianceBlockGroups(config)
	// Canonicalize the two endpoints. Besides avoiding needless wrappers, this
	// guarantees the documented degeneracies preserve the exact established
	// search trajectory, including floating-point operation order.
	if len(groups) == 1 && len(groups[0]) == config.ProblemSize &&
		coordinatesAreIdentity(groups[0]) {
		state := newFullStrategyState(config)
		state.reportsBlocks = true

		return state
	}

	if allSingletonGroupsInCoordinateOrder(groups, config.ProblemSize) {
		state := newSeparableStrategyState(config)
		state.reportsBlocks = true

		return state
	}

	blockC := make([][][]float64, len(groups))
	blockB := make([][][]float64, len(groups))
	blockD := make([][]float64, len(groups))
	coordinateBlock := make([]int, config.ProblemSize)
	coordinateLocal := make([]int, config.ProblemSize)
	flatScales := make([]float64, config.ProblemSize)
	axisOffset := 0

	for block, group := range groups {
		blockC[block] = identityMatrix(len(group))
		blockB[block] = identityMatrix(len(group))
		blockD[block] = make([]float64, len(group))

		for local, coordinate := range group {
			blockD[block][local] = 1
			flatScales[axisOffset+local] = 1
			coordinateBlock[coordinate] = block
			coordinateLocal[coordinate] = local
		}

		axisOffset += len(group)
	}

	return &strategyState{
		mode:            CovarianceBlock,
		m:               append([]float64(nil), config.InitialMean...),
		psigma:          make([]float64, config.ProblemSize),
		pc:              make([]float64, config.ProblemSize),
		blockGroups:     groups,
		blockC:          blockC,
		blockB:          blockB,
		blockD:          blockD,
		coordinateBlock: coordinateBlock,
		coordinateLocal: coordinateLocal,
		d:               flatScales,
		sigma:           config.InitialSigma,
	}
}

func coordinatesAreIdentity(group []int) bool {
	for index, coordinate := range group {
		if coordinate != index {
			return false
		}
	}

	return true
}

func allSingletonGroupsInCoordinateOrder(groups [][]int, problemSize int) bool {
	if len(groups) != problemSize {
		return false
	}

	for coordinate, group := range groups {
		if len(group) != 1 || group[0] != coordinate {
			return false
		}
	}

	return true
}

func transformBlockNormal(state *strategyState, z []float64) []float64 {
	transformed := make([]float64, len(z))

	for block, group := range state.blockGroups {
		for row, coordinate := range group {
			for column, source := range group {
				transformed[coordinate] += state.blockB[block][row][column] *
					(state.blockD[block][column] * z[source])
			}
		}
	}

	return transformed
}

func inverseBlockSquareRootProduct(state *strategyState, step []float64) []float64 {
	product := make([]float64, len(step))
	coordinates := make([]float64, len(step))
	axisOffset := 0

	for block, group := range state.blockGroups {
		for column := range group {
			for row, source := range group {
				coordinates[axisOffset+column] += state.blockB[block][row][column] * step[source]
			}

			if state.blockD[block][column] > 0 {
				coordinates[axisOffset+column] /= state.blockD[block][column]
			} else {
				coordinates[axisOffset+column] = 0
			}
		}

		for row, coordinate := range group {
			for column := range group {
				product[coordinate] += state.blockB[block][row][column] *
					coordinates[axisOffset+column]
			}
		}

		axisOffset += len(group)
	}

	return product
}

func updateBlockCovariance(
	state *strategyState,
	population []candidate,
	hSigma bool,
	parameters strategyParameters,
) {
	decay := covarianceDecay(parameters, hSigma)
	negativeSteps := make([][]float64, len(parameters.negativeWeights))

	for index := range negativeSteps {
		populationIndex := len(parameters.weights) + index
		negativeSteps[index] = activeUpdateVector(
			state, population[populationIndex].adaptationStep())
	}

	for block, group := range state.blockGroups {
		covariance := state.blockC[block]

		for row := range group {
			for column := row; column < len(group); column++ {
				value := decay * covariance[row][column]
				value += parameters.c1 * state.pc[group[row]] * state.pc[group[column]]

				for index, weight := range parameters.weights {
					step := population[index].adaptationStep()
					value += parameters.cmu * weight * step[group[row]] * step[group[column]]
				}

				for index, weight := range parameters.negativeWeights {
					step := negativeSteps[index]
					value += parameters.cmu * weight * step[group[row]] * step[group[column]]
				}

				covariance[row][column] = value
				covariance[column][row] = value
			}
		}
	}
}

func decomposeBlocks(covariances [][][]float64) ([][][]float64, [][]float64, []float64) {
	axes := 0
	for _, covariance := range covariances {
		axes += len(covariance)
	}

	vectors := make([][][]float64, len(covariances))
	scales := make([][]float64, len(covariances))
	flatScales := make([]float64, 0, axes)

	for block, covariance := range covariances {
		values, blockVectors := symmetricEigendecomposition(covariance)
		blockScales := make([]float64, len(values))

		for index, value := range values {
			blockScales[index] = math.Sqrt(value)
		}

		vectors[block] = blockVectors
		scales[block] = blockScales
		flatScales = append(flatScales, blockScales...)
	}

	return vectors, scales, flatScales
}

func blockDistributionSnapshots(state *strategyState) []BlockDistributionSnapshot {
	blocks := make([]BlockDistributionSnapshot, len(state.blockGroups))

	for block, group := range state.blockGroups {
		blocks[block] = BlockDistributionSnapshot{
			Coordinates:  append([]int(nil), group...),
			Eigenvalues:  append([]float64(nil), state.blockD[block]...),
			Eigenvectors: clonePositions(state.blockB[block]),
		}
	}

	return blocks
}

// canonicalizedBlockSnapshots reports a block configuration that
// newBlockStrategyState collapsed onto a full or separable state. The caller
// configured block mode, so the snapshot keeps the block shape whatever
// BlockSize it chose: Blocks is never silently empty, and a separable
// canonicalization reports n singleton blocks rather than the dense n-by-n
// identity separable mode reports, which is what keeps the O(n) observer
// allocation the block documentation promises.
func canonicalizedBlockSnapshots(state *strategyState) []BlockDistributionSnapshot {
	if state.mode == CovarianceSeparable {
		blocks := make([]BlockDistributionSnapshot, len(state.m))
		for coordinate := range state.m {
			blocks[coordinate] = BlockDistributionSnapshot{
				Coordinates:  []int{coordinate},
				Eigenvalues:  []float64{state.d[coordinate]},
				Eigenvectors: [][]float64{{1}},
			}
		}

		return blocks
	}

	coordinates := make([]int, len(state.m))
	for index := range coordinates {
		coordinates[index] = index
	}

	return []BlockDistributionSnapshot{{
		Coordinates:  coordinates,
		Eigenvalues:  append([]float64(nil), state.d...),
		Eigenvectors: clonePositions(state.b),
	}}
}

func blockAxisStepHasEffect(state *strategyState, axis int) bool {
	offset := 0
	for block, group := range state.blockGroups {
		if axis >= offset+len(group) {
			offset += len(group)

			continue
		}

		localAxis := axis - offset
		for row, coordinate := range group {
			step := 0.1 * state.sigma * state.blockD[block][localAxis] *
				state.blockB[block][row][localAxis]
			if state.m[coordinate]+step != state.m[coordinate] {
				return true
			}
		}

		return false
	}

	return false
}
