package cmaes

import (
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"
)

func TestBlockConfigurationValidation(t *testing.T) {
	tests := []struct {
		name   string
		groups [][]int
		size   int
		want   string
	}{
		{name: "zero size", size: 0, want: "block_size"},
		{name: "oversized", size: 5, want: "block_size"},
		{name: "empty group", groups: [][]int{{0, 1}, {}, {2, 3}}, want: "must not be empty"},
		{name: "negative coordinate", groups: [][]int{{0, -1}, {1, 2, 3}}, want: "outside"},
		{name: "large coordinate", groups: [][]int{{0, 4}, {1, 2, 3}}, want: "outside"},
		{name: "duplicate", groups: [][]int{{0, 1}, {1, 2, 3}}, want: "more than once"},
		{name: "missing", groups: [][]int{{0, 2}, {3}}, want: "does not contain coordinate 1"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := NewBlockDiagonalConfig(4, test.size)
			config.ObjectiveFunc = sphere
			config.LowerBound = -5
			config.UpperBound = 5
			config.BlockGroups = test.groups

			err := config.Validate()
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Validate() = %v, want error containing %q", err, test.want)
			}
		})
	}
}

func TestBlockGroupsOverrideBlockSizeAndAreCopied(t *testing.T) {
	config := NewBlockDiagonalConfig(6, 0)
	config.BlockGroups = [][]int{{0, 3}, {1, 4}, {2, 5}}
	config.ObjectiveFunc = sphere
	config.LowerBound = -5
	config.UpperBound = 5

	err := config.Validate()
	if err != nil {
		t.Fatalf("Validate() = %v", err)
	}

	state := newStrategyState(config)
	config.BlockGroups[0][0] = 5

	if got := state.blockGroups[0][0]; got != 0 {
		t.Fatalf("state block coordinate = %d after config mutation, want copied value 0", got)
	}

	if len(state.blockC) != 3 || len(state.blockC[0]) != 2 {
		t.Fatalf("block covariance shape = %d blocks, first size %d, want 3 and 2",
			len(state.blockC), len(state.blockC[0]))
	}
}

func TestBlockStateUsesBoundedStorage(t *testing.T) {
	const (
		dimension = 14_000
		blockSize = 7
	)

	state := newStrategyState(NewBlockDiagonalConfig(dimension, blockSize))
	if state.c != nil || state.b != nil || state.diagonal != nil {
		t.Fatal("block state allocated a full or separable covariance representation")
	}

	entries := 0
	for block := range state.blockC {
		entries += len(state.blockC[block]) * len(state.blockC[block])
		entries += len(state.blockB[block]) * len(state.blockB[block])
	}

	if entries != 2*dimension*blockSize {
		t.Fatalf("matrix entries = %d, want 2*n*k = %d", entries, 2*dimension*blockSize)
	}
}

func TestBlockLearningRateInterpolatesEstablishedEndpoints(t *testing.T) {
	const dimension = 20

	full := NewDefaultConfig(dimension)
	full.ActiveCMA = false
	block := NewBlockDiagonalConfig(dimension, 4)
	block.ActiveCMA = false
	separable := NewSeparableConfig(dimension)
	separable.ActiveCMA = false

	fullParameters := deriveStrategyParameters(full)
	blockParameters := deriveStrategyParameters(block)
	separableParameters := deriveStrategyParameters(separable)
	want := math.Min(
		1-blockParameters.c1,
		fullParameters.cmu*float64(dimension+2)/float64(block.BlockSize+2),
	)

	if blockParameters.cmu != want {
		t.Errorf("block cmu = %.17g, want %.17g", blockParameters.cmu, want)
	}

	if !(fullParameters.cmu < blockParameters.cmu &&
		blockParameters.cmu < separableParameters.cmu) {
		t.Errorf("learning rates full=%g block=%g separable=%g, want strict interpolation",
			fullParameters.cmu, blockParameters.cmu, separableParameters.cmu)
	}
}

func TestBlockSamplingAndInverseTransformWithNonContiguousGroups(t *testing.T) {
	config := NewBlockDiagonalConfig(6, 2)
	config.BlockGroups = [][]int{{0, 3}, {1, 4}, {2, 5}}
	state := newStrategyState(config)
	state.blockB = [][][]float64{
		{{0, -1}, {1, 0}},
		{{1, 0}, {0, 1}},
		{{0, 1}, {-1, 0}},
	}
	state.blockD = [][]float64{{2, 3}, {4, 5}, {6, 7}}

	population := samplePopulation(state, 1, rand.New(rand.NewSource(801)))
	current := population[0]
	want := []float64{
		-3 * current.z[3],
		4 * current.z[1],
		7 * current.z[5],
		2 * current.z[0],
		5 * current.z[4],
		-6 * current.z[2],
	}
	wantWhitened := []float64{
		-current.z[3], current.z[1], current.z[5],
		current.z[0], current.z[4], -current.z[2],
	}

	assertVectorClose(t, current.y, want, 0)
	assertVectorClose(t, inverseCovarianceSquareRootProduct(state, current.y), wantWhitened, 1e-15)
}

func TestBlockDegeneratesExactlyToEstablishedModes(t *testing.T) {
	const dimension = 8

	base := NewDefaultConfig(dimension)
	base.ObjectiveFunc = blockStructuredEllipsoid
	seed := int64(802)
	base.Seed = &seed
	base.LowerBound = -10
	base.UpperBound = 10
	base.InitialMean = filledVector(dimension, 2)
	base.InitialSigma = 1
	base.Convergence = nil
	base.MaxIterations = 80

	tests := []struct {
		name      string
		reference CovarianceMode
		blockSize int
	}{
		{name: "full", reference: CovarianceFull, blockSize: dimension},
		{name: "separable", reference: CovarianceSeparable, blockSize: 1},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			referenceConfig := *base
			referenceConfig.InitialMean = append([]float64(nil), base.InitialMean...)
			referenceConfig.CovarianceMode = test.reference

			blockConfig := *base
			blockConfig.InitialMean = append([]float64(nil), base.InitialMean...)
			blockConfig.CovarianceMode = CovarianceBlock
			blockConfig.BlockSize = test.blockSize

			referenceResult, err := Optimize(&referenceConfig)
			if err != nil {
				t.Fatalf("reference Optimize() = %v", err)
			}

			blockResult, err := Optimize(&blockConfig)
			if err != nil {
				t.Fatalf("block Optimize() = %v", err)
			}

			if !reflect.DeepEqual(blockResult, referenceResult) {
				t.Fatal("block endpoint result differs from its established covariance mode")
			}
		})
	}
}

func TestBlockDistributionSnapshotKeepsSparseEigenvectors(t *testing.T) {
	config := NewBlockDiagonalConfig(4, 2)
	state := newStrategyState(config)
	state.blockB = [][][]float64{
		{{0, -1}, {1, 0}},
		{{1, 0}, {0, 1}},
	}
	state.blockD = [][]float64{{2, 3}, {4, 5}}
	state.d = []float64{2, 3, 4, 5}

	snapshot := distributionSnapshot(state, 3, 30)
	if snapshot.Eigenvectors != nil {
		t.Fatalf("dense eigenvectors = %v, want nil in block mode", snapshot.Eigenvectors)
	}

	if len(snapshot.Blocks) != 2 {
		t.Fatalf("snapshot blocks = %d, want 2", len(snapshot.Blocks))
	}

	assertMatrixClose(t, snapshot.Blocks[0].Eigenvectors, state.blockB[0], 0)
	assertVectorClose(t, snapshot.Blocks[1].Eigenvalues, state.blockD[1], 0)

	if !reflect.DeepEqual(snapshot.Blocks[0].Coordinates, []int{0, 1}) {
		t.Errorf("first block coordinates = %v, want [0 1]", snapshot.Blocks[0].Coordinates)
	}

	snapshot.Blocks[0].Eigenvectors[0][0] = 99
	snapshot.Blocks[0].Coordinates[0] = 99

	if state.blockB[0][0][0] == 99 || state.blockGroups[0][0] == 99 {
		t.Fatal("distribution snapshot aliases block eigensystem state")
	}
}

func TestBlockDistributionSnapshotReportsBlocksWhenCanonicalized(t *testing.T) {
	tests := []struct {
		name       string
		blockSize  int
		wantBlocks int
		wantSize   int
	}{
		{name: "separable canonicalization", blockSize: 1, wantBlocks: 4, wantSize: 1},
		{name: "full canonicalization", blockSize: 4, wantBlocks: 1, wantSize: 4},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := NewBlockDiagonalConfig(4, test.blockSize)
			state := newStrategyState(config)

			snapshot := distributionSnapshot(state, 1, 10)
			if snapshot.Eigenvectors != nil {
				t.Fatalf("dense eigenvectors = %v, want nil for a block configuration",
					snapshot.Eigenvectors)
			}

			if len(snapshot.Blocks) != test.wantBlocks {
				t.Fatalf("snapshot blocks = %d, want %d", len(snapshot.Blocks), test.wantBlocks)
			}

			covered := 0

			for _, block := range snapshot.Blocks {
				if len(block.Coordinates) != test.wantSize ||
					len(block.Eigenvalues) != test.wantSize ||
					len(block.Eigenvectors) != test.wantSize {
					t.Fatalf("block shape = (%d, %d, %d), want %d",
						len(block.Coordinates), len(block.Eigenvalues),
						len(block.Eigenvectors), test.wantSize)
				}

				covered += len(block.Coordinates)
			}

			if covered != config.ProblemSize {
				t.Errorf("covered coordinates = %d, want %d", covered, config.ProblemSize)
			}

			snapshot.Blocks[0].Eigenvectors[0][0] = 99
			if state.b != nil && state.b[0][0] == 99 {
				t.Fatal("distribution snapshot aliases the canonicalized eigensystem")
			}
		})
	}
}

func TestBlockCMARecoversRotatedBlockEllipsoid(t *testing.T) {
	const dimension = 20

	run := func(mode CovarianceMode) *Result {
		t.Helper()

		config := NewDefaultConfig(dimension)
		config.ObjectiveFunc = blockStructuredEllipsoid
		seed := int64(803)
		config.Seed = &seed
		config.LowerBound = -10
		config.UpperBound = 10
		config.InitialMean = filledVector(dimension, 3)
		config.InitialSigma = 1
		config.CovarianceMode = mode
		config.BlockSize = 2
		config.Convergence = nil
		config.MaxIterations = 800

		result, err := Optimize(config)
		if err != nil {
			t.Fatalf("Optimize(%s) = %v", mode, err)
		}

		return result
	}

	block := run(CovarianceBlock)
	separable := run(CovarianceSeparable)

	if block.GlobalBest.Cost >= 1e-10 {
		t.Errorf("block CMA cost = %g, want < 1e-10", block.GlobalBest.Cost)
	}

	if separable.GlobalBest.Cost <= 1 {
		t.Errorf("separable cost = %g, want > 1 under the same budget",
			separable.GlobalBest.Cost)
	}

	if separable.GlobalBest.Cost <= block.GlobalBest.Cost*1e4 {
		t.Errorf("separable cost = %g, want > 10,000 * block cost %g",
			separable.GlobalBest.Cost, block.GlobalBest.Cost)
	}
}

func TestBlockCMALearnsNonContiguousGroups(t *testing.T) {
	const half = 6

	groups := make([][]int, half)
	for index := range groups {
		groups[index] = []int{index, index + half}
	}

	config := NewBlockDiagonalConfig(2*half, 0)
	config.BlockGroups = groups
	config.ObjectiveFunc = func(position []float64) float64 {
		cost := 0.0

		for index := range half {
			easy := (position[index] + position[index+half]) / math.Sqrt2
			hard := (position[index] - position[index+half]) / math.Sqrt2
			cost += easy*easy + 1e6*hard*hard
		}

		return cost
	}
	seed := int64(806)
	config.Seed = &seed
	config.LowerBound = -10
	config.UpperBound = 10
	config.InitialMean = filledVector(2*half, 3)
	config.InitialSigma = 1
	config.Convergence = nil
	config.MaxIterations = 700

	result, err := Optimize(config)
	if err != nil {
		t.Fatalf("Optimize() = %v", err)
	}

	if result.GlobalBest.Cost >= 1e-10 {
		t.Errorf("non-contiguous block cost = %g, want < 1e-10", result.GlobalBest.Cost)
	}
}

func TestOptimizeBlockIsDeterministicAndParallelEquivalent(t *testing.T) {
	run := func(parallel bool) *Result {
		t.Helper()

		config := NewBlockDiagonalConfig(12, 3)
		config.ObjectiveFunc = blockStructuredEllipsoid
		seed := int64(804)
		config.Seed = &seed
		config.LowerBound = -10
		config.UpperBound = 10
		config.InitialMean = filledVector(12, 2)
		config.InitialSigma = 1
		config.Convergence = nil
		config.MaxIterations = 100
		config.EnableParallel = parallel
		config.MaxWorkers = 3

		result, err := Optimize(config)
		if err != nil {
			t.Fatalf("Optimize(parallel=%t) = %v", parallel, err)
		}

		return result
	}

	serial := run(false)
	parallel := run(true)

	if !reflect.DeepEqual(serial, parallel) {
		t.Fatal("serial and parallel block results differ")
	}
}

func blockStructuredEllipsoid(position []float64) float64 {
	cost := 0.0

	for index := 0; index+1 < len(position); index += 2 {
		easy := (position[index] + position[index+1]) / math.Sqrt2
		hard := (position[index] - position[index+1]) / math.Sqrt2
		cost += easy*easy + 1e6*hard*hard
	}

	return cost
}

func BenchmarkBlockDiagonalN14000K7(b *testing.B) {
	for range b.N {
		config := NewBlockDiagonalConfig(14_000, 7)
		config.ObjectiveFunc = sphere
		config.Rand = rand.New(rand.NewSource(805))
		config.LowerBound = -10
		config.UpperBound = 10
		config.InitialMean = filledVector(14_000, 2)
		config.InitialSigma = 1
		config.Convergence = nil
		config.MaxIterations = 1

		_, err := Optimize(config)
		if err != nil {
			b.Fatal(err)
		}
	}
}
