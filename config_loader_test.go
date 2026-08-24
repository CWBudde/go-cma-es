package cmaes

import (
	"bytes"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func loaderTestConfig() *Config {
	config := testConfig(5)
	seed := int64(73)
	target := -0.5
	config.Seed = &seed
	config.InitialMean = []float64{1, 2, 3, 4, 5}
	config.InitialSigma = 0.75
	config.Lambda = 12
	config.Mu = 6
	config.MaxIterations = 123
	config.MaxEvaluations = 456
	config.MaxWorkers = 3
	config.BoundaryMethod = BoundaryReflect
	config.CovarianceMode = CovarianceSeparable
	config.EnableParallel = true
	config.Convergence = &ConvergenceConfig{
		TargetCost:           &target,
		MinImprovement:       0.001,
		StagnationIterations: 17,
		MinIterations:        4,
	}
	config.Constraints = &ConstraintConfig{
		Handling:          ConstraintHandlingPenalty,
		PenaltyMethod:     PenaltyQuadratic,
		Inequalities:      []ConstraintFunction{func([]float64) float64 { return 0 }},
		Equalities:        []ConstraintFunction{func([]float64) float64 { return 0 }},
		PenaltyFactor:     9,
		EqualityTolerance: 1e-7,
	}

	return config
}

func TestSaveAndLoadConfigRoundTrip(t *testing.T) {
	original := loaderTestConfig()
	path := filepath.Join(t.TempDir(), "config.json")

	err := SaveConfig(original, path)
	if err != nil {
		t.Fatalf("SaveConfig() = %v", err)
	}

	loaded, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig() = %v", err)
	}

	wantJSON, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("json.Marshal(original) = %v", err)
	}

	gotJSON, err := json.Marshal(loaded)
	if err != nil {
		t.Fatalf("json.Marshal(loaded) = %v", err)
	}

	if !bytes.Equal(gotJSON, wantJSON) {
		t.Errorf("round-trip JSON differs:\n got: %s\nwant: %s", gotJSON, wantJSON)
	}

	if loaded.ObjectiveFunc != nil || loaded.Rand != nil {
		t.Error("loaded config retained an unserializable ObjectiveFunc or Rand")
	}

	if loaded.Constraints == nil {
		t.Fatal("loaded Constraints = nil")
	}

	if loaded.Constraints.Inequalities != nil || loaded.Constraints.Equalities != nil {
		t.Error("loaded config retained unserializable constraint callbacks")
	}

	err = loaded.Validate()
	if err == nil || !strings.Contains(err.Error(), "ObjectiveFunc") {
		t.Fatalf("loaded.Validate() = %v, want missing ObjectiveFunc error", err)
	}

	loaded.ObjectiveFunc = original.ObjectiveFunc

	err = loaded.Validate()
	if err != nil {
		t.Fatalf("loaded.Validate() after restoring objective = %v", err)
	}
}

func TestSaveConfigUsesSnakeCaseAndOmitsFunctions(t *testing.T) {
	config := loaderTestConfig()
	config.Rand = rand.New(rand.NewSource(1))
	config.Seed = nil
	path := filepath.Join(t.TempDir(), "config.json")

	err := SaveConfig(config, path)
	if err != nil {
		t.Fatalf("SaveConfig() = %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() = %v", err)
	}

	raw := map[string]json.RawMessage{}

	err = json.Unmarshal(data, &raw)
	if err != nil {
		t.Fatalf("saved file is not a JSON object: %v", err)
	}

	for _, key := range []string{"initial_mean", "initial_sigma", "problem_size", "active_cma"} {
		if _, found := raw[key]; !found {
			t.Errorf("saved config is missing key %q", key)
		}
	}

	for _, key := range []string{"ObjectiveFunc", "objective_func", "Rand", "rand"} {
		if _, found := raw[key]; found {
			t.Errorf("saved config contains unserializable key %q", key)
		}
	}
}

func TestLoadConfigRejectsBadInput(t *testing.T) {
	tests := []struct {
		name string
		data string
		want string
	}{
		{"malformed", "{not json", "parse config file"},
		{"unknown field", `{"unknown": true}`, "unknown field"},
		{"trailing value", `{}` + "\n" + `{}`, "trailing data"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.json")

			writeErr := os.WriteFile(path, []byte(test.data), 0o600)
			if writeErr != nil {
				t.Fatalf("WriteFile() = %v", writeErr)
			}

			_, err := LoadConfig(path)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("LoadConfig() = %v, want error containing %q", err, test.want)
			}
		})
	}
}

func TestLoadConfigRejectsInvalidContents(t *testing.T) {
	config := loaderTestConfig()
	config.Lambda = 1
	path := filepath.Join(t.TempDir(), "invalid.json")

	saveErr := SaveConfig(config, path)
	if saveErr != nil {
		t.Fatalf("SaveConfig() = %v", saveErr)
	}

	_, err := LoadConfig(path)
	if err == nil || !strings.Contains(err.Error(), "invalid config") {
		t.Fatalf("LoadConfig() = %v, want invalid config error", err)
	}
}

func TestLoadConfigMissingFile(t *testing.T) {
	_, err := LoadConfig(filepath.Join(t.TempDir(), "missing.json"))
	if err == nil || !strings.Contains(err.Error(), "read config file") {
		t.Fatalf("LoadConfig() = %v, want read error", err)
	}
}

func TestSaveConfigFailures(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")

	err := SaveConfig(nil, path)
	if err == nil {
		t.Fatal("SaveConfig(nil) = nil, want an error")
	}

	config := loaderTestConfig()

	config.InitialSigma = math.NaN()

	err = SaveConfig(config, path)
	if err == nil || !strings.Contains(err.Error(), "marshal") {
		t.Fatalf("SaveConfig(NaN) = %v, want marshal error", err)
	}

	config.InitialSigma = defaultInitialSigma

	err = SaveConfig(config, t.TempDir())
	if err == nil || !strings.Contains(err.Error(), "write") {
		t.Fatalf("SaveConfig(directory) = %v, want write error", err)
	}
}

func TestValidateWithoutObjectiveRejectsNil(t *testing.T) {
	err := validateWithoutObjective(nil)
	if err == nil {
		t.Fatal("validateWithoutObjective(nil) = nil, want an error")
	}
}

func TestPlaceholderObjective(t *testing.T) {
	if got := placeholderObjective([]float64{1, 2}); got != 0 {
		t.Errorf("placeholderObjective() = %v, want 0", got)
	}
}
