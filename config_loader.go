// JSON configuration persistence for CMA-ES.

package cmaes

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// LoadConfig reads and validates a configuration written by SaveConfig.
// ObjectiveFunc, Rand, and constraint callbacks are not serializable and must
// be restored by the caller before the loaded configuration can be optimized.
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	config := &Config{}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()

	err = decoder.Decode(config)
	if err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	trailingErr := decoder.Decode(&struct{}{})
	if !errors.Is(trailingErr, io.EOF) {
		if trailingErr == nil {
			trailingErr = errors.New("multiple JSON values")
		}

		return nil, fmt.Errorf("failed to parse config file: trailing data: %w", trailingErr)
	}

	err = validateWithoutObjective(config)
	if err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	return config, nil
}

// SaveConfig writes config as indented JSON, creating or truncating path.
// Function fields and Rand are omitted because they cannot be serialized.
func SaveConfig(config *Config, path string) error {
	if config == nil {
		return errors.New("config must not be nil")
	}

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	err = os.WriteFile(filepath.Clean(path), data, 0o600)
	if err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

func validateWithoutObjective(config *Config) error {
	if config == nil {
		return errors.New("config must not be nil")
	}

	probe := *config
	if probe.ObjectiveFunc == nil {
		probe.ObjectiveFunc = placeholderObjective
	}

	return probe.Validate()
}

func placeholderObjective(_ []float64) float64 {
	return 0
}
