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

// configFileVersion is the schema version SaveConfig stamps into every file.
// Bump it whenever a Config field is renamed or removed, so an older file fails
// with a version message instead of an unknown-field parse error.
const configFileVersion = 1

// configFile is the persisted shape: the Config fields inline, plus the schema
// guard. The version lives here rather than on Config so it cannot be mistaken
// for a run parameter. A file without the key predates the guard and is read as
// the current version.
type configFile struct {
	*Config

	FormatVersion int `json:"format_version"`
}

// LoadConfig reads and validates a configuration written by SaveConfig.
// ObjectiveFunc, Rand, and constraint callbacks are not serializable and must
// be restored by the caller before the loaded configuration can be optimized.
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	config := &Config{}
	file := configFile{Config: config}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()

	err = decoder.Decode(&file)
	if err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Zero means the key was absent, which predates the guard.
	if file.FormatVersion < 0 || file.FormatVersion > configFileVersion {
		return nil, fmt.Errorf("unsupported config format_version %d (this build writes %d)",
			file.FormatVersion, configFileVersion)
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
// Function fields and Rand are omitted because they cannot be serialized. The
// file carries a format_version key that LoadConfig checks.
func SaveConfig(config *Config, path string) error {
	if config == nil {
		return errors.New("config must not be nil")
	}

	data, err := json.MarshalIndent(
		configFile{Config: config, FormatVersion: configFileVersion}, "", "  ")
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

	// Constraint callbacks are as unserializable as ObjectiveFunc, so a saved
	// file can never carry them and refusing to load one would make every
	// constrained configuration unloadable. Loading is therefore allowed while
	// the configuration stays invalid to run: no placeholder is substituted, the
	// slices stay empty, and the caller's own Validate fails with this same
	// sentinel until real functions are reattached.
	err := probe.Validate()
	if err != nil && !errors.Is(err, errConstraintFunctionsMissing) {
		return err
	}

	return nil
}

func placeholderObjective(_ []float64) float64 {
	return 0
}
