//go:build js && wasm

package main

import (
	"fmt"
	"math"
	"syscall/js"
)

func guard(name string, fn func(js.Value) any) js.Func {
	return js.FuncOf(func(_ js.Value, args []js.Value) (result any) {
		defer func() {
			if recovered := recover(); recovered != nil {
				result = js.ValueOf(map[string]any{
					"error": fmt.Sprintf("%s: %v", name, recovered),
					"panic": true,
				})
			}
		}()

		opts := js.Undefined()
		if len(args) > 0 {
			opts = args[0]
		}

		return fn(opts)
	})
}

func errorResult(format string, args ...any) map[string]any {
	return map[string]any{"error": fmt.Sprintf(format, args...), "panic": false}
}

func isObject(value js.Value) bool {
	return value.Type() == js.TypeObject && !value.IsNull()
}

func readInt(opts js.Value, key string, fallback int) int {
	if !isObject(opts) {
		return fallback
	}

	value := opts.Get(key)
	if value.Type() != js.TypeNumber {
		return fallback
	}

	number := value.Float()
	if math.IsNaN(number) || math.IsInf(number, 0) {
		return fallback
	}

	return int(number)
}

func readFloat(opts js.Value, key string, fallback float64) float64 {
	if !isObject(opts) {
		return fallback
	}

	value := opts.Get(key)
	if value.Type() != js.TypeNumber {
		return fallback
	}

	number := value.Float()
	if math.IsNaN(number) || math.IsInf(number, 0) {
		return fallback
	}

	return number
}

func readString(opts js.Value, key, fallback string) string {
	if !isObject(opts) {
		return fallback
	}

	value := opts.Get(key)
	if value.Type() != js.TypeString {
		return fallback
	}

	return value.String()
}

func readBool(opts js.Value, key string, fallback bool) bool {
	if !isObject(opts) {
		return fallback
	}

	value := opts.Get(key)
	if value.Type() != js.TypeBoolean {
		return fallback
	}

	return value.Bool()
}

func clampInt(value, low, high int) int {
	return min(high, max(low, value))
}

func finiteNumber(value float64) any {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return nil
	}

	return value
}
