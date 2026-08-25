//go:build !js || !wasm

package main

// The stub keeps the nested module buildable on native platforms so the
// repository gate covers both sides of the build tags.
func main() {}
