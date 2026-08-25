//go:build js && wasm

// Command wasm-demo is the browser showcase for github.com/CWBudde/go-cma-es.
// Optimization and objective evaluation stay in Go; JavaScript only renders
// the histories returned by this package.
package main

import "syscall/js"

var exports = map[string]func(js.Value) any{
	"info":      jsInfo,
	"landscape": jsLandscape,
	"run":       jsRun,
	"compare":   jsCompare,
	"restart":   jsRestart,
}

// live prevents the exported js.Func values from being garbage-collected.
var live []js.Func

func main() {
	namespace := js.Global().Get("Object").New()

	for name, fn := range exports {
		wrapped := guard(name, fn)
		live = append(live, wrapped)
		namespace.Set(name, wrapped)
	}

	js.Global().Set("cmaes", namespace)
	select {}
}
