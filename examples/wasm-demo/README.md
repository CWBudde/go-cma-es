# Covariance Lab — the go-cma-es WebAssembly demo

Four pages run [github.com/CWBudde/go-cma-es](https://github.com/CWBudde/go-cma-es)
compiled to `js/wasm`, with plain JavaScript and canvas as the display layer:

- **`index.html` — Covariance Lab.** Replays sampled populations, the updated mean, and
  the 2σ ellipse reconstructed from `DistributionSnapshot` on any of ten landscapes:
  Rosenbrock, a rotated condition-10⁶ ellipsoid, Rastrigin, Himmelblau, Sphere, Ackley,
  Schwefel, Michalewicz, Zakharov, and the expanded Schaffer F6. Ackley and Schaffer are
  drawn on tighter-than-standard bounds so their central funnel and their rings survive
  the 150×150 sampling grid.
- **`compare.html` — Metric Test.** Active full-covariance CMA-ES beside a fixed isotropic
  weighted search, using the same seed, Gaussian draw order, λ, σ₀, and evaluation budget.
- **`charts.html` — Telemetry.** Global and per-generation best cost, σ, and covariance
  condition number under one replay cursor.
- **`restart.html` — Restart Map.** The public IPOP restart API doubles λ between runs,
  on any of the landscapes above, with each run's winner and the local optimum it landed
  in marked. It defaults to Schaffer F6, where a single run cannot reach the center at
  any σ or λ but the restart schedule usually can.

The organizing rule, inherited from the Mayfly and Dragonfly demos, is that **no
optimization logic lives in JavaScript**. Go evaluates the landscapes and performs the
searches. JavaScript owns the DOM, canvas, and animation clock. Large histories cross the
WASM boundary through reusable `Float32Array` buffers.

## Build and run

```bash
just run-wasm-demo
just build-wasm-demo
./scripts/build-wasm-demo.sh /tmp/cmaes-demo
```

An HTTP server is required because the pages fetch `cmaes.wasm`. The `just
run-wasm-demo` recipe serves the built `dist/` directory at <http://localhost:8090>.

`wasm_exec.js` is copied from the active Go toolchain at build time and is not vendored.
It must match the compiler that produced the WASM module.

## What it exercises

- `WithPopulationObserver` supplies every displayed sample and best-so-far position.
- `WithDistributionObserver` supplies the mean, σ, eigenvectors, eigenvalue roots, and
  condition number used to reconstruct the ellipse.
- Full and separable covariance modes make the cost of giving up rotation visible.
- Active covariance adaptation can be toggled without changing any other run setting.
- A seed replays bit-identically; browser execution disables parallel evaluation because
  Go's `js/wasm` runtime executes on one browser thread.
- The telemetry view reads `Result.ConvergenceCurve`, `IterationBestHistory`,
  `SigmaHistory`, and `ConditionNumberHistory` directly.
- The restart view calls `OptimizeWithRestarts` and reads its per-run records directly.

The contour image is rank-normalized. This preserves the geometry of low basins across
objectives with very different dynamic ranges; its colors communicate relative height,
not a common numerical cost scale.

## Layout

| File                                              | Role                                                        |
| ------------------------------------------------- | ----------------------------------------------------------- |
| `main.go` / `main_stub.go`                        | WASM export table and native-build stub                     |
| `bridge.go`                                       | Panic containment and tolerant request readers              |
| `marshal.go`                                      | Reusable JavaScript-owned typed-array buffers               |
| `landscape.go`                                    | The ten landscape specs and rank normalization              |
| `demo.go`                                         | CMA histories, isotropic control, and restart orchestration |
| `boot.js`                                         | Shared WASM loader, calls, sinks, and replay transport      |
| `render.js`                                       | Landscapes, populations, ellipses, charts, and basin map    |
| `app.js`, `compare.js`, `charts.js`, `restart.js` | Page controllers                                            |

Published from `main` by
[`.github/workflows/wasm-demo-pages.yml`](../../.github/workflows/wasm-demo-pages.yml).
