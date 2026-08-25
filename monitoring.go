package cmaes

import (
	"context"
	"fmt"
	"log/slog"
)

const (
	eventOptimizationStarted   = "optimization_started"
	eventIterationCompleted    = "iteration_completed"
	eventOptimizationCompleted = "optimization_completed"
	eventOptimizationFailed    = "optimization_failed"
	eventObserverPanic         = "observer_panic"
)

// logSafely forwards one lifecycle event to a caller-supplied logger. A panic
// raised by Log is contained, for the same reason observer panics are: a
// faulty reporting sink must not destroy an in-progress run.
func logSafely(
	ctx context.Context,
	logger Logger,
	level slog.Level,
	message string,
	args ...any,
) {
	if logger == nil {
		return
	}

	defer func() { _ = recover() }()

	logger.Log(ctx, level, message, args...)
}

func logOptimizationStarted(ctx context.Context, logger Logger, config *Config) {
	logSafely(
		ctx,
		logger,
		slog.LevelInfo,
		"optimization started",
		"event", eventOptimizationStarted,
		"problem_size", config.ProblemSize,
		"max_iterations", config.MaxIterations,
		"max_evaluations", config.MaxEvaluations,
		"population", config.Lambda,
		"parallel", config.EnableParallel,
	)
}

func logIterationCompleted(
	ctx context.Context,
	logger Logger,
	iteration, evaluations int,
	best Best,
	sigma, condition float64,
) {
	// Debug level: one iteration event per generation would otherwise flood an
	// info-level sink with a line for every one of MaxIterations generations.
	logSafely(
		ctx,
		logger,
		slog.LevelDebug,
		"optimization iteration completed",
		"event", eventIterationCompleted,
		"iteration", iteration,
		"evaluations", evaluations,
		"best_cost", best.Cost,
		"constraint_violation", best.ConstraintViolation,
		"sigma", sigma,
		"condition_number", condition,
	)
}

func logOptimizationCompleted(ctx context.Context, logger Logger, result *Result) {
	logSafely(
		ctx,
		logger,
		slog.LevelInfo,
		"optimization completed",
		"event", eventOptimizationCompleted,
		"iterations", result.IterationCount,
		"evaluations", result.FuncEvalCount,
		"best_cost", result.GlobalBest.Cost,
		"constraint_violation", result.GlobalBest.ConstraintViolation,
		"termination_reason", result.TerminationReason,
	)
}

// logOptimizationFailed reports the terminal event of a run that ended with an
// error, so that every logged start is followed by exactly one logged end.
func logOptimizationFailed(ctx context.Context, logger Logger, runErr error) {
	logSafely(
		ctx,
		logger,
		slog.LevelError,
		"optimization failed",
		"event", eventOptimizationFailed,
		"error", runErr.Error(),
	)
}

func logObserverPanic(ctx context.Context, logger Logger, source string, recovered any) {
	logSafely(
		ctx,
		logger,
		slog.LevelError,
		"optimization observer panicked",
		"event", eventObserverPanic,
		"source", source,
		"panic", fmt.Sprint(recovered),
	)
}
