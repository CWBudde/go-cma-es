package cmaes

import (
	"context"
	"log/slog"
)

const (
	eventOptimizationStarted   = "optimization_started"
	eventIterationCompleted    = "iteration_completed"
	eventOptimizationCompleted = "optimization_completed"
)

func logOptimizationStarted(ctx context.Context, logger Logger, config *Config) {
	if logger == nil {
		return
	}

	logger.Log(
		ctx,
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
	if logger == nil {
		return
	}

	logger.Log(
		ctx,
		slog.LevelInfo,
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
	if logger == nil {
		return
	}

	logger.Log(
		ctx,
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
