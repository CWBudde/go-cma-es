package cmaes

// Version is the released version of this library, without a leading "v".
//
// It exists so that a consumer can record which implementation produced a
// result and refuse to resume work that a different one started. That is not
// hypothetical: MayFlyCircleFit pins its optimizer versions in checkpoints and
// rejects a resume across a version that changed the search trajectory, because
// a seed no longer reproduces the run it originally described. Any release of
// this library that alters the update rules must bump this constant for that
// guard to work.
//
// It is updated by hand as part of the release checklist, and `just
// release-check` verifies that CHANGELOG.md carries a matching entry.
const Version = "0.0.0-dev"
