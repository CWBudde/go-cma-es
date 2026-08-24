package cmaes

import (
	"regexp"
	"testing"
)

// semver matches a version without a leading "v", optionally carrying a
// pre-release suffix. It is deliberately strict: the release recipe greps
// CHANGELOG.md for the exact string, and a version this pattern rejects would
// fail that grep at tag time instead of here.
var semver = regexp.MustCompile(`^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$`)

func TestVersionIsSemantic(t *testing.T) {
	if !semver.MatchString(Version) {
		t.Fatalf("Version = %q, want a semantic version without a leading v", Version)
	}
}
