//go:build !race

package storage

// raceDetectorEnabled reports whether this test binary was built with -race.
//
// It exists because sync.Pool deliberately drops a randomly chosen quarter of
// Put calls when the race detector is on (see the race.Enabled branch in
// sync/pool.go), which is invisible to production builds but makes any exact
// pool-reuse assertion flaky under `go test -race`. Tests assert the strict
// property in a normal build and a correspondingly weaker one under -race,
// rather than weakening it everywhere.
const raceDetectorEnabled = false
