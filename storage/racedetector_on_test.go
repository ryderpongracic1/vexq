//go:build race

package storage

// raceDetectorEnabled reports whether this test binary was built with -race.
// See the !race variant of this file for why the distinction matters.
const raceDetectorEnabled = true
