package encoding

import "math"

func math64bits(f float64) uint64  { return math.Float64bits(f) }
func math64float(b uint64) float64 { return math.Float64frombits(b) }
