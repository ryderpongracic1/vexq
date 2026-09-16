package storage

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// ComputeColumnStats recomputes the zone map of column col in row group rg from
// the values stored in the file, using the same rules Writer applies when it
// writes the row group. It also checks what the zone map cannot express but a
// reader relies on: that the column holds exactly the row group's row count, and
// that every non-null STRING code indexes the row group's dictionary.
//
// Zone maps are trusted by the query planner — a row group whose min/max rule
// out a predicate is never read — so a zone map that disagrees with the data
// makes queries silently miss rows. CRCs cannot catch that: the footer's own
// checksum covers a wrong zone map as readily as a right one.
func (r *Reader) ComputeColumnStats(ctx context.Context, rg, col int) (ZoneMap, error) {
	cr, err := r.OpenColumn(ctx, rg, col)
	if err != nil {
		return ZoneMap{}, err
	}
	defer cr.Close()

	field := r.meta.Schema.Fields[col]
	var dictLen int
	if field.Type == TypeString {
		dict, err := cr.Dictionary()
		if err != nil {
			return ZoneMap{}, fmt.Errorf("dictionary: %w", err)
		}
		dictLen = dict.Len()
	}

	var (
		stats      ZoneMap
		sumF       float64
		minF, maxF float64
		rows       int
	)
	observe := func(isNull bool, update func()) {
		if isNull {
			stats.NullCount++
			return
		}
		update()
	}

	for {
		nulls, payload, n, err := cr.NextBlock(ctx)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return ZoneMap{}, fmt.Errorf("block at row %d: %w", rows, err)
		}
		blockStart := rows
		rows += n

		switch field.Type {
		case TypeInt64:
			for i := 0; i < n; i++ {
				v := int64(binary.LittleEndian.Uint64(payload[i*8:]))
				observe(IsNullBit(nulls, i), func() {
					stats.Sum += v
					if !stats.HasMinMax {
						stats.Min, stats.Max, stats.HasMinMax = uint64(v), uint64(v), true
						return
					}
					if v < int64(stats.Min) {
						stats.Min = uint64(v)
					}
					if v > int64(stats.Max) {
						stats.Max = uint64(v)
					}
				})
			}
		case TypeFloat64:
			for i := 0; i < n; i++ {
				v := math.Float64frombits(binary.LittleEndian.Uint64(payload[i*8:]))
				observe(IsNullBit(nulls, i), func() {
					sumF += v
					if !stats.HasMinMax {
						minF, maxF, stats.HasMinMax = v, v, true
						return
					}
					if v < minF {
						minF = v
					}
					if v > maxF {
						maxF = v
					}
				})
			}
		case TypeDate:
			for i := 0; i < n; i++ {
				v := int32(binary.LittleEndian.Uint32(payload[i*4:]))
				observe(IsNullBit(nulls, i), func() {
					uv := uint64(uint32(v))
					if !stats.HasMinMax {
						stats.Min, stats.Max, stats.HasMinMax = uv, uv, true
						return
					}
					if int32(stats.Min) > v {
						stats.Min = uv
					}
					if int32(stats.Max) < v {
						stats.Max = uv
					}
				})
			}
		case TypeString:
			for i := 0; i < n; i++ {
				code := binary.LittleEndian.Uint32(payload[i*4:])
				if IsNullBit(nulls, i) {
					stats.NullCount++
					continue
				}
				if int(code) >= dictLen {
					return ZoneMap{}, fmt.Errorf("row %d: dictionary code %d out of range (dictionary has %d entries)", blockStart+i, code, dictLen)
				}
				stats.HasMinMax = true
			}
		case TypeBool:
			_, boolNulls, decoded, err := DecodeRLEBool(payload)
			if err != nil {
				return ZoneMap{}, fmt.Errorf("block at row %d: %w", blockStart, err)
			}
			if decoded != n {
				return ZoneMap{}, fmt.Errorf("block at row %d: RLE decodes to %d rows, want %d", blockStart, decoded, n)
			}
			for i := 0; i < n; i++ {
				if IsNullBit(boolNulls, i) {
					stats.NullCount++
				}
			}
		default:
			return ZoneMap{}, fmt.Errorf("unsupported type %v", field.Type)
		}
	}

	if rows != r.meta.RowGroups[rg].NumRows {
		return ZoneMap{}, fmt.Errorf("column holds %d rows, row group declares %d", rows, r.meta.RowGroups[rg].NumRows)
	}
	switch field.Type {
	case TypeFloat64:
		stats.Sum = int64(math.Float64bits(sumF))
		if stats.HasMinMax {
			stats.Min, stats.Max = math.Float64bits(minF), math.Float64bits(maxF)
		}
	case TypeString:
		// The writer's STRING zone map spans the dictionary's code range.
		if dictLen > 0 {
			stats.HasMinMax = true
			stats.Min, stats.Max = 0, uint64(dictLen-1)
		}
	}
	return stats, nil
}

// VerifyColumnStats reports an error when the zone map stored in the footer for
// column col of row group rg differs from the one recomputed from the column's
// data (see ComputeColumnStats).
func (r *Reader) VerifyColumnStats(ctx context.Context, rg, col int) error {
	got, err := r.ComputeColumnStats(ctx, rg, col)
	if err != nil {
		return err
	}
	want := r.meta.RowGroups[rg].Columns[col].Stats
	if got == want {
		return nil
	}
	t := r.meta.Schema.Fields[col].Type
	return fmt.Errorf("zone map does not match data: footer has %s, data has %s", formatZoneMap(want, t), formatZoneMap(got, t))
}

func formatZoneMap(z ZoneMap, t DataType) string {
	val := func(raw uint64) string {
		switch t {
		case TypeInt64:
			return fmt.Sprint(int64(raw))
		case TypeFloat64:
			return fmt.Sprint(math.Float64frombits(raw))
		case TypeDate:
			return fmt.Sprint(int32(uint32(raw)))
		}
		return fmt.Sprint(raw)
	}
	s := fmt.Sprintf("nulls=%d", z.NullCount)
	if z.HasMinMax {
		s += fmt.Sprintf(" min=%s max=%s", val(z.Min), val(z.Max))
	} else {
		s += " no min/max"
	}
	switch t {
	case TypeInt64:
		s += fmt.Sprintf(" sum=%d", z.Sum)
	case TypeFloat64:
		s += fmt.Sprintf(" sum=%v", math.Float64frombits(uint64(z.Sum)))
	}
	return s
}
