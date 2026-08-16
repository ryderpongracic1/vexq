package exec

import (
	"context"
	"strconv"
	"testing"

	"github.com/ryderpongracic1/vexq/storage"
)

// Tests for MIN/MAX over STRING columns.
//
// Before the fix these returned 0 for every input: the planner gave a string
// column's MIN/MAX a TypeInt64 accumulator, extractInt64 has no case for
// *StringVector and so returned 0 from its default arm, and 0 beats both the
// MaxInt64 seed a MIN starts from and the MinInt64 seed a MAX starts from. The
// string value never entered the accumulator at all, so dictionary codes were
// never compared either — which matters, because .vxq dictionaries assign codes
// in first-occurrence order (storage.DictBuilder.Add), so comparing codes would
// have been wrong too, and wrong in a way that can look plausible.
//
// Semantics asserted throughout: lexicographic byte order over the string values,
// NULLs excluded, NULL result when a group has no non-null input.

// strAggSchema mirrors aggRowsSchema but adds a second string column so a query
// can group by one string and aggregate over another, and an INT64 group-by
// column so the composite string-key path can be reached.
var strAggSchema = Schema{Fields: []Field{
	{Name: "grp", Type: TypeString},   // 0: dict-encoded group-by (drives intkey path)
	{Name: "mode", Type: TypeString},  // 1: dict-encoded aggregate input
	{Name: "region", Type: TypeInt64}, // 2: INT64 group-by (forces string-key path)
	{Name: "price", Type: TypeFloat64},
	{Name: "day", Type: TypeDate},
	{Name: "flagb", Type: TypeBool},
}}

// strRow is one input row for strAggBatch. An empty *Null field means present.
type strRow struct {
	grp      string
	mode     string
	modeNull bool
	region   int64
	price    float64
	day      int32
	flagb    bool
}

// strAggBatch builds one batch over strAggSchema. modeDictOrder fixes the order
// the batch's `mode` dictionary assigns codes in, so a test can make code order
// disagree with lexicographic order on purpose.
func strAggBatch(rows []strRow, modeDictOrder []string, sel SelectionVector) *Batch {
	n := len(rows)

	grpDict := storage.NewDictBuilder()
	for _, r := range rows {
		grpDict.Add(r.grp)
	}
	grpCodes := make([]uint32, n)
	for i, r := range rows {
		grpCodes[i], _ = grpDict.Lookup(r.grp)
	}
	grpRead, err := storage.UnmarshalDictionary(grpDict.Marshal())
	if err != nil {
		panic(err)
	}

	modeDict := storage.NewDictBuilder()
	for _, s := range modeDictOrder {
		modeDict.Add(s)
	}
	modeCodes := make([]uint32, n)
	modeNulls := make([]byte, (n+7)/8)
	for i, r := range rows {
		if r.modeNull {
			continue // leave invalid
		}
		c, ok := modeDict.Lookup(r.mode)
		if !ok {
			panic("mode value not in modeDictOrder: " + r.mode)
		}
		modeCodes[i] = c
		storage.SetValidBit(modeNulls, i)
	}
	modeRead, err := storage.UnmarshalDictionary(modeDict.Marshal())
	if err != nil {
		panic(err)
	}

	regions := make([]int64, n)
	prices := make([]float64, n)
	days := make([]int32, n)
	bools := &BoolVector{Bits: make([]byte, (n+7)/8), NullBitmap: storage.FullBitmap(n), Length: n}
	for i, r := range rows {
		regions[i] = r.region
		prices[i] = r.price
		days[i] = r.day
		bools.Set(i, r.flagb)
	}

	length := n
	if sel != nil {
		length = len(sel)
	}
	return &Batch{
		Schema: strAggSchema,
		Vectors: []Vector{
			&StringVector{Codes: grpCodes, Dict: grpRead, NullBitmap: storage.FullBitmap(n)},
			&StringVector{Codes: modeCodes, Dict: modeRead, NullBitmap: modeNulls},
			&Int64Vector{Values: regions, NullBitmap: storage.FullBitmap(n)},
			&Float64Vector{Values: prices, NullBitmap: storage.FullBitmap(n)},
			&DateVector{Values: days, NullBitmap: storage.FullBitmap(n)},
			bools,
		},
		Length: length,
		SelVec: sel,
	}
}

// aggCell is one output cell rendered so a test can assert on it regardless of
// which vector kind carried it.
type aggCell struct {
	isNull bool
	str    string
	num    float64
}

func (c aggCell) String() string {
	if c.isNull {
		return "NULL"
	}
	if c.str != "" {
		return c.str
	}
	return "num(" + fmtFloat(c.num) + ")"
}

func fmtFloat(f float64) string {
	if f == float64(int64(f)) {
		return strconv.FormatInt(int64(f), 10)
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}

// drainStrAgg runs a HashAggregate to completion and returns its rows as
// []aggCell, plus the concrete Go type name of each output vector so a test can
// assert that the emitted vector matches the type the schema declares.
func drainStrAgg(t *testing.T, batches []*Batch, groupBy []int, aggs []AggExpr) ([][]aggCell, []string, Schema) {
	t.Helper()
	child := &sliceBatchOp{batches: batches, schema: strAggSchema}
	ha, err := NewHashAggregate(child, groupBy, aggs)
	if err != nil {
		t.Fatalf("new hash aggregate: %v", err)
	}
	return drainToCells(t, ha)
}

func drainToCells(t *testing.T, op Operator) ([][]aggCell, []string, Schema) {
	t.Helper()
	schema := op.Schema()
	var rows [][]aggCell
	var vecTypes []string
	ctx := context.Background()
	for {
		batch, err := op.Next(ctx)
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if batch == nil {
			break
		}
		if vecTypes == nil {
			for _, v := range batch.Vectors {
				vecTypes = append(vecTypes, vectorKind(v))
			}
		}
		for r := 0; r < batch.Length; r++ {
			row := make([]aggCell, len(batch.Vectors))
			for c, v := range batch.Vectors {
				if v.IsNull(r) {
					row[c] = aggCell{isNull: true}
					continue
				}
				switch vv := v.(type) {
				case *StringVector:
					row[c] = aggCell{str: vv.Get(r)}
				case *Int64Vector:
					row[c] = aggCell{num: float64(vv.Values[r])}
				case *Float64Vector:
					row[c] = aggCell{num: vv.Values[r]}
				case *DateVector:
					row[c] = aggCell{num: float64(vv.Values[r])}
				case *BoolVector:
					n := 0.0
					if vv.Get(r) {
						n = 1
					}
					row[c] = aggCell{num: n}
				default:
					t.Fatalf("unexpected output vector %T", v)
				}
			}
			rows = append(rows, row)
		}
	}
	return rows, vecTypes, schema
}

func vectorKind(v Vector) string {
	switch v.(type) {
	case *StringVector:
		return "string"
	case *Int64Vector:
		return "int64"
	case *Float64Vector:
		return "float64"
	case *DateVector:
		return "date"
	case *BoolVector:
		return "bool"
	}
	return "unknown"
}

// declaredKind maps a schema DataType to the vectorKind a batch must carry for it.
func declaredKind(t DataType) string {
	switch t {
	case TypeString:
		return "string"
	case TypeFloat64:
		return "float64"
	case TypeDate:
		return "date"
	case TypeBool:
		return "bool"
	default:
		return "int64"
	}
}

// ---- Ungrouped (accumulateDirect) ------------------------------------------

func TestStringMinMaxUngrouped(t *testing.T) {
	// Dictionary code order is deliberately the reverse of lexicographic order:
	// "TRUCK" takes code 0 and "AIR" the highest code. An implementation that
	// compared dictionary codes would report MIN=TRUCK, MAX=AIR — exactly
	// backwards — so this batch distinguishes "compares values" from "compares
	// codes" as well as from the original "never compares anything".
	batch := strAggBatch([]strRow{
		{mode: "MAIL"},
		{mode: "TRUCK"},
		{mode: "AIR"},
		{mode: "SHIP"},
	}, []string{"TRUCK", "SHIP", "MAIL", "AIR"}, nil)

	rows, vecTypes, schema := drainStrAgg(t, []*Batch{batch}, nil, []AggExpr{
		{Kind: AggMin, ColIdx: 1, OutName: "min_mode"},
		{Kind: AggMax, ColIdx: 1, OutName: "max_mode"},
		{Kind: AggCount, ColIdx: 1, OutName: "cnt"},
	})

	if len(rows) != 1 {
		t.Fatalf("want 1 output row, got %d", len(rows))
	}
	if got := rows[0][0].str; got != "AIR" {
		t.Errorf("MIN(mode) = %q, want %q", got, "AIR")
	}
	if got := rows[0][1].str; got != "TRUCK" {
		t.Errorf("MAX(mode) = %q, want %q", got, "TRUCK")
	}
	if got := rows[0][2].num; got != 4 {
		t.Errorf("COUNT(mode) = %v, want 4 (COUNT must be unaffected)", got)
	}
	assertVectorTypesMatchSchema(t, schema, vecTypes)
}

func TestStringMinMaxUngroupedWithNulls(t *testing.T) {
	batch := strAggBatch([]strRow{
		{modeNull: true},
		{mode: "MAIL"},
		{modeNull: true},
		{mode: "AIR"},
		{modeNull: true},
	}, []string{"MAIL", "AIR"}, nil)

	rows, _, _ := drainStrAgg(t, []*Batch{batch}, nil, []AggExpr{
		{Kind: AggMin, ColIdx: 1, OutName: "min_mode"},
		{Kind: AggMax, ColIdx: 1, OutName: "max_mode"},
		{Kind: AggCount, ColIdx: 1, OutName: "cnt"},
	})
	if len(rows) != 1 {
		t.Fatalf("want 1 row, got %d", len(rows))
	}
	if rows[0][0].str != "AIR" || rows[0][1].str != "MAIL" {
		t.Errorf("MIN/MAX with NULLs = %v/%v, want AIR/MAIL", rows[0][0], rows[0][1])
	}
	if rows[0][2].num != 2 {
		t.Errorf("COUNT(mode) = %v, want 2 (NULLs excluded)", rows[0][2].num)
	}
}

// TestStringMinMaxAllNullUngrouped covers the group that saw rows but no value.
// MIN/MAX must be NULL, not "" — the zero value of the string accumulator.
func TestStringMinMaxAllNullUngrouped(t *testing.T) {
	batch := strAggBatch([]strRow{
		{modeNull: true},
		{modeNull: true},
	}, []string{"AIR"}, nil)

	rows, _, _ := drainStrAgg(t, []*Batch{batch}, nil, []AggExpr{
		{Kind: AggMin, ColIdx: 1, OutName: "min_mode"},
		{Kind: AggMax, ColIdx: 1, OutName: "max_mode"},
		{Kind: AggCount, ColIdx: 1, OutName: "cnt"},
	})
	if len(rows) != 1 {
		t.Fatalf("want 1 row, got %d", len(rows))
	}
	if !rows[0][0].isNull || !rows[0][1].isNull {
		t.Errorf("all-NULL input: MIN/MAX = %v/%v, want NULL/NULL", rows[0][0], rows[0][1])
	}
	if rows[0][2].num != 0 {
		t.Errorf("COUNT = %v, want 0", rows[0][2].num)
	}
}

// TestStringMinMaxEmptyInput covers buildEmptyGlobalResult: an ungrouped
// aggregate that consumed no rows at all must still emit one row of NULLs, in a
// vector whose kind matches the declared STRING type.
func TestStringMinMaxEmptyInput(t *testing.T) {
	rows, vecTypes, schema := drainStrAgg(t, nil, nil, []AggExpr{
		{Kind: AggMin, ColIdx: 1, OutName: "min_mode"},
		{Kind: AggMax, ColIdx: 1, OutName: "max_mode"},
		{Kind: AggCount, ColIdx: -1, OutName: "cnt_star"},
	})
	if len(rows) != 1 {
		t.Fatalf("empty input: want exactly 1 row, got %d", len(rows))
	}
	if !rows[0][0].isNull || !rows[0][1].isNull {
		t.Errorf("empty input: MIN/MAX = %v/%v, want NULL/NULL", rows[0][0], rows[0][1])
	}
	if rows[0][2].num != 0 {
		t.Errorf("empty input: COUNT(*) = %v, want 0", rows[0][2].num)
	}
	assertVectorTypesMatchSchema(t, schema, vecTypes)
}

// TestStringMinMaxEmptyStringIsData pins the one hazard of using "" as the
// accumulator's zero value: the empty string is a legitimate column value and
// must not be mistaken for "no value seen yet". Sorted lexicographically, ""
// precedes every other string, so MIN must return it.
func TestStringMinMaxEmptyStringIsData(t *testing.T) {
	batch := strAggBatch([]strRow{
		{mode: "MAIL"},
		{mode: ""},
		{mode: "AIR"},
	}, []string{"MAIL", "", "AIR"}, nil)

	rows, _, _ := drainStrAgg(t, []*Batch{batch}, nil, []AggExpr{
		{Kind: AggMin, ColIdx: 1, OutName: "min_mode"},
		{Kind: AggMax, ColIdx: 1, OutName: "max_mode"},
	})
	if len(rows) != 1 {
		t.Fatalf("want 1 row, got %d", len(rows))
	}
	if rows[0][0].isNull {
		t.Errorf("MIN over a column containing \"\" returned NULL; \"\" is a value, not an absence")
	}
	if rows[0][0].str != "" {
		t.Errorf("MIN = %q, want \"\" (the empty string sorts first)", rows[0][0].str)
	}
	if rows[0][1].str != "MAIL" {
		t.Errorf("MAX = %q, want MAIL", rows[0][1].str)
	}
}

// TestStringMinMaxOnlyEmptyString is the sharper form: the sole non-null value is
// "". The result must be "" and non-NULL, which is only distinguishable from the
// all-NULL case by the non-null counter rather than by the stored value.
func TestStringMinMaxOnlyEmptyString(t *testing.T) {
	batch := strAggBatch([]strRow{
		{mode: ""},
		{modeNull: true},
	}, []string{""}, nil)

	rows, _, _ := drainStrAgg(t, []*Batch{batch}, nil, []AggExpr{
		{Kind: AggMin, ColIdx: 1, OutName: "min_mode"},
		{Kind: AggMax, ColIdx: 1, OutName: "max_mode"},
	})
	if rows[0][0].isNull || rows[0][1].isNull {
		t.Fatalf("only-\"\" input: MIN/MAX = %v/%v, want \"\"/\"\" (not NULL)", rows[0][0], rows[0][1])
	}
	if rows[0][0].str != "" || rows[0][1].str != "" {
		t.Errorf("only-\"\" input: MIN/MAX = %q/%q, want \"\"/\"\"", rows[0][0].str, rows[0][1].str)
	}
}

// ---- Grouped: both key paths ----------------------------------------------

// TestStringMinMaxGroupedIntKeyPath groups by a dict-encoded string column,
// which is the shape canUseIntKey accepts, so accumulation runs through
// accumulateIntKey and its results are later materialized to string keys.
func TestStringMinMaxGroupedIntKeyPath(t *testing.T) {
	batch := strAggBatch([]strRow{
		{grp: "east", mode: "MAIL"},
		{grp: "west", mode: "TRUCK"},
		{grp: "east", mode: "AIR"},
		{grp: "west", mode: "SHIP"},
		{grp: "east", modeNull: true},
		{grp: "solo", mode: "RAIL"}, // single-value group
	}, []string{"TRUCK", "SHIP", "RAIL", "MAIL", "AIR"}, nil)

	rows, vecTypes, schema := drainStrAgg(t, []*Batch{batch}, []int{0}, []AggExpr{
		{Kind: AggMin, ColIdx: 1, OutName: "min_mode"},
		{Kind: AggMax, ColIdx: 1, OutName: "max_mode"},
		{Kind: AggCount, ColIdx: 1, OutName: "cnt"},
	})
	assertVectorTypesMatchSchema(t, schema, vecTypes)

	want := map[string][3]string{
		"east": {"AIR", "MAIL", "2"},
		"west": {"SHIP", "TRUCK", "2"},
		"solo": {"RAIL", "RAIL", "1"},
	}
	assertGroupedStringMinMax(t, rows, want)
}

// TestStringMinMaxGroupedStringKeyPath groups by an INT64 column, which
// canUseIntKey rejects, so accumulation runs through the composite string-key
// path (buildKey + the inline fold in accumulate).
func TestStringMinMaxGroupedStringKeyPath(t *testing.T) {
	batch := strAggBatch([]strRow{
		{region: 1, mode: "MAIL"},
		{region: 2, mode: "TRUCK"},
		{region: 1, mode: "AIR"},
		{region: 2, mode: "SHIP"},
		{region: 3, modeNull: true}, // group with rows but no value
	}, []string{"TRUCK", "SHIP", "MAIL", "AIR"}, nil)

	rows, vecTypes, schema := drainStrAgg(t, []*Batch{batch}, []int{2}, []AggExpr{
		{Kind: AggMin, ColIdx: 1, OutName: "min_mode"},
		{Kind: AggMax, ColIdx: 1, OutName: "max_mode"},
		{Kind: AggCount, ColIdx: 1, OutName: "cnt"},
	})
	assertVectorTypesMatchSchema(t, schema, vecTypes)

	byGroup := map[float64][]aggCell{}
	for _, r := range rows {
		byGroup[r[0].num] = r
	}
	if got := byGroup[1]; got[1].str != "AIR" || got[2].str != "MAIL" {
		t.Errorf("region 1: MIN/MAX = %v/%v, want AIR/MAIL", got[1], got[2])
	}
	if got := byGroup[2]; got[1].str != "SHIP" || got[2].str != "TRUCK" {
		t.Errorf("region 2: MIN/MAX = %v/%v, want SHIP/TRUCK", got[1], got[2])
	}
	if got := byGroup[3]; !got[1].isNull || !got[2].isNull {
		t.Errorf("region 3 (all NULL): MIN/MAX = %v/%v, want NULL/NULL", got[1], got[2])
	}
}

// TestStringMinMaxCrossBatchDictOrder makes two batches disagree about local
// dictionary codes for the same strings, the cross-row-group case the intkey
// remap tables exist for. A correct implementation compares values, so the
// per-batch code assignment cannot change the answer.
func TestStringMinMaxCrossBatchDictOrder(t *testing.T) {
	b1 := strAggBatch([]strRow{
		{grp: "east", mode: "MAIL"},
		{grp: "east", mode: "SHIP"},
	}, []string{"MAIL", "SHIP"}, nil)
	// Reversed code order, and the values that bracket the answer arrive here.
	b2 := strAggBatch([]strRow{
		{grp: "east", mode: "TRUCK"},
		{grp: "east", mode: "AIR"},
	}, []string{"TRUCK", "AIR"}, nil)

	rows, _, _ := drainStrAgg(t, []*Batch{b1, b2}, []int{0}, []AggExpr{
		{Kind: AggMin, ColIdx: 1, OutName: "min_mode"},
		{Kind: AggMax, ColIdx: 1, OutName: "max_mode"},
	})
	if len(rows) != 1 {
		t.Fatalf("want 1 group, got %d", len(rows))
	}
	if rows[0][1].str != "AIR" || rows[0][2].str != "TRUCK" {
		t.Errorf("cross-batch MIN/MAX = %v/%v, want AIR/TRUCK", rows[0][1], rows[0][2])
	}
}

// TestStringMinMaxSelVec runs both grouped paths over batches carrying a
// selection vector, so only the selected physical rows may contribute. The
// unselected rows hold values that would win the comparison if they leaked in.
func TestStringMinMaxSelVec(t *testing.T) {
	rows := []strRow{
		{grp: "east", region: 1, mode: "MAIL"},  // 0 selected
		{grp: "east", region: 1, mode: "AAAAA"}, // 1 NOT selected — would win MIN
		{grp: "east", region: 1, mode: "SHIP"},  // 2 selected
		{grp: "east", region: 1, mode: "ZZZZZ"}, // 3 NOT selected — would win MAX
	}
	dict := []string{"ZZZZZ", "SHIP", "MAIL", "AAAAA"}

	for _, tc := range []struct {
		name    string
		groupBy []int
	}{
		{"intkey", []int{0}},
		{"stringkey", []int{2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			batch := strAggBatch(rows, dict, SelectionVector{0, 2})
			out, _, _ := drainStrAgg(t, []*Batch{batch}, tc.groupBy, []AggExpr{
				{Kind: AggMin, ColIdx: 1, OutName: "min_mode"},
				{Kind: AggMax, ColIdx: 1, OutName: "max_mode"},
				{Kind: AggCount, ColIdx: 1, OutName: "cnt"},
			})
			if len(out) != 1 {
				t.Fatalf("want 1 group, got %d", len(out))
			}
			if out[0][1].str != "MAIL" || out[0][2].str != "SHIP" {
				t.Errorf("MIN/MAX under selection vector = %v/%v, want MAIL/SHIP", out[0][1], out[0][2])
			}
			if out[0][3].num != 2 {
				t.Errorf("COUNT under selection vector = %v, want 2", out[0][3].num)
			}
		})
	}
}

// TestStringMinMaxMixedWithNumericAggregates checks that putting a string
// MIN/MAX beside numeric aggregates in one operator leaves the numeric ones
// alone — the string accumulator lives in a side map, and its slot in the int64
// accumulator array must not be read or written as a number.
func TestStringMinMaxMixedWithNumericAggregates(t *testing.T) {
	batch := strAggBatch([]strRow{
		{grp: "east", mode: "MAIL", price: 10, day: 100, flagb: false},
		{grp: "east", mode: "AIR", price: 30, day: 300, flagb: true},
		{grp: "east", mode: "SHIP", price: 20, day: 200, flagb: true},
	}, []string{"MAIL", "AIR", "SHIP"}, nil)

	rows, vecTypes, schema := drainStrAgg(t, []*Batch{batch}, []int{0}, []AggExpr{
		{Kind: AggMin, ColIdx: 1, OutName: "min_mode"},
		{Kind: AggMax, ColIdx: 1, OutName: "max_mode"},
		{Kind: AggMin, ColIdx: 3, OutName: "min_price"},
		{Kind: AggMax, ColIdx: 3, OutName: "max_price"},
		{Kind: AggSum, ColIdx: 3, OutName: "sum_price"},
		{Kind: AggAvg, ColIdx: 3, OutName: "avg_price"},
		{Kind: AggMin, ColIdx: 4, OutName: "min_day"},
		{Kind: AggMax, ColIdx: 4, OutName: "max_day"},
		{Kind: AggMin, ColIdx: 5, OutName: "min_flag"},
		{Kind: AggMax, ColIdx: 5, OutName: "max_flag"},
		{Kind: AggCount, ColIdx: -1, OutName: "cnt_star"},
	})
	assertVectorTypesMatchSchema(t, schema, vecTypes)

	if len(rows) != 1 {
		t.Fatalf("want 1 group, got %d", len(rows))
	}
	r := rows[0]
	checks := []struct {
		name string
		got  aggCell
		want aggCell
	}{
		{"MIN(mode)", r[1], aggCell{str: "AIR"}},
		{"MAX(mode)", r[2], aggCell{str: "SHIP"}},
		{"MIN(price)", r[3], aggCell{num: 10}},
		{"MAX(price)", r[4], aggCell{num: 30}},
		{"SUM(price)", r[5], aggCell{num: 60}},
		{"AVG(price)", r[6], aggCell{num: 20}},
		{"MIN(day)", r[7], aggCell{num: 100}},
		{"MAX(day)", r[8], aggCell{num: 300}},
		{"MIN(flagb)", r[9], aggCell{num: 0}},
		{"MAX(flagb)", r[10], aggCell{num: 1}},
		{"COUNT(*)", r[11], aggCell{num: 3}},
	}
	for _, c := range checks {
		if c.got.isNull != c.want.isNull || c.got.str != c.want.str || c.got.num != c.want.num {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
}

// ---- Output vector type must match the declared schema type -----------------

// TestMinMaxOutputVectorMatchesDeclaredType is the regression test for the
// second half of the defect. NewHashAggregate declares MIN/MAX's output column
// with the input column's type, but buildOutputBatch used to emit an
// Int64Vector for every accumulator that was not float64. A STRING column
// therefore produced a batch whose schema said STRING while the vector was
// Int64Vector holding 0, and a DATE column produced raw day numbers under a DATE
// schema. Any consumer that type-asserts on the schema type — the golden test's
// drainOperator does — panics on that batch rather than reading a wrong value.
func TestMinMaxOutputVectorMatchesDeclaredType(t *testing.T) {
	cols := []struct {
		name   string
		colIdx int
	}{
		{"string", 1},
		{"int64", 2},
		{"float64", 3},
		{"date", 4},
		{"bool", 5},
	}
	for _, c := range cols {
		t.Run(c.name, func(t *testing.T) {
			batch := strAggBatch([]strRow{
				{grp: "g", mode: "MAIL", region: 7, price: 1.5, day: 42, flagb: true},
			}, []string{"MAIL"}, nil)
			_, vecTypes, schema := drainStrAgg(t, []*Batch{batch}, nil, []AggExpr{
				{Kind: AggMin, ColIdx: c.colIdx, OutName: "mn"},
				{Kind: AggMax, ColIdx: c.colIdx, OutName: "mx"},
			})
			assertVectorTypesMatchSchema(t, schema, vecTypes)
		})
	}
}

func assertVectorTypesMatchSchema(t *testing.T, schema Schema, vecTypes []string) {
	t.Helper()
	if vecTypes == nil {
		return // no batch emitted
	}
	if len(vecTypes) != len(schema.Fields) {
		t.Fatalf("vector count %d != schema field count %d", len(vecTypes), len(schema.Fields))
	}
	for i, f := range schema.Fields {
		if want := declaredKind(f.Type); vecTypes[i] != want {
			t.Errorf("column %q: schema declares %v (%s vector) but batch carries a %s vector",
				f.Name, f.Type, want, vecTypes[i])
		}
	}
}

func assertGroupedStringMinMax(t *testing.T, rows [][]aggCell, want map[string][3]string) {
	t.Helper()
	if len(rows) != len(want) {
		t.Fatalf("group count = %d, want %d (rows: %v)", len(rows), len(want), rows)
	}
	for _, r := range rows {
		grp := r[0].str
		exp, ok := want[grp]
		if !ok {
			t.Errorf("unexpected group %q", grp)
			continue
		}
		if r[1].str != exp[0] {
			t.Errorf("group %q: MIN = %v, want %s", grp, r[1], exp[0])
		}
		if r[2].str != exp[1] {
			t.Errorf("group %q: MAX = %v, want %s", grp, r[2], exp[1])
		}
		if got := fmtFloat(r[3].num); got != exp[2] {
			t.Errorf("group %q: COUNT = %s, want %s", grp, got, exp[2])
		}
	}
}
