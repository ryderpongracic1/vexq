// Package goldentest provides end-to-end correctness tests for vexq by comparing
// engine query results against an independent naive reference evaluator. This
// package is intentionally internal — it exists solely for correctness verification
// and must not be imported by engine code.
package goldentest

import (
	"context"
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/ryderpongracic1/vexq/storage"
)

// Value represents a nullable value in our reference tables.
// For NULL, IsNull is true and the typed fields are zero-valued.
type Value struct {
	IsNull bool
	Int64  int64
	Float  float64
	Str    string
	Bool   bool
	Date   int32 // days since 1970-01-01
}

// Row is a slice of Values, one per column.
type Row []Value

// Table is the in-memory representation of a test dataset.
type Table struct {
	Name   string
	Schema storage.Schema
	Rows   []Row
}

// lowCardStrings are the dictionary-friendly values used for string columns.
var lowCardStrings = []string{
	"alpha", "beta", "gamma", "delta", "epsilon",
	"zeta", "eta", "theta", "iota", "kappa",
}

// DatasetConfig controls how a test dataset is generated.
type DatasetConfig struct {
	Seed     int64
	NumRows  int
	NullRate float64 // probability [0,1] that any cell is NULL
}

// DefaultConfig returns a reasonable default configuration.
func DefaultConfig() DatasetConfig {
	return DatasetConfig{
		Seed:     42,
		NumRows:  500,
		NullRate: 0.15,
	}
}

// GenerateDataset builds deterministic in-memory tables and writes them to .vxq
// files in the given directory. Returns the tables for reference evaluation and
// a map of table name → file path for catalog registration.
func GenerateDataset(t testing.TB, dir string, cfg DatasetConfig) ([]Table, map[string]string) {
	t.Helper()
	rng := rand.New(rand.NewSource(cfg.Seed))

	// Table "orders" — simulates an orders-like fact table.
	orders := generateOrdersTable(rng, cfg)
	// Table "items" — simulates a line-item table for join testing.
	items := generateItemsTable(rng, cfg, orders)

	tables := []Table{orders, items}
	paths := make(map[string]string)

	for _, tbl := range tables {
		path := filepath.Join(dir, tbl.Name+".vxq")
		writeTable(t, path, tbl)
		paths[tbl.Name] = path
	}

	return tables, paths
}

func generateOrdersTable(rng *rand.Rand, cfg DatasetConfig) Table {
	schema := storage.Schema{Fields: []storage.Field{
		{Name: "order_id", Type: storage.TypeInt64, Nullable: false},
		{Name: "customer_id", Type: storage.TypeInt64, Nullable: true},
		{Name: "amount", Type: storage.TypeFloat64, Nullable: true},
		{Name: "status", Type: storage.TypeString, Nullable: true},
		{Name: "is_express", Type: storage.TypeBool, Nullable: true},
		{Name: "order_date", Type: storage.TypeDate, Nullable: true},
	}}

	rows := make([]Row, cfg.NumRows)
	for i := range rows {
		row := make(Row, len(schema.Fields))
		// order_id: sequential, never null
		row[0] = Value{Int64: int64(i + 1)}
		// customer_id: 1-50, nullable
		row[1] = maybeNull(rng, cfg.NullRate, Value{Int64: int64(rng.Intn(50) + 1)})
		// amount: 0.01 to 9999.99, nullable
		row[2] = maybeNull(rng, cfg.NullRate, Value{Float: math.Round(rng.Float64()*999998+1) / 100.0})
		// status: low-cardinality string
		row[3] = maybeNull(rng, cfg.NullRate, Value{Str: lowCardStrings[rng.Intn(len(lowCardStrings))]})
		// is_express: bool
		row[4] = maybeNull(rng, cfg.NullRate, Value{Bool: rng.Intn(2) == 1})
		// order_date: days since epoch in range [15000, 20000] (roughly 2011-2024)
		row[5] = maybeNull(rng, cfg.NullRate, Value{Date: int32(rng.Intn(5000) + 15000)})
		rows[i] = row
	}
	return Table{Name: "orders", Schema: schema, Rows: rows}
}

func generateItemsTable(rng *rand.Rand, cfg DatasetConfig, orders Table) Table {
	schema := storage.Schema{Fields: []storage.Field{
		{Name: "item_id", Type: storage.TypeInt64, Nullable: false},
		{Name: "order_id", Type: storage.TypeInt64, Nullable: false},
		{Name: "quantity", Type: storage.TypeInt64, Nullable: true},
		{Name: "price", Type: storage.TypeFloat64, Nullable: true},
		{Name: "category", Type: storage.TypeString, Nullable: true},
	}}

	// Generate 2-4 items per order for a subset of orders (about 60% have items).
	var rows []Row
	itemID := int64(1)
	for _, orow := range orders.Rows {
		if rng.Float64() > 0.6 {
			continue
		}
		orderID := orow[0].Int64
		numItems := rng.Intn(3) + 2 // 2 to 4
		for j := 0; j < numItems; j++ {
			row := make(Row, len(schema.Fields))
			row[0] = Value{Int64: itemID}
			row[1] = Value{Int64: orderID}
			row[2] = maybeNull(rng, cfg.NullRate, Value{Int64: int64(rng.Intn(20) + 1)})
			row[3] = maybeNull(rng, cfg.NullRate, Value{Float: math.Round(rng.Float64()*9998+1) / 100.0})
			row[4] = maybeNull(rng, cfg.NullRate, Value{Str: lowCardStrings[rng.Intn(5)]}) // fewer categories
			rows = append(rows, row)
			itemID++
		}
	}
	return Table{Name: "items", Schema: schema, Rows: rows}
}

func maybeNull(rng *rand.Rand, rate float64, v Value) Value {
	if rng.Float64() < rate {
		return Value{IsNull: true}
	}
	return v
}

// writeTable writes an in-memory Table to a .vxq file at path.
func writeTable(t testing.TB, path string, tbl Table) {
	t.Helper()
	ctx := context.Background()

	w, err := storage.NewWriter(path, tbl.Schema)
	if err != nil {
		t.Fatalf("NewWriter(%s): %v", tbl.Name, err)
	}

	rows := tbl.Rows
	numCols := len(tbl.Schema.Fields)

	for start := 0; start < len(rows); start += storage.RowGroupRows {
		end := start + storage.RowGroupRows
		if end > len(rows) {
			end = len(rows)
		}
		chunk := rows[start:end]
		n := len(chunk)

		if err := w.BeginRowGroup(n); err != nil {
			t.Fatalf("BeginRowGroup: %v", err)
		}

		for col := 0; col < numCols; col++ {
			field := tbl.Schema.Fields[col]
			nullBitmap := make([]byte, (n+7)/8)
			// Initialize to all-null (0 = null in LSB-first bitmap where 1=valid).

			switch field.Type {
			case storage.TypeInt64:
				vals := make([]int64, n)
				for i, row := range chunk {
					if !row[col].IsNull {
						vals[i] = row[col].Int64
						storage.SetValidBit(nullBitmap, i)
					}
				}
				if err := w.AppendColumn(ctx, col, nullBitmap, vals); err != nil {
					t.Fatalf("AppendColumn(%s.%s): %v", tbl.Name, field.Name, err)
				}
			case storage.TypeFloat64:
				vals := make([]float64, n)
				for i, row := range chunk {
					if !row[col].IsNull {
						vals[i] = row[col].Float
						storage.SetValidBit(nullBitmap, i)
					}
				}
				if err := w.AppendColumn(ctx, col, nullBitmap, vals); err != nil {
					t.Fatalf("AppendColumn(%s.%s): %v", tbl.Name, field.Name, err)
				}
			case storage.TypeString:
				vals := make([]string, n)
				for i, row := range chunk {
					if !row[col].IsNull {
						vals[i] = row[col].Str
						storage.SetValidBit(nullBitmap, i)
					}
				}
				if err := w.AppendColumn(ctx, col, nullBitmap, vals); err != nil {
					t.Fatalf("AppendColumn(%s.%s): %v", tbl.Name, field.Name, err)
				}
			case storage.TypeBool:
				vals := make([]bool, n)
				for i, row := range chunk {
					if !row[col].IsNull {
						vals[i] = row[col].Bool
						storage.SetValidBit(nullBitmap, i)
					}
				}
				if err := w.AppendColumn(ctx, col, nullBitmap, vals); err != nil {
					t.Fatalf("AppendColumn(%s.%s): %v", tbl.Name, field.Name, err)
				}
			case storage.TypeDate:
				vals := make([]int32, n)
				for i, row := range chunk {
					if !row[col].IsNull {
						vals[i] = row[col].Date
						storage.SetValidBit(nullBitmap, i)
					}
				}
				if err := w.AppendColumn(ctx, col, nullBitmap, vals); err != nil {
					t.Fatalf("AppendColumn(%s.%s): %v", tbl.Name, field.Name, err)
				}
			}
		}

		if err := w.EndRowGroup(); err != nil {
			t.Fatalf("EndRowGroup: %v", err)
		}
	}

	if err := w.Finish(ctx); err != nil {
		t.Fatalf("Finish(%s): %v", tbl.Name, err)
	}
}
