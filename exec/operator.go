// Package exec implements the vectorized execution engine for vexq.
// All operators process data in batches of up to BlockRows (1024) rows.
package exec

import (
	"context"

	"github.com/ryderpongracic1/vexq/storage"
)

// Schema and DataType are aliases so the rest of the exec package doesn't need
// to import storage directly.
type Schema = storage.Schema
type Field = storage.Field
type DataType = storage.DataType

const (
	TypeInt64   = storage.TypeInt64
	TypeFloat64 = storage.TypeFloat64
	TypeBool    = storage.TypeBool
	TypeString  = storage.TypeString
	TypeDate    = storage.TypeDate
)

// Operator is the pull-based interface implemented by all physical operators.
// Next returns (nil, nil) at EOF.  Callers must call Close even if Next
// returns an error.
type Operator interface {
	Next(ctx context.Context) (*Batch, error)
	Schema() Schema
	Close() error
}
