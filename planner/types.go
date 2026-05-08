package planner

// Type aliases so planner sub-files don't need to re-import exec.
import (
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/storage"
)

type Schema = exec.Schema
type DataType = exec.DataType
type Field = storage.Field

const (
	TypeInt64   = exec.TypeInt64
	TypeFloat64 = exec.TypeFloat64
	TypeBool    = exec.TypeBool
	TypeString  = exec.TypeString
	TypeDate    = exec.TypeDate
)
