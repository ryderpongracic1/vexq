package storage

// Magic bytes written at the file header and footer trailer.
const Magic = "VXQ1"

const (
	// BlockRows is the number of rows per column block (one vector worth).
	BlockRows = 1024
	// RowGroupRows is the maximum number of rows in a row group.
	RowGroupRows = 65536
	// NullBitmapBytes is the size of the per-block null bitmap in bytes.
	NullBitmapBytes = (BlockRows + 7) / 8 // 128
	// FooterTrailerSize: 4B CRC + 8B length + 4B magic.
	FooterTrailerSize = 16
)

// DataType identifies the logical type of a column.
type DataType uint8

const (
	TypeInt64   DataType = 0x01
	TypeFloat64 DataType = 0x02
	TypeBool    DataType = 0x03
	TypeString  DataType = 0x04
	TypeDate    DataType = 0x05 // int32 days since 1970-01-01
)

func (t DataType) String() string {
	switch t {
	case TypeInt64:
		return "INT64"
	case TypeFloat64:
		return "FLOAT64"
	case TypeBool:
		return "BOOL"
	case TypeString:
		return "STRING"
	case TypeDate:
		return "DATE"
	default:
		return "UNKNOWN"
	}
}

// ValueSize returns bytes per value for fixed-width types, 0 for variable.
func (t DataType) ValueSize() int {
	switch t {
	case TypeInt64, TypeFloat64:
		return 8
	case TypeDate:
		return 4
	case TypeString:
		return 4 // dict code (uint32)
	default:
		return 0
	}
}

// Encoding is the physical encoding applied to a column's data.
type Encoding uint8

const (
	EncPlain Encoding = 0 // raw values, no transformation
	EncRLE   Encoding = 1 // run-length encoding (Bool only in v1)
	EncDict  Encoding = 2 // dictionary encoding (String only in v1)
)
