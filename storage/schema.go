package storage

// Field describes a single column.
type Field struct {
	Name     string
	Type     DataType
	Encoding Encoding
	Nullable bool
}

// Schema is the ordered list of fields in a table.
type Schema struct {
	Fields []Field
}

// IndexOf returns the column index for name, or -1.
func (s Schema) IndexOf(name string) int {
	for i, f := range s.Fields {
		if f.Name == name {
			return i
		}
	}
	return -1
}

// ZoneMap holds per-row-group statistics for a column used in predicate pushdown.
type ZoneMap struct {
	NullCount int64
	// Sum is raw bits: int64 bits for INT64, float64 bits for FLOAT64, 0 otherwise.
	Sum       int64
	// Min and Max are raw bits: uint64 for INT64/FLOAT64/DATE (zero-extended),
	// dictionary code for STRING.
	Min, Max  uint64
	HasMinMax bool // false if the row group is entirely null
}

// ColumnSectionMeta describes one column's section within a row group.
type ColumnSectionMeta struct {
	SectionOffset int64
	SectionLength int64
	Stats         ZoneMap
	// DictOffset is relative to SectionOffset; 0 if not a dictionary column.
	DictOffset uint32
	DictLength uint32
}

// RowGroupMeta describes one row group in the file.
type RowGroupMeta struct {
	FileOffset int64
	NumRows    int
	Columns    []ColumnSectionMeta
}

// FileMeta is the complete parsed footer of a .vxq file.
type FileMeta struct {
	Schema    Schema
	RowGroups []RowGroupMeta
}
