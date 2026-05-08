package encoding

import "encoding/binary"

func AppendUint16(b []byte, v uint16) []byte {
	return binary.LittleEndian.AppendUint16(b, v)
}

func AppendUint32(b []byte, v uint32) []byte {
	return binary.LittleEndian.AppendUint32(b, v)
}

func AppendInt32(b []byte, v int32) []byte {
	return binary.LittleEndian.AppendUint32(b, uint32(v))
}

func AppendUint64(b []byte, v uint64) []byte {
	return binary.LittleEndian.AppendUint64(b, v)
}

func AppendInt64(b []byte, v int64) []byte {
	return binary.LittleEndian.AppendUint64(b, uint64(v))
}

func AppendFloat64(b []byte, v float64) []byte {
	return binary.LittleEndian.AppendUint64(b, math64bits(v))
}

func AppendBytes(b, v []byte) []byte {
	return append(b, v...)
}

func ReadUint16(b []byte) (uint16, []byte) {
	return binary.LittleEndian.Uint16(b), b[2:]
}

func ReadUint32(b []byte) (uint32, []byte) {
	return binary.LittleEndian.Uint32(b), b[4:]
}

func ReadInt32(b []byte) (int32, []byte) {
	return int32(binary.LittleEndian.Uint32(b)), b[4:]
}

func ReadUint64(b []byte) (uint64, []byte) {
	return binary.LittleEndian.Uint64(b), b[8:]
}

func ReadInt64(b []byte) (int64, []byte) {
	return int64(binary.LittleEndian.Uint64(b)), b[8:]
}

func ReadFloat64(b []byte) (float64, []byte) {
	bits := binary.LittleEndian.Uint64(b)
	return math64float(bits), b[8:]
}

func PutUint32(b []byte, v uint32) { binary.LittleEndian.PutUint32(b, v) }
func PutUint64(b []byte, v uint64) { binary.LittleEndian.PutUint64(b, v) }
func PutInt64(b []byte, v int64)   { binary.LittleEndian.PutUint64(b, uint64(v)) }

func GetUint32(b []byte) uint32 { return binary.LittleEndian.Uint32(b) }
func GetUint64(b []byte) uint64 { return binary.LittleEndian.Uint64(b) }
func GetInt64(b []byte) int64   { return int64(binary.LittleEndian.Uint64(b)) }
