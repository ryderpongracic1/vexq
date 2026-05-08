package encoding

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
)

var crcTable = crc32.MakeTable(crc32.IEEE)

// ErrChecksum is returned when a CRC32 validation fails.
var ErrChecksum = errors.New("encoding: checksum mismatch")

// ChecksumIEEE returns the CRC32-IEEE checksum of b.
func ChecksumIEEE(b []byte) uint32 {
	return crc32.Checksum(b, crcTable)
}

// AppendChecksum appends the 4-byte little-endian CRC32-IEEE of b to b and
// returns the extended slice.
func AppendChecksum(b []byte) []byte {
	sum := ChecksumIEEE(b)
	return binary.LittleEndian.AppendUint32(b, sum)
}

// VerifyTrailing verifies that the last 4 bytes of b are the little-endian
// CRC32-IEEE of b[:len(b)-4].  It returns the payload (b without the trailing
// CRC) on success, or ErrChecksum on mismatch.
func VerifyTrailing(b []byte) ([]byte, error) {
	if len(b) < 4 {
		return nil, ErrChecksum
	}
	payload := b[:len(b)-4]
	want := binary.LittleEndian.Uint32(b[len(b)-4:])
	got := ChecksumIEEE(payload)
	if got != want {
		return nil, ErrChecksum
	}
	return payload, nil
}
