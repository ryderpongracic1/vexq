package storage

import (
	"errors"
	"fmt"
)

// Sentinel errors.
var (
	ErrChecksum       = errors.New("storage: checksum mismatch")
	ErrCorruptFooter  = errors.New("storage: corrupt footer")
	ErrBadMagic       = errors.New("storage: bad magic bytes")
	ErrTypeMismatch   = errors.New("storage: type mismatch")
	ErrColumnComplete = errors.New("storage: all columns for row group already written")
	ErrRowGroupOpen   = errors.New("storage: row group not started")
	ErrMissingColumns = errors.New("storage: not all columns written before EndRowGroup")
)

// wrap prefixes an error with "storage: op: " following distrikv convention.
func wrap(op string, err error) error {
	return fmt.Errorf("storage: %s: %w", op, err)
}
