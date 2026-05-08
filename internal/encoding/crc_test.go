package encoding

import "testing"

func TestAppendAndVerify(t *testing.T) {
	payload := []byte("hello, vexq")
	b := AppendChecksum(payload)
	if len(b) != len(payload)+4 {
		t.Fatalf("expected len %d, got %d", len(payload)+4, len(b))
	}
	got, err := VerifyTrailing(b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(got) != string(payload) {
		t.Fatalf("payload mismatch")
	}
}

func TestVerifyTrailingCorrupt(t *testing.T) {
	payload := []byte("hello, vexq")
	b := AppendChecksum(payload)
	b[3] ^= 0xFF // flip a bit in the payload
	_, err := VerifyTrailing(b)
	if err != ErrChecksum {
		t.Fatalf("expected ErrChecksum, got %v", err)
	}
}

func TestVerifyTrailingTooShort(t *testing.T) {
	_, err := VerifyTrailing([]byte{0, 1, 2})
	if err != ErrChecksum {
		t.Fatalf("expected ErrChecksum for short input, got %v", err)
	}
}

func TestVerifyTrailingEmpty(t *testing.T) {
	b := AppendChecksum(nil)
	got, err := VerifyTrailing(b)
	if err != nil {
		t.Fatalf("empty payload should verify: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected empty payload, got %d bytes", len(got))
	}
}
