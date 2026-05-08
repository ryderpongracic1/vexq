// Package catalog manages the mapping from table names to .vxq file paths and schemas.
// Phase 1: JSON config backend (mutation-free public API for easy upgrade later).
package catalog

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"

	"github.com/ryderpongracic1/vexq/storage"
)

// TableEntry is a registered table in the catalog.
type TableEntry struct {
	Name     string
	FilePath string
	Schema   storage.Schema
}

// Catalog maps table names to .vxq file entries.
type Catalog struct {
	mu     sync.RWMutex
	tables map[string]*TableEntry
}

// configFile is the JSON format for the on-disk catalog config.
type configFile struct {
	Tables []struct {
		Name string `json:"name"`
		Path string `json:"path"`
	} `json:"tables"`
}

// Open reads a JSON catalog config and returns a ready Catalog.
// If configPath is empty, returns an empty catalog.
func Open(ctx context.Context, configPath string) (*Catalog, error) {
	c := &Catalog{tables: make(map[string]*TableEntry)}
	if configPath == "" {
		return c, nil
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("catalog: open %q: %w", configPath, err)
	}
	var cfg configFile
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("catalog: parse %q: %w", configPath, err)
	}
	for _, t := range cfg.Tables {
		entry := &TableEntry{Name: t.Name, FilePath: t.Path}
		c.tables[t.Name] = entry
	}
	return c, nil
}

// OpenSingle returns a catalog with exactly one table registered from a .vxq file path.
func OpenSingle(ctx context.Context, name, path string) (*Catalog, error) {
	r, err := storage.Open(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("catalog: open %q: %w", path, err)
	}
	schema := r.Meta().Schema
	_ = r.Close()

	c := &Catalog{tables: make(map[string]*TableEntry)}
	c.tables[name] = &TableEntry{Name: name, FilePath: path, Schema: schema}
	return c, nil
}

// OpenMulti returns a catalog with multiple tables registered from .vxq file paths.
// tables maps table name → file path.
func OpenMulti(ctx context.Context, tables map[string]string) (*Catalog, error) {
	c := &Catalog{tables: make(map[string]*TableEntry)}
	for name, path := range tables {
		r, err := storage.Open(ctx, path)
		if err != nil {
			return nil, fmt.Errorf("catalog: open %q: %w", path, err)
		}
		schema := r.Meta().Schema
		_ = r.Close()
		c.tables[name] = &TableEntry{Name: name, FilePath: path, Schema: schema}
	}
	return c, nil
}

// Register adds or replaces a table entry.
func (c *Catalog) Register(name, path string, schema storage.Schema) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tables[name] = &TableEntry{Name: name, FilePath: path, Schema: schema}
}

// Lookup returns the TableEntry for name, loading its schema from the .vxq
// footer on first access.
func (c *Catalog) Lookup(ctx context.Context, name string) (TableEntry, bool) {
	c.mu.RLock()
	e, ok := c.tables[name]
	c.mu.RUnlock()
	if !ok {
		return TableEntry{}, false
	}
	// Lazy-load schema if not yet populated.
	if len(e.Schema.Fields) == 0 && e.FilePath != "" {
		if err := c.loadSchema(ctx, e); err != nil {
			return TableEntry{}, false
		}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return *e, true
}

func (c *Catalog) loadSchema(ctx context.Context, e *TableEntry) error {
	r, err := storage.Open(ctx, e.FilePath)
	if err != nil {
		return fmt.Errorf("catalog: load schema for %q: %w", e.Name, err)
	}
	schema := r.Meta().Schema
	_ = r.Close()

	c.mu.Lock()
	e.Schema = schema
	c.mu.Unlock()
	return nil
}
