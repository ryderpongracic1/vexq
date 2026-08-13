package planner

import (
	"context"
	"strings"
	"testing"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

// registerTwoTables sets up a catalog with two tables that share an "id" column.
func registerTwoTables(cat *catalog.Catalog) {
	cat.Register("a", "", storage.Schema{
		Fields: []storage.Field{
			{Name: "id", Type: storage.TypeInt64},
			{Name: "name", Type: storage.TypeString},
		},
	})
	cat.Register("b", "", storage.Schema{
		Fields: []storage.Field{
			{Name: "id", Type: storage.TypeInt64},
			{Name: "value", Type: storage.TypeInt64},
		},
	})
}

func TestQualifiedColumnInWhere(t *testing.T) {
	cat := mustNewCatalog()
	registerTwoTables(cat)

	p := sql.NewParser("SELECT a.id, b.value FROM a, b WHERE a.id = b.id")
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	plan, err := Build(context.Background(), stmt, cat)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	if plan == nil {
		t.Fatal("expected non-nil plan")
	}
	found := findJoin(plan)
	if found == nil {
		t.Fatal("expected LogicalJoin in plan for qualified join condition")
	}
}

func TestQualifiedColumnInWhereWithAlias(t *testing.T) {
	cat := mustNewCatalog()
	registerTwoTables(cat)

	p := sql.NewParser("SELECT x.id, y.value FROM a x, b y WHERE x.id = y.id")
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	plan, err := Build(context.Background(), stmt, cat)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	if plan == nil {
		t.Fatal("expected non-nil plan")
	}
	found := findJoin(plan)
	if found == nil {
		t.Fatal("expected LogicalJoin in plan for aliased qualified join condition")
	}
}

func TestAmbiguousColumnError(t *testing.T) {
	cat := mustNewCatalog()
	registerTwoTables(cat)

	p := sql.NewParser("SELECT id FROM a, b WHERE id = 1")
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	_, err = Build(context.Background(), stmt, cat)
	if err == nil {
		t.Fatal("expected error for ambiguous column reference")
	}
	if !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("expected 'ambiguous' in error, got: %v", err)
	}
}

func TestNonExistentQualifiedColumn(t *testing.T) {
	cat := mustNewCatalog()
	registerTwoTables(cat)

	p := sql.NewParser("SELECT a.id FROM a, b WHERE a.nonexistent = b.id")
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	_, err = Build(context.Background(), stmt, cat)
	if err == nil {
		t.Fatal("expected error for non-existent qualified column")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Fatalf("expected 'not found' in error, got: %v", err)
	}
}

func TestUnambiguousUnqualifiedStillWorks(t *testing.T) {
	cat := mustNewCatalog()
	registerTwoTables(cat)

	p := sql.NewParser("SELECT name, value FROM a, b WHERE a.id = b.id")
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	plan, err := Build(context.Background(), stmt, cat)
	if err != nil {
		t.Fatalf("Build: %v (expected success for unambiguous unqualified columns)", err)
	}
	if plan == nil {
		t.Fatal("expected non-nil plan")
	}
}

func TestUniqueColumnsBackwardCompatible(t *testing.T) {
	cat := mustNewCatalog()
	cat.Register("orders", "", storage.Schema{
		Fields: []storage.Field{
			{Name: "o_orderkey", Type: storage.TypeInt64},
			{Name: "o_custkey", Type: storage.TypeInt64},
		},
	})
	cat.Register("lineitem", "", storage.Schema{
		Fields: []storage.Field{
			{Name: "l_orderkey", Type: storage.TypeInt64},
			{Name: "l_quantity", Type: storage.TypeInt64},
		},
	})

	p := sql.NewParser("SELECT o_orderkey, l_quantity FROM orders, lineitem WHERE o_orderkey = l_orderkey")
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	plan, err := Build(context.Background(), stmt, cat)
	if err != nil {
		t.Fatalf("Build: %v (backward-compatible unique column names should work)", err)
	}
	if plan == nil {
		t.Fatal("expected non-nil plan")
	}
}

func TestNonExistentTableQualifier(t *testing.T) {
	cat := mustNewCatalog()
	registerTwoTables(cat)

	p := sql.NewParser("SELECT a.id FROM a, b WHERE c.id = b.id")
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	_, err = Build(context.Background(), stmt, cat)
	if err == nil {
		t.Fatal("expected error for non-existent table qualifier")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Fatalf("expected 'not found' in error, got: %v", err)
	}
}

// findJoin walks the plan tree and returns the first LogicalJoin, or nil.
func findJoin(n LogicalNode) *LogicalJoin {
	switch x := n.(type) {
	case *LogicalJoin:
		return x
	case *LogicalFilter:
		return findJoin(x.Child)
	case *LogicalProject:
		return findJoin(x.Child)
	case *LogicalAggregate:
		return findJoin(x.Child)
	case *LogicalSort:
		return findJoin(x.Child)
	case *LogicalLimit:
		return findJoin(x.Child)
	case *LogicalDistinct:
		return findJoin(x.Child)
	}
	return nil
}
