package planner

import (
	"context"
	"testing"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

// mustNewCatalog creates an empty catalog for plan-shape testing.
func mustNewCatalog() *catalog.Catalog {
	cat, _ := catalog.Open(context.Background(), "")
	return cat
}

func TestBuildHavingProducesFilterAboveAggregate(t *testing.T) {
	cat := mustNewCatalog()
	cat.Register("t", "", storage.Schema{
		Fields: []storage.Field{
			{Name: "a", Type: storage.TypeString},
			{Name: "b", Type: storage.TypeInt64},
		},
	})

	p := sql.NewParser("SELECT a, COUNT(*) AS cnt FROM t GROUP BY a HAVING cnt > 5")
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	plan, err := Build(context.Background(), stmt, cat)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	// The plan should have a Filter (for HAVING) wrapping an Aggregate.
	// Expected shape:
	//   LogicalFilter (HAVING)
	//     └── LogicalAggregate
	//           └── LogicalScan
	filter, ok := plan.(*LogicalFilter)
	if !ok {
		t.Fatalf("expected root to be *LogicalFilter (HAVING), got %T", plan)
	}

	_, ok = filter.Child.(*LogicalAggregate)
	if !ok {
		t.Fatalf("expected Filter child to be *LogicalAggregate, got %T", filter.Child)
	}

	// The filter predicate should be a BinaryExpr with >.
	bin, ok := filter.Predicate.(*sql.BinaryExpr)
	if !ok {
		t.Fatalf("expected BinaryExpr predicate, got %T", filter.Predicate)
	}
	if bin.Op != sql.OpGT {
		t.Fatalf("expected > operator in HAVING predicate, got %s", bin.Op)
	}
}

func TestBuildHavingWithWhereKeepsSeparate(t *testing.T) {
	cat := mustNewCatalog()
	cat.Register("t", "", storage.Schema{
		Fields: []storage.Field{
			{Name: "a", Type: storage.TypeString},
			{Name: "b", Type: storage.TypeInt64},
		},
	})

	p := sql.NewParser("SELECT a, SUM(b) AS total FROM t WHERE b > 0 GROUP BY a HAVING total > 100")
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	plan, err := Build(context.Background(), stmt, cat)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Expected shape:
	//   LogicalFilter (HAVING total > 100)
	//     └── LogicalAggregate
	//           └── LogicalFilter (WHERE b > 0)
	//                 └── LogicalScan
	havingFilter, ok := plan.(*LogicalFilter)
	if !ok {
		t.Fatalf("expected root to be *LogicalFilter (HAVING), got %T", plan)
	}

	agg, ok := havingFilter.Child.(*LogicalAggregate)
	if !ok {
		t.Fatalf("expected HAVING child to be *LogicalAggregate, got %T", havingFilter.Child)
	}

	whereFilter, ok := agg.Child.(*LogicalFilter)
	if !ok {
		t.Fatalf("expected Aggregate child to be *LogicalFilter (WHERE), got %T", agg.Child)
	}

	_, ok = whereFilter.Child.(*LogicalScan)
	if !ok {
		t.Fatalf("expected WHERE child to be *LogicalScan, got %T", whereFilter.Child)
	}
}
