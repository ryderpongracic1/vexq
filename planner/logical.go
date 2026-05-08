// Package planner converts a SQL AST into a logical plan, optimises it, and
// maps it to a physical operator tree ready for execution.
package planner

import (
	"fmt"

	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

// LogicalNode is the interface for all logical plan nodes.
type LogicalNode interface {
	logicalTag()
	OutputSchema() exec.Schema
}

// LogicalScan reads a table.
type LogicalScan struct {
	TableName string
	FilePath  string
	Schema    exec.Schema
	// Pushed-down predicate (set by optimizer).
	Predicate sql.Expr
	// Column set needed by ancestors (set by optimizer); nil = all.
	NeededCols []string
}

func (*LogicalScan) logicalTag() {}
func (n *LogicalScan) OutputSchema() exec.Schema {
	if len(n.NeededCols) == 0 {
		return n.Schema
	}
	var fields []exec.Field
	for _, name := range n.NeededCols {
		for _, f := range n.Schema.Fields {
			if f.Name == name {
				fields = append(fields, f)
				break
			}
		}
	}
	return exec.Schema{Fields: fields}
}

// LogicalFilter filters rows with a predicate.
type LogicalFilter struct {
	Child     LogicalNode
	Predicate sql.Expr
}

func (*LogicalFilter) logicalTag() {}
func (n *LogicalFilter) OutputSchema() exec.Schema { return n.Child.OutputSchema() }

// LogicalProject evaluates a list of named expressions.
type LogicalProject struct {
	Child   LogicalNode
	Exprs   []ProjectItem
}

func (*LogicalProject) logicalTag() {}
func (n *LogicalProject) OutputSchema() exec.Schema {
	fields := make([]storage.Field, len(n.Exprs))
	for i, pe := range n.Exprs {
		fields[i] = storage.Field{Name: pe.Alias, Type: pe.Type, Nullable: true}
	}
	return exec.Schema{Fields: fields}
}

type ProjectItem struct {
	Alias string
	Expr  sql.Expr
	Type  exec.DataType
}

// LogicalAggregate groups and aggregates.
type LogicalAggregate struct {
	Child    LogicalNode
	GroupBy  []sql.Expr
	Aggs     []AggItem
}

func (*LogicalAggregate) logicalTag() {}
func (n *LogicalAggregate) OutputSchema() exec.Schema {
	childSchema := n.Child.OutputSchema()
	var fields []storage.Field
	for _, expr := range n.GroupBy {
		name := exprName(expr)
		for _, f := range childSchema.Fields {
			if f.Name == name {
				fields = append(fields, f)
				break
			}
		}
	}
	for _, agg := range n.Aggs {
		var t exec.DataType
		switch agg.Func {
		case "COUNT":
			t = exec.TypeInt64
		case "SUM", "MIN", "MAX":
			if agg.ColName == "" {
				t = exec.TypeInt64
			} else {
				for _, f := range childSchema.Fields {
					if f.Name == agg.ColName {
						t = f.Type
						break
					}
				}
				if t == 0 {
					t = exec.TypeFloat64
				}
			}
		case "AVG":
			t = exec.TypeFloat64
		default:
			t = exec.TypeFloat64
		}
		fields = append(fields, storage.Field{Name: agg.Alias, Type: t, Nullable: true})
	}
	return exec.Schema{Fields: fields}
}

type AggItem struct {
	Func    string   // COUNT, SUM, AVG, MIN, MAX
	ColName string   // source column name ("" for COUNT(*) or complex expressions)
	AggExpr sql.Expr // non-nil for complex expressions (e.g. SUM(a * (1-b)))
	Alias   string
	ColIdx  int // resolved during physical planning
}

// LogicalSort sorts the output.
type LogicalSort struct {
	Child    LogicalNode
	OrderBy  []sql.OrderByItem
}

func (*LogicalSort) logicalTag() {}
func (n *LogicalSort) OutputSchema() exec.Schema { return n.Child.OutputSchema() }

// LogicalLimit limits output rows.
type LogicalLimit struct {
	Child LogicalNode
	Count int64
}

func (*LogicalLimit) logicalTag() {}
func (n *LogicalLimit) OutputSchema() exec.Schema { return n.Child.OutputSchema() }

// LogicalJoin is an inner join.
type LogicalJoin struct {
	Left, Right LogicalNode
	Condition   sql.Expr
}

func (*LogicalJoin) logicalTag() {}
func (n *LogicalJoin) OutputSchema() exec.Schema {
	l := n.Left.OutputSchema()
	r := n.Right.OutputSchema()
	fields := append(append([]storage.Field{}, l.Fields...), r.Fields...)
	return exec.Schema{Fields: fields}
}

// exprName returns a best-effort string name for an expression (for schema purposes).
func exprName(e sql.Expr) string {
	switch x := e.(type) {
	case *sql.ColumnRefExpr:
		return x.Name
	case *sql.StarExpr:
		return "*"
	case *sql.AggFuncExpr:
		if x.Arg != nil {
			return x.Func + "_" + exprName(x.Arg)
		}
		return x.Func
	default:
		return fmt.Sprintf("%T", e)
	}
}
