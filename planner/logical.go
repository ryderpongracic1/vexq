// Package planner converts a SQL AST into a logical plan, optimises it, and
// maps it to a physical operator tree ready for execution.
package planner

import (
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
	// SourceNames maps an output column name in Schema to the column's name in
	// the table file, for columns the planner renamed (see Build); nil when
	// every column keeps its file name. Schema, NeededCols and Predicate all use
	// output names.
	SourceNames map[string]string
}

// openScan builds the exec scan for n over row groups [rgStart, rgEnd) of r,
// reading the needed columns by their file names and emitting them under their
// output names.
func (n *LogicalScan) openScan(r *storage.Reader, zonePred exec.ZonePredicate, rgStart, rgEnd int) (*exec.TableScan, error) {
	cols := n.NeededCols
	if n.SourceNames != nil && len(cols) > 0 {
		cols = make([]string, len(n.NeededCols))
		for i, name := range n.NeededCols {
			cols[i] = name
			if src, ok := n.SourceNames[name]; ok {
				cols[i] = src
			}
		}
	}
	ts, err := exec.NewTableScanRange(r, cols, zonePred, rgStart, rgEnd)
	if err != nil {
		return nil, err
	}
	if n.SourceNames == nil {
		return ts, nil
	}
	out := n.NeededCols
	if len(out) == 0 {
		out = make([]string, len(n.Schema.Fields))
		for i, f := range n.Schema.Fields {
			out[i] = f.Name
		}
	}
	if err := ts.SetOutputNames(out); err != nil {
		// Not closed: the caller owns r and closes it on error.
		return nil, err
	}
	return ts, nil
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

func (*LogicalFilter) logicalTag()                 {}
func (n *LogicalFilter) OutputSchema() exec.Schema { return n.Child.OutputSchema() }

// LogicalProject evaluates a list of named expressions.
type LogicalProject struct {
	Child LogicalNode
	Exprs []ProjectItem
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
	Child   LogicalNode
	GroupBy []sql.Expr
	Aggs    []AggItem
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
			if agg.AggExpr != nil {
				t = resolveExprType(agg.AggExpr, childSchema)
			} else if agg.ColName == "" {
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
	Func     string   // COUNT, SUM, AVG, MIN, MAX
	ColName  string   // source column name ("" for COUNT(*) or complex expressions)
	AggExpr  sql.Expr // non-nil for complex expressions (e.g. SUM(a * (1-b)))
	Alias    string
	ColIdx   int  // resolved during physical planning
	Distinct bool // true for COUNT(DISTINCT col)
}

// LogicalSort sorts the output.
type LogicalSort struct {
	Child   LogicalNode
	OrderBy []sql.OrderByItem
}

func (*LogicalSort) logicalTag()                 {}
func (n *LogicalSort) OutputSchema() exec.Schema { return n.Child.OutputSchema() }

// LogicalLimit skips Offset rows and then passes at most Count rows; a
// negative Count passes every remaining row (OFFSET without LIMIT).
type LogicalLimit struct {
	Child  LogicalNode
	Count  int64
	Offset int64
}

func (*LogicalLimit) logicalTag()                 {}
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
//
// Simple column and star arguments keep the historical FUNC_col / FUNC_* header
// form (SUM_l_quantity, COUNT_*) so existing output is unchanged. Anything else
// — an aggregate over a computed expression, or an unaliased expression
// projection — renders as readable SQL rather than a Go type name.
func exprName(e sql.Expr) string {
	switch x := e.(type) {
	case *sql.ColumnRefExpr:
		return x.Name
	case *sql.StarExpr:
		return "*"
	case *sql.AggFuncExpr:
		if x.Arg == nil {
			return x.Func
		}
		switch arg := x.Arg.(type) {
		case *sql.ColumnRefExpr:
			return x.Func + "_" + arg.Name
		case *sql.StarExpr:
			return x.Func + "_*"
		}
		return sql.FormatExpr(x)
	default:
		return sql.FormatExpr(e)
	}
}

// LogicalDistinct deduplicates rows.
type LogicalDistinct struct {
	Child LogicalNode
}

func (*LogicalDistinct) logicalTag()                 {}
func (d *LogicalDistinct) OutputSchema() exec.Schema { return d.Child.OutputSchema() }
