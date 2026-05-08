package planner

import (
	"context"
	"fmt"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/sql"
)

// Build converts a SQL AST into a logical plan tree.
func Build(ctx context.Context, stmt *sql.SelectStmt, cat *catalog.Catalog) (LogicalNode, error) {
	// FROM clause → scan.
	var scan LogicalNode
	if stmt.From.Name != "" {
		entry, ok := cat.Lookup(ctx, stmt.From.Name)
		if !ok {
			return nil, fmt.Errorf("planner: table %q not found", stmt.From.Name)
		}
		scan = &LogicalScan{
			TableName: entry.Name,
			FilePath:  entry.FilePath,
			Schema:    entry.Schema,
		}
	}

	var root LogicalNode = scan

	// WHERE.
	if stmt.Where != nil && root != nil {
		root = &LogicalFilter{Child: root, Predicate: stmt.Where}
	}

	// GROUP BY / aggregates.
	hasAggs := hasAggregates(stmt.Columns)
	if hasAggs || len(stmt.GroupBy) > 0 {
		agg, err := buildAggregate(root, stmt)
		if err != nil {
			return nil, err
		}
		root = agg
	} else {
		// Project.
		if !isSelectStar(stmt.Columns) {
			proj, err := buildProject(root, stmt)
			if err != nil {
				return nil, err
			}
			root = proj
		}
	}

	// ORDER BY.
	if len(stmt.OrderBy) > 0 {
		root = &LogicalSort{Child: root, OrderBy: stmt.OrderBy}
	}

	// LIMIT.
	if stmt.Limit != nil {
		root = &LogicalLimit{Child: root, Count: *stmt.Limit}
	}

	return root, nil
}

func buildProject(child LogicalNode, stmt *sql.SelectStmt) (*LogicalProject, error) {
	schema := child.OutputSchema()
	var items []ProjectItem
	for _, col := range stmt.Columns {
		alias := col.Alias
		if alias == "" {
			alias = exprName(col.Expr)
		}
		t := resolveExprType(col.Expr, schema)
		items = append(items, ProjectItem{Alias: alias, Expr: col.Expr, Type: t})
	}
	return &LogicalProject{Child: child, Exprs: items}, nil
}

func buildAggregate(child LogicalNode, stmt *sql.SelectStmt) (*LogicalAggregate, error) {
	schema := child.OutputSchema()
	var aggs []AggItem
	for _, col := range stmt.Columns {
		ae, ok := col.Expr.(*sql.AggFuncExpr)
		if !ok {
			continue // group-by columns handled separately
		}
		alias := col.Alias
		if alias == "" {
			alias = exprName(col.Expr)
		}
		colName := ""
		if ae.Arg != nil {
			if cr, ok := ae.Arg.(*sql.ColumnRefExpr); ok {
				colName = cr.Name
			}
		}
		_ = schema
		aggs = append(aggs, AggItem{Func: ae.Func, ColName: colName, Alias: alias})
	}
	return &LogicalAggregate{
		Child:   child,
		GroupBy: stmt.GroupBy,
		Aggs:    aggs,
	}, nil
}

func isSelectStar(cols []sql.SelectColumn) bool {
	return len(cols) == 1 && isStarExpr(cols[0].Expr)
}

func isStarExpr(e sql.Expr) bool {
	_, ok := e.(*sql.StarExpr)
	return ok
}

func hasAggregates(cols []sql.SelectColumn) bool {
	for _, col := range cols {
		if _, ok := col.Expr.(*sql.AggFuncExpr); ok {
			return true
		}
	}
	return false
}

func resolveExprType(expr sql.Expr, schema Schema) DataType {
	switch e := expr.(type) {
	case *sql.ColumnRefExpr:
		for _, f := range schema.Fields {
			if f.Name == e.Name {
				return f.Type
			}
		}
	case *sql.IntLiteral:
		return TypeInt64
	case *sql.FloatLiteral:
		return TypeFloat64
	case *sql.StringLiteral:
		return TypeString
	case *sql.BoolLiteral:
		return TypeBool
	case *sql.BinaryExpr:
		l := resolveExprType(e.Left, schema)
		r := resolveExprType(e.Right, schema)
		if l == TypeFloat64 || r == TypeFloat64 {
			return TypeFloat64
		}
		return l
	case *sql.AggFuncExpr:
		switch e.Func {
		case "COUNT":
			return TypeInt64
		case "AVG":
			return TypeFloat64
		default:
			if e.Arg != nil {
				return resolveExprType(e.Arg, schema)
			}
		}
	}
	return TypeInt64
}
