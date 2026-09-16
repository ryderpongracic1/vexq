package planner

import (
	"fmt"

	"github.com/ryderpongracic1/vexq/sql"
)

// walkExpr calls visit for e and every expression nested inside it, parents
// before children. It is the single place that knows every AST expression's
// children, so column collection, aggregate detection and rewriting cannot
// disagree about which sub-expressions exist — a walker that forgot LIKE's
// operand is how a LIKE term used to vanish from a join query.
func walkExpr(e sql.Expr, visit func(sql.Expr)) {
	if e == nil {
		return
	}
	visit(e)
	switch x := e.(type) {
	case *sql.BinaryExpr:
		walkExpr(x.Left, visit)
		walkExpr(x.Right, visit)
	case *sql.UnaryExpr:
		walkExpr(x.Expr, visit)
	case *sql.IsNullExpr:
		walkExpr(x.Expr, visit)
	case *sql.BetweenExpr:
		walkExpr(x.Expr, visit)
		walkExpr(x.Lo, visit)
		walkExpr(x.Hi, visit)
	case *sql.InExpr:
		walkExpr(x.Expr, visit)
		for _, item := range x.List {
			walkExpr(item, visit)
		}
	case *sql.LikeExpr:
		walkExpr(x.Expr, visit)
		walkExpr(x.Pattern, visit)
	case *sql.AggFuncExpr:
		walkExpr(x.Arg, visit)
	case *sql.FuncExpr:
		for _, arg := range x.Args {
			walkExpr(arg, visit)
		}
	case *sql.CaseExpr:
		for _, w := range x.Whens {
			walkExpr(w.Cond, visit)
			walkExpr(w.Result, visit)
		}
		walkExpr(x.Else, visit)
	}
}

// rewriteExpr returns a copy of e in which fn has replaced nodes. fn is called
// on each node before its children; when it returns handled=true its result is
// used as-is and the node's children are not visited. Otherwise the node is
// rebuilt from its rewritten children. The input tree is never mutated, so a
// rewritten plan cannot alias the parsed statement.
func rewriteExpr(e sql.Expr, fn func(sql.Expr) (sql.Expr, bool, error)) (sql.Expr, error) {
	if e == nil {
		return nil, nil
	}
	if out, handled, err := fn(e); err != nil || handled {
		return out, err
	}
	rw := func(c sql.Expr) (sql.Expr, error) { return rewriteExpr(c, fn) }
	switch x := e.(type) {
	case *sql.ColumnRefExpr, *sql.StarExpr, *sql.IntLiteral, *sql.FloatLiteral,
		*sql.StringLiteral, *sql.BoolLiteral, *sql.NullLiteral:
		return e, nil
	case *sql.BinaryExpr:
		l, err := rw(x.Left)
		if err != nil {
			return nil, err
		}
		r, err := rw(x.Right)
		if err != nil {
			return nil, err
		}
		return &sql.BinaryExpr{Op: x.Op, Left: l, Right: r}, nil
	case *sql.UnaryExpr:
		c, err := rw(x.Expr)
		if err != nil {
			return nil, err
		}
		return &sql.UnaryExpr{Op: x.Op, Expr: c}, nil
	case *sql.IsNullExpr:
		c, err := rw(x.Expr)
		if err != nil {
			return nil, err
		}
		return &sql.IsNullExpr{Expr: c, IsNot: x.IsNot}, nil
	case *sql.BetweenExpr:
		c, err := rw(x.Expr)
		if err != nil {
			return nil, err
		}
		lo, err := rw(x.Lo)
		if err != nil {
			return nil, err
		}
		hi, err := rw(x.Hi)
		if err != nil {
			return nil, err
		}
		return &sql.BetweenExpr{Expr: c, Lo: lo, Hi: hi, Not: x.Not}, nil
	case *sql.InExpr:
		c, err := rw(x.Expr)
		if err != nil {
			return nil, err
		}
		list := make([]sql.Expr, len(x.List))
		for i, item := range x.List {
			if list[i], err = rw(item); err != nil {
				return nil, err
			}
		}
		return &sql.InExpr{Expr: c, List: list, Not: x.Not}, nil
	case *sql.LikeExpr:
		c, err := rw(x.Expr)
		if err != nil {
			return nil, err
		}
		pat, err := rw(x.Pattern)
		if err != nil {
			return nil, err
		}
		return &sql.LikeExpr{Expr: c, Pattern: pat, Not: x.Not}, nil
	case *sql.AggFuncExpr:
		arg, err := rw(x.Arg)
		if err != nil {
			return nil, err
		}
		return &sql.AggFuncExpr{Func: x.Func, Arg: arg, Distinct: x.Distinct}, nil
	case *sql.FuncExpr:
		args := make([]sql.Expr, len(x.Args))
		for i, a := range x.Args {
			var err error
			if args[i], err = rw(a); err != nil {
				return nil, err
			}
		}
		return &sql.FuncExpr{Func: x.Func, Args: args}, nil
	case *sql.CaseExpr:
		whens := make([]sql.WhenClause, len(x.Whens))
		for i, w := range x.Whens {
			cond, err := rw(w.Cond)
			if err != nil {
				return nil, err
			}
			res, err := rw(w.Result)
			if err != nil {
				return nil, err
			}
			whens[i] = sql.WhenClause{Cond: cond, Result: res}
		}
		els, err := rw(x.Else)
		if err != nil {
			return nil, err
		}
		return &sql.CaseExpr{Whens: whens, Else: els}, nil
	case *sql.SubqueryExpr:
		return nil, fmt.Errorf("planner: subqueries are not supported")
	}
	return nil, fmt.Errorf("planner: unsupported expression %T", e)
}

// containsAggregate reports whether e contains an aggregate function call at any
// depth.
func containsAggregate(e sql.Expr) bool {
	found := false
	walkExpr(e, func(n sql.Expr) {
		if _, ok := n.(*sql.AggFuncExpr); ok {
			found = true
		}
	})
	return found
}
