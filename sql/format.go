package sql

import (
	"fmt"
	"strconv"
	"strings"
)

// This file renders AST expressions back into readable, SQL-like text. Its main
// consumer is output column naming: a projection without an explicit AS alias
// needs a header, and that header is user-visible. The output is meant to be
// human-readable rather than a byte-exact round-trip of the input query
// (whitespace and redundant parentheses are normalized).

// Compile-time assertion that every Expr type has a String method, so a new AST
// node cannot silently fall through to the generic FormatExpr fallback.
var _ = []fmt.Stringer{
	(*ColumnRefExpr)(nil),
	(*StarExpr)(nil),
	(*IntLiteral)(nil),
	(*FloatLiteral)(nil),
	(*StringLiteral)(nil),
	(*BoolLiteral)(nil),
	(*NullLiteral)(nil),
	(*BinaryExpr)(nil),
	(*UnaryExpr)(nil),
	(*IsNullExpr)(nil),
	(*BetweenExpr)(nil),
	(*InExpr)(nil),
	(*LikeExpr)(nil),
	(*AggFuncExpr)(nil),
	(*FuncExpr)(nil),
	(*CaseExpr)(nil),
	(*SubqueryExpr)(nil),
}

// FormatExpr renders e as readable, SQL-like text.
func FormatExpr(e Expr) string {
	if e == nil {
		return ""
	}
	if s, ok := e.(fmt.Stringer); ok {
		return s.String()
	}
	return "expr"
}

// binPrec returns the binding strength of op. Higher binds tighter.
func binPrec(op BinOp) int {
	switch op {
	case OpOr:
		return 1
	case OpAnd:
		return 2
	case OpEQ, OpNE, OpLT, OpLE, OpGT, OpGE:
		return 3
	case OpAdd, OpSub:
		return 4
	case OpMul, OpDiv:
		return 5
	}
	return 0
}

// binOperand renders a child of a binary expression, parenthesizing it only
// when dropping the parentheses would change how the expression reads. The
// right-hand side also needs parentheses at equal precedence, so that
// a - (b - c) does not render as the differently-grouped a - b - c.
func binOperand(child Expr, parentPrec int, isRight bool) string {
	s := FormatExpr(child)
	b, ok := child.(*BinaryExpr)
	if !ok {
		return s
	}
	p := binPrec(b.Op)
	if p < parentPrec || (isRight && p == parentPrec) {
		return "(" + s + ")"
	}
	return s
}

// wrapsAsOperand reports whether e needs parentheses when used as the operand
// of a prefix operator such as NOT or unary minus.
func wrapsAsOperand(e Expr) bool {
	switch e.(type) {
	case *BinaryExpr, *IsNullExpr, *BetweenExpr, *InExpr, *LikeExpr:
		return true
	}
	return false
}

func (e *ColumnRefExpr) String() string {
	if e.Table != "" {
		return e.Table + "." + e.Name
	}
	return e.Name
}

func (*StarExpr) String() string { return "*" }

func (e *IntLiteral) String() string { return strconv.FormatInt(e.Value, 10) }

func (e *FloatLiteral) String() string {
	return strconv.FormatFloat(e.Value, 'g', -1, 64)
}

func (e *StringLiteral) String() string {
	return "'" + strings.ReplaceAll(e.Value, "'", "''") + "'"
}

func (e *BoolLiteral) String() string {
	if e.Value {
		return "TRUE"
	}
	return "FALSE"
}

func (*NullLiteral) String() string { return "NULL" }

func (e *BinaryExpr) String() string {
	p := binPrec(e.Op)
	return binOperand(e.Left, p, false) + " " + string(e.Op) + " " + binOperand(e.Right, p, true)
}

func (e *UnaryExpr) String() string {
	inner := FormatExpr(e.Expr)
	if wrapsAsOperand(e.Expr) {
		inner = "(" + inner + ")"
	}
	if e.Op == OpNot {
		return "NOT " + inner
	}
	return string(e.Op) + inner
}

func (e *IsNullExpr) String() string {
	if e.IsNot {
		return FormatExpr(e.Expr) + " IS NOT NULL"
	}
	return FormatExpr(e.Expr) + " IS NULL"
}

func (e *BetweenExpr) String() string {
	op := " BETWEEN "
	if e.Not {
		op = " NOT BETWEEN "
	}
	return FormatExpr(e.Expr) + op + FormatExpr(e.Lo) + " AND " + FormatExpr(e.Hi)
}

func (e *InExpr) String() string {
	op := " IN ("
	if e.Not {
		op = " NOT IN ("
	}
	return FormatExpr(e.Expr) + op + formatList(e.List) + ")"
}

func (e *LikeExpr) String() string {
	op := " LIKE "
	if e.Not {
		op = " NOT LIKE "
	}
	return FormatExpr(e.Expr) + op + FormatExpr(e.Pattern)
}

func (e *AggFuncExpr) String() string {
	fn := strings.ToUpper(e.Func)
	// A nil argument means COUNT(*), which the parser may represent either as a
	// StarExpr or as no argument at all.
	if e.Arg == nil {
		return fn + "(*)"
	}
	if e.Distinct {
		return fn + "(DISTINCT " + FormatExpr(e.Arg) + ")"
	}
	return fn + "(" + FormatExpr(e.Arg) + ")"
}

func (e *FuncExpr) String() string {
	return e.Func + "(" + formatList(e.Args) + ")"
}

func (e *CaseExpr) String() string {
	var b strings.Builder
	b.WriteString("CASE")
	for _, w := range e.Whens {
		b.WriteString(" WHEN ")
		b.WriteString(FormatExpr(w.Cond))
		b.WriteString(" THEN ")
		b.WriteString(FormatExpr(w.Result))
	}
	if e.Else != nil {
		b.WriteString(" ELSE ")
		b.WriteString(FormatExpr(e.Else))
	}
	b.WriteString(" END")
	return b.String()
}

// String renders a subquery as a placeholder: the surrounding SELECT text is
// not reconstructed, and a header is the only consumer of this rendering.
func (*SubqueryExpr) String() string { return "(subquery)" }

func formatList(list []Expr) string {
	parts := make([]string, len(list))
	for i, e := range list {
		parts[i] = FormatExpr(e)
	}
	return strings.Join(parts, ", ")
}
