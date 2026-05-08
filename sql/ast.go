// Package sql provides lexing, parsing, and AST types for the vexq SQL subset.
package sql

// Node is the base interface for all AST nodes.
type Node interface{ nodeTag() }

// ---- Statements -------------------------------------------------------------

// SelectStmt represents a SELECT ... FROM ... WHERE ... GROUP BY ... ORDER BY ... LIMIT ...
type SelectStmt struct {
	Columns []SelectColumn // projections (or * for all)
	From    []TableRef     // one or more tables (comma-separated implicit cross join)
	Where   Expr           // nil if absent
	GroupBy []Expr         // nil if absent
	OrderBy []OrderByItem
	Limit   *int64 // nil if absent
}

func (*SelectStmt) nodeTag() {}

type SelectColumn struct {
	Expr  Expr   // the expression (or * represented as StarExpr)
	Alias string // optional AS alias
}

type TableRef struct {
	Name  string
	Alias string
}

type OrderByItem struct {
	Expr       Expr
	Descending bool
}

// ---- Expressions ------------------------------------------------------------

// Expr is the interface for all expression AST nodes.
type Expr interface {
	Node
	exprTag()
}

// ColumnRef references a table column.
type ColumnRefExpr struct {
	Table string // optional table qualifier
	Name  string
}

func (*ColumnRefExpr) nodeTag() {}
func (*ColumnRefExpr) exprTag() {}

// StarExpr represents *.
type StarExpr struct{}

func (*StarExpr) nodeTag() {}
func (*StarExpr) exprTag() {}

// IntLiteral is an integer constant.
type IntLiteral struct{ Value int64 }

func (*IntLiteral) nodeTag() {}
func (*IntLiteral) exprTag() {}

// FloatLiteral is a floating-point constant.
type FloatLiteral struct{ Value float64 }

func (*FloatLiteral) nodeTag() {}
func (*FloatLiteral) exprTag() {}

// StringLiteral is a single-quoted string constant.
type StringLiteral struct{ Value string }

func (*StringLiteral) nodeTag() {}
func (*StringLiteral) exprTag() {}

// BoolLiteral is TRUE or FALSE.
type BoolLiteral struct{ Value bool }

func (*BoolLiteral) nodeTag() {}
func (*BoolLiteral) exprTag() {}

// NullLiteral is NULL.
type NullLiteral struct{}

func (*NullLiteral) nodeTag() {}
func (*NullLiteral) exprTag() {}

// BinaryExpr is a binary operation.
type BinaryExpr struct {
	Op    BinOp
	Left  Expr
	Right Expr
}

func (*BinaryExpr) nodeTag() {}
func (*BinaryExpr) exprTag() {}

// BinOp is a binary operator.
type BinOp string

const (
	OpEQ  BinOp = "="
	OpNE  BinOp = "<>"
	OpLT  BinOp = "<"
	OpLE  BinOp = "<="
	OpGT  BinOp = ">"
	OpGE  BinOp = ">="
	OpAnd BinOp = "AND"
	OpOr  BinOp = "OR"
	OpAdd BinOp = "+"
	OpSub BinOp = "-"
	OpMul BinOp = "*"
	OpDiv BinOp = "/"
)

// UnaryExpr is a unary operation (e.g. NOT, unary minus).
type UnaryExpr struct {
	Op   UnaryOp
	Expr Expr
}

func (*UnaryExpr) nodeTag() {}
func (*UnaryExpr) exprTag() {}

type UnaryOp string

const (
	OpNot   UnaryOp = "NOT"
	OpMinus UnaryOp = "-"
)

// IsNullExpr is IS NULL / IS NOT NULL.
type IsNullExpr struct {
	Expr   Expr
	IsNot  bool
}

func (*IsNullExpr) nodeTag() {}
func (*IsNullExpr) exprTag() {}

// BetweenExpr is BETWEEN lo AND hi.
type BetweenExpr struct {
	Expr Expr
	Lo   Expr
	Hi   Expr
	Not  bool
}

func (*BetweenExpr) nodeTag() {}
func (*BetweenExpr) exprTag() {}

// InExpr is IN (list) or NOT IN (list).
type InExpr struct {
	Expr Expr
	List []Expr
	Not  bool
}

func (*InExpr) nodeTag() {}
func (*InExpr) exprTag() {}

// LikeExpr is LIKE / NOT LIKE pattern.
type LikeExpr struct {
	Expr    Expr
	Pattern Expr
	Not     bool
}

func (*LikeExpr) nodeTag() {}
func (*LikeExpr) exprTag() {}

// AggFuncExpr is an aggregate function call.
type AggFuncExpr struct {
	Func     string // "COUNT", "SUM", "AVG", "MIN", "MAX"
	Arg      Expr   // nil or StarExpr for COUNT(*)
	Distinct bool
}

func (*AggFuncExpr) nodeTag() {}
func (*AggFuncExpr) exprTag() {}

// FuncExpr is a non-aggregate function call.
type FuncExpr struct {
	Func string
	Args []Expr
}

func (*FuncExpr) nodeTag() {}
func (*FuncExpr) exprTag() {}

// CaseExpr is CASE WHEN ... THEN ... ELSE ... END.
type CaseExpr struct {
	Whens []WhenClause
	Else  Expr
}

func (*CaseExpr) nodeTag() {}
func (*CaseExpr) exprTag() {}

type WhenClause struct {
	Cond   Expr
	Result Expr
}

// SubqueryExpr is a scalar subquery (for WHERE col IN (SELECT ...)).
type SubqueryExpr struct {
	Query *SelectStmt
}

func (*SubqueryExpr) nodeTag() {}
func (*SubqueryExpr) exprTag() {}
