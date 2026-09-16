package sql

import (
	"fmt"
	"strconv"
	"strings"
)

// Parser parses a SQL string into an AST using recursive descent.
type Parser struct {
	lexer *Lexer
}

func NewParser(src string) *Parser { return &Parser{lexer: NewLexer(src)} }

// ParseStatement parses one SQL statement.
func (p *Parser) ParseStatement() (Node, error) {
	tok, err := p.peek()
	if err != nil {
		return nil, err
	}
	if tok.Kind != TokSELECT {
		return nil, fmt.Errorf("sql: expected SELECT at position %d, got %q", tok.Pos, tok.Text)
	}
	stmt, err := p.parseSelect()
	if err != nil {
		return nil, err
	}
	// The statement must consume the whole input. Anything the grammar does not
	// recognise is an error rather than silently ignored: a trailing clause the
	// engine skipped would otherwise return a different result than the query
	// asked for.
	if tok, _ := p.peek(); tok.Kind == TokSemicolon {
		p.next()
	}
	tok, err = p.peek()
	if err != nil {
		return nil, err
	}
	switch tok.Kind {
	case TokEOF:
		return stmt, nil
	case TokUNION, TokINTERSECT, TokEXCEPT:
		return nil, fmt.Errorf("sql: %s is not supported (position %d)", strings.ToUpper(tok.Text), tok.Pos)
	default:
		return nil, fmt.Errorf("sql: unexpected %q at position %d", tok.Text, tok.Pos)
	}
}

// ---- SELECT -----------------------------------------------------------------

func (p *Parser) parseSelect() (*SelectStmt, error) {
	if _, err := p.expect(TokSELECT); err != nil {
		return nil, err
	}

	stmt := &SelectStmt{}

	// DISTINCT keyword immediately after SELECT.
	if tok, _ := p.peek(); tok.Kind == TokDISTINCT {
		p.next()
		stmt.Distinct = true
	}

	// Parse column list.
	cols, err := p.parseSelectColumns()
	if err != nil {
		return nil, err
	}
	stmt.Columns = cols

	// FROM (comma-separated table list).
	if tok, _ := p.peek(); tok.Kind == TokFROM {
		p.next()
		for {
			ref, err := p.parseTableRef()
			if err != nil {
				return nil, err
			}
			stmt.From = append(stmt.From, ref)
			if tok, _ := p.peek(); tok.Kind != TokComma {
				break
			}
			p.next() // consume comma
		}
	}

	// WHERE.
	if tok, _ := p.peek(); tok.Kind == TokWHERE {
		p.next()
		expr, err := p.parseExpr(0)
		if err != nil {
			return nil, err
		}
		stmt.Where = expr
	}

	// GROUP BY.
	if tok, _ := p.peek(); tok.Kind == TokGROUP {
		p.next()
		if _, err := p.expect(TokBY); err != nil {
			return nil, err
		}
		groupBy, err := p.parseExprList()
		if err != nil {
			return nil, err
		}
		stmt.GroupBy = groupBy
	}

	// HAVING — post-aggregate filter.
	if tok, _ := p.peek(); tok.Kind == TokHAVING {
		p.next()
		havingExpr, err := p.parseExpr(0)
		if err != nil {
			return nil, err
		}
		stmt.Having = havingExpr
	}

	// ORDER BY.
	if tok, _ := p.peek(); tok.Kind == TokORDER {
		p.next()
		if _, err := p.expect(TokBY); err != nil {
			return nil, err
		}
		items, err := p.parseOrderByList()
		if err != nil {
			return nil, err
		}
		stmt.OrderBy = items
	}

	// LIMIT and OFFSET, in either order.
	for {
		tok, _ := p.peek()
		var dst **int64
		switch tok.Kind {
		case TokLIMIT:
			dst = &stmt.Limit
		case TokOFFSET:
			dst = &stmt.Offset
		}
		if dst == nil {
			break
		}
		if *dst != nil {
			return nil, fmt.Errorf("sql: duplicate %s at position %d", strings.ToUpper(tok.Text), tok.Pos)
		}
		p.next()
		n, err := p.parseCount(strings.ToUpper(tok.Text))
		if err != nil {
			return nil, err
		}
		*dst = &n
	}

	return stmt, nil
}

// parseCount parses the non-negative integer that follows LIMIT or OFFSET.
func (p *Parser) parseCount(clause string) (int64, error) {
	tok, err := p.next()
	if err != nil {
		return 0, err
	}
	if tok.Kind != TokInt {
		return 0, fmt.Errorf("sql: %s requires a non-negative integer, got %q at position %d", clause, tok.Text, tok.Pos)
	}
	n, err := strconv.ParseInt(tok.Text, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("sql: %s value %q at position %d is out of range", clause, tok.Text, tok.Pos)
	}
	return n, nil
}

func (p *Parser) parseSelectColumns() ([]SelectColumn, error) {
	var cols []SelectColumn
	for {
		tok, _ := p.peek()
		if tok.Kind == TokFROM || tok.Kind == TokEOF || tok.Kind == TokSemicolon {
			break
		}

		var sc SelectColumn

		expr, err := p.parseExpr(0)
		if err != nil {
			return nil, err
		}
		sc.Expr = expr

		// Optional AS alias.
		if tok, _ := p.peek(); tok.Kind == TokAS {
			p.next()
			aliasTok, err := p.expectIdent()
			if err != nil {
				return nil, err
			}
			sc.Alias = aliasTok.Text
		} else if tok, _ := p.peek(); tok.Kind == TokIdent {
			// Implicit alias (no AS keyword).
			sc.Alias = tok.Text
			p.next()
		}

		cols = append(cols, sc)

		if tok, _ := p.peek(); tok.Kind != TokComma {
			break
		}
		p.next() // consume comma
	}
	return cols, nil
}

func (p *Parser) parseTableRef() (TableRef, error) {
	nameTok, err := p.expectIdent()
	if err != nil {
		return TableRef{}, err
	}
	ref := TableRef{Name: nameTok.Text}

	// Optional AS alias.
	if tok, _ := p.peek(); tok.Kind == TokAS {
		p.next()
		aliasTok, err := p.expectIdent()
		if err != nil {
			return TableRef{}, err
		}
		ref.Alias = aliasTok.Text
	} else if tok, _ := p.peek(); tok.Kind == TokIdent {
		ref.Alias = tok.Text
		p.next()
	}
	return ref, nil
}

func (p *Parser) parseOrderByList() ([]OrderByItem, error) {
	var items []OrderByItem
	for {
		expr, err := p.parseExpr(0)
		if err != nil {
			return nil, err
		}
		item := OrderByItem{Expr: expr}
		if tok, _ := p.peek(); tok.Kind == TokDESC {
			p.next()
			item.Descending = true
		} else if tok.Kind == TokASC {
			p.next()
		}
		items = append(items, item)
		if tok, _ := p.peek(); tok.Kind != TokComma {
			break
		}
		p.next()
	}
	return items, nil
}

func (p *Parser) parseExprList() ([]Expr, error) {
	var exprs []Expr
	for {
		expr, err := p.parseExpr(0)
		if err != nil {
			return nil, err
		}
		exprs = append(exprs, expr)
		if tok, _ := p.peek(); tok.Kind != TokComma {
			break
		}
		p.next()
	}
	return exprs, nil
}

// ---- Expression parsing (Pratt / precedence climbing) ----------------------

// precedence returns the infix precedence for an operator token.
func precedence(tok Token) int {
	switch tok.Kind {
	case TokOR:
		return 1
	case TokAND:
		return 2
	case TokNOT:
		return 3
	case TokEQ, TokNE:
		return 4
	case TokLT, TokLE, TokGT, TokGE:
		return 5
	case TokIS, TokBETWEEN, TokLIKE, TokIN:
		return 6
	case TokPlus, TokMinus:
		return 7
	case TokStar, TokSlash:
		return 8
	}
	return 0
}

func (p *Parser) parseExpr(minPrec int) (Expr, error) {
	left, err := p.parseUnary()
	if err != nil {
		return nil, err
	}

	for {
		tok, _ := p.peek()
		prec := precedence(tok)
		if prec <= minPrec {
			break
		}
		p.next()

		switch tok.Kind {
		case TokAND:
			right, err := p.parseExpr(prec)
			if err != nil {
				return nil, err
			}
			left = &BinaryExpr{Op: OpAnd, Left: left, Right: right}
		case TokOR:
			right, err := p.parseExpr(prec)
			if err != nil {
				return nil, err
			}
			left = &BinaryExpr{Op: OpOr, Left: left, Right: right}
		case TokEQ:
			right, err := p.parseExpr(prec)
			if err != nil {
				return nil, err
			}
			left = &BinaryExpr{Op: OpEQ, Left: left, Right: right}
		case TokNE:
			right, err := p.parseExpr(prec)
			if err != nil {
				return nil, err
			}
			left = &BinaryExpr{Op: OpNE, Left: left, Right: right}
		case TokLT:
			right, err := p.parseExpr(prec)
			if err != nil {
				return nil, err
			}
			left = &BinaryExpr{Op: OpLT, Left: left, Right: right}
		case TokLE:
			right, err := p.parseExpr(prec)
			if err != nil {
				return nil, err
			}
			left = &BinaryExpr{Op: OpLE, Left: left, Right: right}
		case TokGT:
			right, err := p.parseExpr(prec)
			if err != nil {
				return nil, err
			}
			left = &BinaryExpr{Op: OpGT, Left: left, Right: right}
		case TokGE:
			right, err := p.parseExpr(prec)
			if err != nil {
				return nil, err
			}
			left = &BinaryExpr{Op: OpGE, Left: left, Right: right}
		case TokPlus:
			right, err := p.parseExpr(prec)
			if err != nil {
				return nil, err
			}
			left = &BinaryExpr{Op: OpAdd, Left: left, Right: right}
		case TokMinus:
			right, err := p.parseExpr(prec)
			if err != nil {
				return nil, err
			}
			left = &BinaryExpr{Op: OpSub, Left: left, Right: right}
		case TokStar:
			right, err := p.parseExpr(prec)
			if err != nil {
				return nil, err
			}
			left = &BinaryExpr{Op: OpMul, Left: left, Right: right}
		case TokSlash:
			right, err := p.parseExpr(prec)
			if err != nil {
				return nil, err
			}
			left = &BinaryExpr{Op: OpDiv, Left: left, Right: right}
		case TokIS:
			notTok, _ := p.peek()
			isNot := false
			if notTok.Kind == TokNOT {
				p.next()
				isNot = true
			}
			if _, err := p.expect(TokNULL); err != nil {
				return nil, err
			}
			left = &IsNullExpr{Expr: left, IsNot: isNot}
		case TokBETWEEN:
			lo, err := p.parseExpr(prec)
			if err != nil {
				return nil, err
			}
			if _, err := p.expect(TokAND); err != nil {
				return nil, err
			}
			hi, err := p.parseExpr(prec)
			if err != nil {
				return nil, err
			}
			left = &BetweenExpr{Expr: left, Lo: lo, Hi: hi}
		case TokLIKE:
			pattern, err := p.parseExpr(prec)
			if err != nil {
				return nil, err
			}
			left = &LikeExpr{Expr: left, Pattern: pattern}
		case TokIN:
			if _, err := p.expect(TokLParen); err != nil {
				return nil, err
			}
			list, err := p.parseExprList()
			if err != nil {
				return nil, err
			}
			if _, err := p.expect(TokRParen); err != nil {
				return nil, err
			}
			left = &InExpr{Expr: left, List: list}
		case TokNOT:
			// NOT LIKE, NOT BETWEEN, NOT IN.
			next, _ := p.peek()
			p.next()
			switch next.Kind {
			case TokLIKE:
				pattern, err := p.parseExpr(prec)
				if err != nil {
					return nil, err
				}
				left = &LikeExpr{Expr: left, Pattern: pattern, Not: true}
			case TokBETWEEN:
				lo, err := p.parseExpr(prec)
				if err != nil {
					return nil, err
				}
				if _, err := p.expect(TokAND); err != nil {
					return nil, err
				}
				hi, err := p.parseExpr(prec)
				if err != nil {
					return nil, err
				}
				left = &BetweenExpr{Expr: left, Lo: lo, Hi: hi, Not: true}
			case TokIN:
				if _, err := p.expect(TokLParen); err != nil {
					return nil, err
				}
				list, err := p.parseExprList()
				if err != nil {
					return nil, err
				}
				if _, err := p.expect(TokRParen); err != nil {
					return nil, err
				}
				left = &InExpr{Expr: left, List: list, Not: true}
			default:
				return nil, fmt.Errorf("sql: unexpected NOT ... at position %d", next.Pos)
			}
		default:
			return left, nil
		}
	}
	return left, nil
}

func (p *Parser) parseUnary() (Expr, error) {
	tok, _ := p.peek()
	switch tok.Kind {
	case TokNOT:
		p.next()
		// NOT must bind looser than comparisons so that NOT x = y parses as
		// NOT (x = y), not (NOT x) = y.  Use parseExpr with NOT's infix
		// precedence (3) so the recursive call captures all operators that
		// bind tighter (comparisons at 4–6, arithmetic at 7–8).
		expr, err := p.parseExpr(3)
		if err != nil {
			return nil, err
		}
		return &UnaryExpr{Op: OpNot, Expr: expr}, nil
	case TokMinus:
		p.next()
		// Fold a minus directly in front of a numeric literal into a negative
		// literal. Consumers that only accept literals (IN lists, zone-map
		// pruning, LIMIT-style constants) would otherwise see a UnaryExpr and
		// could not treat -2 as the constant it is. Parsing the sign together
		// with the digits also admits the minimum int64, whose magnitude does
		// not fit in an int64 on its own.
		if next, _ := p.peek(); next.Kind == TokInt || next.Kind == TokFloat {
			p.next()
			return numericLiteral(next, "-")
		}
		expr, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		return &UnaryExpr{Op: OpMinus, Expr: expr}, nil
	}
	return p.parsePrimary()
}

func (p *Parser) parsePrimary() (Expr, error) {
	tok, err := p.next()
	if err != nil {
		return nil, err
	}

	switch tok.Kind {
	case TokInt, TokFloat:
		return numericLiteral(tok, "")

	case TokString:
		return &StringLiteral{Value: tok.Text}, nil

	case TokNULL:
		return &NullLiteral{}, nil

	case TokTRUE:
		return &BoolLiteral{Value: true}, nil

	case TokFALSE:
		return &BoolLiteral{Value: false}, nil

	case TokStar:
		return &StarExpr{}, nil

	case TokLParen:
		// Subquery or parenthesised expression.
		if inner, _ := p.peek(); inner.Kind == TokSELECT {
			sub, err := p.parseSelect()
			if err != nil {
				return nil, err
			}
			if _, err := p.expect(TokRParen); err != nil {
				return nil, err
			}
			return &SubqueryExpr{Query: sub}, nil
		}
		expr, err := p.parseExpr(0)
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(TokRParen); err != nil {
			return nil, err
		}
		return expr, nil

	case TokCASE:
		return p.parseCaseExpr()

	case TokIdent:
		name := tok.Text
		// Check for qualified name: table.column
		if next, _ := p.peek(); next.Kind == TokDot {
			p.next() // consume dot
			colTok, err := p.expectIdent()
			if err != nil {
				return nil, err
			}
			// Possibly a function call: schema.func(...)
			if paren, _ := p.peek(); paren.Kind == TokLParen {
				return p.parseFuncCall(strings.ToUpper(colTok.Text))
			}
			return &ColumnRefExpr{Table: name, Name: colTok.Text}, nil
		}
		upper := strings.ToUpper(name)
		// Aggregate or function call?
		if next, _ := p.peek(); next.Kind == TokLParen {
			return p.parseFuncCall(upper)
		}
		return &ColumnRefExpr{Name: name}, nil

	// Keyword identifiers that can appear as column names.
	case TokSELECT, TokFROM, TokWHERE, TokGROUP, TokBY, TokORDER,
		TokLIMIT, TokOFFSET, TokAS, TokON, TokJOIN, TokINNER, TokLEFT, TokRIGHT:
		return &ColumnRefExpr{Name: tok.Text}, nil
	}

	return nil, fmt.Errorf("sql: unexpected token %q at position %d", tok.Text, tok.Pos)
}

// numericLiteral converts an integer or float token, with an optional sign
// prefix, into a literal node. An integer that does not fit in int64 is an error
// rather than a silent zero.
func numericLiteral(tok Token, sign string) (Expr, error) {
	text := sign + tok.Text
	if tok.Kind == TokInt {
		v, err := strconv.ParseInt(text, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("sql: integer literal %s at position %d is out of range", text, tok.Pos)
		}
		return &IntLiteral{Value: v}, nil
	}
	v, err := strconv.ParseFloat(text, 64)
	if err != nil {
		return nil, fmt.Errorf("sql: invalid numeric literal %s at position %d", text, tok.Pos)
	}
	return &FloatLiteral{Value: v}, nil
}

var aggFuncs = map[string]bool{
	"COUNT": true, "SUM": true, "AVG": true, "MIN": true, "MAX": true,
}

func (p *Parser) parseFuncCall(name string) (Expr, error) {
	if _, err := p.expect(TokLParen); err != nil {
		return nil, err
	}

	if aggFuncs[name] {
		distinct := false
		if tok, _ := p.peek(); tok.Kind == TokDISTINCT {
			p.next()
			distinct = true
		}
		// COUNT(*) special case.
		if tok, _ := p.peek(); tok.Kind == TokStar && name == "COUNT" {
			p.next()
			if _, err := p.expect(TokRParen); err != nil {
				return nil, err
			}
			return &AggFuncExpr{Func: name, Arg: &StarExpr{}, Distinct: distinct}, nil
		}
		arg, err := p.parseExpr(0)
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(TokRParen); err != nil {
			return nil, err
		}
		return &AggFuncExpr{Func: name, Arg: arg, Distinct: distinct}, nil
	}

	// General function.
	var args []Expr
	if tok, _ := p.peek(); tok.Kind != TokRParen {
		list, err := p.parseExprList()
		if err != nil {
			return nil, err
		}
		args = list
	}
	if _, err := p.expect(TokRParen); err != nil {
		return nil, err
	}
	return &FuncExpr{Func: name, Args: args}, nil
}

func (p *Parser) parseCaseExpr() (Expr, error) {
	var whens []WhenClause
	for {
		tok, _ := p.peek()
		if tok.Kind != TokWHEN {
			break
		}
		p.next()
		cond, err := p.parseExpr(0)
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(TokTHEN); err != nil {
			return nil, err
		}
		result, err := p.parseExpr(0)
		if err != nil {
			return nil, err
		}
		whens = append(whens, WhenClause{Cond: cond, Result: result})
	}
	var elseExpr Expr
	if tok, _ := p.peek(); tok.Kind == TokELSE {
		p.next()
		var err error
		elseExpr, err = p.parseExpr(0)
		if err != nil {
			return nil, err
		}
	}
	if _, err := p.expect(TokEND); err != nil {
		return nil, err
	}
	return &CaseExpr{Whens: whens, Else: elseExpr}, nil
}

// ---- helpers ----------------------------------------------------------------

func (p *Parser) next() (Token, error) { return p.lexer.Next() }

func (p *Parser) peek() (Token, error) { return p.lexer.Peek() }

func (p *Parser) expect(kind TokenKind) (Token, error) {
	tok, err := p.lexer.Next()
	if err != nil {
		return Token{}, err
	}
	if tok.Kind != kind {
		return Token{}, fmt.Errorf("sql: expected %d, got %q at position %d", kind, tok.Text, tok.Pos)
	}
	return tok, nil
}

func (p *Parser) expectIdent() (Token, error) {
	tok, err := p.lexer.Next()
	if err != nil {
		return Token{}, err
	}
	// Allow keywords that can also serve as identifiers in column/table names.
	if tok.Kind == TokIdent || tok.Kind >= TokSELECT {
		return tok, nil
	}
	return Token{}, fmt.Errorf("sql: expected identifier, got %q at position %d", tok.Text, tok.Pos)
}
