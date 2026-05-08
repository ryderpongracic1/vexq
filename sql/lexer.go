package sql

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// TokenKind identifies the category of a lexical token.
type TokenKind int

const (
	TokEOF TokenKind = iota
	TokIdent
	TokInt
	TokFloat
	TokString // single-quoted
	// Punctuation
	TokLParen
	TokRParen
	TokComma
	TokDot
	TokStar
	TokSemicolon
	// Operators
	TokEQ   // =
	TokNE   // <> or !=
	TokLT   // <
	TokLE   // <=
	TokGT   // >
	TokGE   // >=
	TokPlus
	TokMinus
	TokSlash
	// Keywords (resolved from TokIdent at lex time)
	TokSELECT
	TokFROM
	TokWHERE
	TokGROUP
	TokBY
	TokORDER
	TokLIMIT
	TokAS
	TokAND
	TokOR
	TokNOT
	TokIN
	TokIS
	TokNULL
	TokTRUE
	TokFALSE
	TokBETWEEN
	TokLIKE
	TokCASE
	TokWHEN
	TokTHEN
	TokELSE
	TokEND
	TokDISTINCT
	TokASC
	TokDESC
	TokINNER
	TokJOIN
	TokON
	TokLEFT
	TokRIGHT
	TokOUTER
	TokCROSS
	TokHAVING
)

var keywords = map[string]TokenKind{
	"SELECT":   TokSELECT,
	"FROM":     TokFROM,
	"WHERE":    TokWHERE,
	"GROUP":    TokGROUP,
	"BY":       TokBY,
	"ORDER":    TokORDER,
	"LIMIT":    TokLIMIT,
	"AS":       TokAS,
	"AND":      TokAND,
	"OR":       TokOR,
	"NOT":      TokNOT,
	"IN":       TokIN,
	"IS":       TokIS,
	"NULL":     TokNULL,
	"TRUE":     TokTRUE,
	"FALSE":    TokFALSE,
	"BETWEEN":  TokBETWEEN,
	"LIKE":     TokLIKE,
	"CASE":     TokCASE,
	"WHEN":     TokWHEN,
	"THEN":     TokTHEN,
	"ELSE":     TokELSE,
	"END":      TokEND,
	"DISTINCT": TokDISTINCT,
	"ASC":      TokASC,
	"DESC":     TokDESC,
	"INNER":    TokINNER,
	"JOIN":     TokJOIN,
	"ON":       TokON,
	"LEFT":     TokLEFT,
	"RIGHT":    TokRIGHT,
	"OUTER":    TokOUTER,
	"CROSS":    TokCROSS,
	"HAVING":   TokHAVING,
}

// Token is a lexical token produced by the Lexer.
type Token struct {
	Kind TokenKind
	Text string
	Pos  int
}

// Lexer tokenises a SQL string.
type Lexer struct {
	src  string
	pos  int
	peek *Token
}

func NewLexer(src string) *Lexer { return &Lexer{src: src} }

// Peek returns the next token without consuming it.
func (l *Lexer) Peek() (Token, error) {
	if l.peek != nil {
		return *l.peek, nil
	}
	tok, err := l.scan()
	if err != nil {
		return Token{}, err
	}
	l.peek = &tok
	return tok, nil
}

// Next consumes and returns the next token.
func (l *Lexer) Next() (Token, error) {
	if l.peek != nil {
		tok := *l.peek
		l.peek = nil
		return tok, nil
	}
	return l.scan()
}

// scan reads the next token from l.src[l.pos:].
func (l *Lexer) scan() (Token, error) {
	l.skipWhitespaceAndComments()
	if l.pos >= len(l.src) {
		return Token{Kind: TokEOF, Pos: l.pos}, nil
	}

	start := l.pos
	ch := l.src[l.pos]

	switch {
	case ch == '\'':
		return l.scanString(start)
	case ch == '"' || ch == '`':
		return l.scanQuotedIdent(start)
	case ch >= '0' && ch <= '9':
		return l.scanNumber(start)
	case isLetter(ch):
		return l.scanIdent(start)
	}

	l.pos++
	switch ch {
	case '(':
		return Token{Kind: TokLParen, Text: "(", Pos: start}, nil
	case ')':
		return Token{Kind: TokRParen, Text: ")", Pos: start}, nil
	case ',':
		return Token{Kind: TokComma, Text: ",", Pos: start}, nil
	case '.':
		return Token{Kind: TokDot, Text: ".", Pos: start}, nil
	case '*':
		return Token{Kind: TokStar, Text: "*", Pos: start}, nil
	case ';':
		return Token{Kind: TokSemicolon, Text: ";", Pos: start}, nil
	case '+':
		return Token{Kind: TokPlus, Text: "+", Pos: start}, nil
	case '-':
		return Token{Kind: TokMinus, Text: "-", Pos: start}, nil
	case '/':
		return Token{Kind: TokSlash, Text: "/", Pos: start}, nil
	case '=':
		return Token{Kind: TokEQ, Text: "=", Pos: start}, nil
	case '<':
		if l.pos < len(l.src) && l.src[l.pos] == '=' {
			l.pos++
			return Token{Kind: TokLE, Text: "<=", Pos: start}, nil
		}
		if l.pos < len(l.src) && l.src[l.pos] == '>' {
			l.pos++
			return Token{Kind: TokNE, Text: "<>", Pos: start}, nil
		}
		return Token{Kind: TokLT, Text: "<", Pos: start}, nil
	case '>':
		if l.pos < len(l.src) && l.src[l.pos] == '=' {
			l.pos++
			return Token{Kind: TokGE, Text: ">=", Pos: start}, nil
		}
		return Token{Kind: TokGT, Text: ">", Pos: start}, nil
	case '!':
		if l.pos < len(l.src) && l.src[l.pos] == '=' {
			l.pos++
			return Token{Kind: TokNE, Text: "!=", Pos: start}, nil
		}
	}
	return Token{}, fmt.Errorf("sql: unexpected character %q at position %d", ch, start)
}

func (l *Lexer) scanString(start int) (Token, error) {
	l.pos++ // skip opening '
	var sb strings.Builder
	for l.pos < len(l.src) {
		ch := l.src[l.pos]
		if ch == '\'' {
			l.pos++
			if l.pos < len(l.src) && l.src[l.pos] == '\'' {
				// Escaped quote ''
				sb.WriteByte('\'')
				l.pos++
				continue
			}
			return Token{Kind: TokString, Text: sb.String(), Pos: start}, nil
		}
		sb.WriteByte(ch)
		l.pos++
	}
	return Token{}, fmt.Errorf("sql: unterminated string at position %d", start)
}

func (l *Lexer) scanQuotedIdent(start int) (Token, error) {
	delim := l.src[l.pos]
	l.pos++
	end := delim
	if delim == '"' {
		end = '"'
	} else if delim == '`' {
		end = '`'
	}
	var sb strings.Builder
	for l.pos < len(l.src) {
		ch := l.src[l.pos]
		if ch == end {
			l.pos++
			return Token{Kind: TokIdent, Text: sb.String(), Pos: start}, nil
		}
		sb.WriteByte(ch)
		l.pos++
	}
	return Token{}, fmt.Errorf("sql: unterminated quoted identifier at position %d", start)
}

func (l *Lexer) scanNumber(start int) (Token, error) {
	isFloat := false
	for l.pos < len(l.src) {
		ch := l.src[l.pos]
		if ch >= '0' && ch <= '9' {
			l.pos++
		} else if ch == '.' && !isFloat {
			isFloat = true
			l.pos++
		} else if ch == 'e' || ch == 'E' {
			isFloat = true
			l.pos++
			if l.pos < len(l.src) && (l.src[l.pos] == '+' || l.src[l.pos] == '-') {
				l.pos++
			}
		} else {
			break
		}
	}
	text := l.src[start:l.pos]
	if isFloat {
		return Token{Kind: TokFloat, Text: text, Pos: start}, nil
	}
	return Token{Kind: TokInt, Text: text, Pos: start}, nil
}

func (l *Lexer) scanIdent(start int) (Token, error) {
	for l.pos < len(l.src) && (isLetter(l.src[l.pos]) || (l.src[l.pos] >= '0' && l.src[l.pos] <= '9') || l.src[l.pos] == '_') {
		l.pos++
	}
	text := l.src[start:l.pos]
	upper := strings.ToUpper(text)
	if kw, ok := keywords[upper]; ok {
		return Token{Kind: kw, Text: upper, Pos: start}, nil
	}
	return Token{Kind: TokIdent, Text: text, Pos: start}, nil
}

func (l *Lexer) skipWhitespaceAndComments() {
	for l.pos < len(l.src) {
		ch := l.src[l.pos]
		if ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' {
			l.pos++
			continue
		}
		// Line comment --
		if ch == '-' && l.pos+1 < len(l.src) && l.src[l.pos+1] == '-' {
			for l.pos < len(l.src) && l.src[l.pos] != '\n' {
				l.pos++
			}
			continue
		}
		// Block comment /* ... */
		if ch == '/' && l.pos+1 < len(l.src) && l.src[l.pos+1] == '*' {
			l.pos += 2
			for l.pos+1 < len(l.src) {
				if l.src[l.pos] == '*' && l.src[l.pos+1] == '/' {
					l.pos += 2
					break
				}
				l.pos++
			}
			continue
		}
		break
	}
}

func isLetter(ch byte) bool {
	r, _ := utf8.DecodeRune([]byte{ch})
	return unicode.IsLetter(r) || ch == '_'
}
