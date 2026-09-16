package planner

import (
	"context"
	"fmt"
	"strings"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/sql"
)

// Build converts a SQL AST into a logical plan tree.
//
// Every column reference in the statement is resolved here, against the tables
// in FROM, before any plan node is built. Resolution replaces each reference
// with the unique name its column carries through the plan: the bare column name
// when only one FROM table has a column by that name, and "table.column" (using
// the table's alias when it has one) when several do. Scans emit those names, so
// every operator above a scan — joins, filters, projections, aggregates, sorts —
// can locate columns by name without two tables' same-named columns being
// confused. Unknown and ambiguous references are reported here, wherever in the
// statement they appear.
//
// The plan shape, bottom to top:
//
//	scans and joins (with per-table filters pushed into the scans, and any WHERE
//	term that is not a join key applied as a filter above the joins)
//	→ Aggregate → Filter(HAVING)           (aggregate queries only)
//	→ Project                              (SELECT list, plus hidden ORDER BY keys)
//	→ Distinct → Sort → Limit
//	→ Project                              (only when hidden keys or duplicate
//	                                        output names must be removed)
//
// A Project that would pass its input through unchanged is omitted.
func Build(ctx context.Context, stmt *sql.SelectStmt, cat *catalog.Catalog) (LogicalNode, error) {
	if len(stmt.From) == 0 {
		return nil, fmt.Errorf("planner: no FROM clause")
	}

	scans := make([]*LogicalScan, len(stmt.From))
	schemas := make([]exec.Schema, len(stmt.From))
	for i, ref := range stmt.From {
		entry, ok := cat.Lookup(ctx, ref.Name)
		if !ok {
			return nil, fmt.Errorf("planner: table %q not found", ref.Name)
		}
		scans[i] = &LogicalScan{
			TableName: entry.Name,
			FilePath:  entry.FilePath,
			Schema:    entry.Schema,
		}
		schemas[i] = entry.Schema
	}

	st, err := newSymbolTable(schemas, stmt.From)
	if err != nil {
		return nil, err
	}
	for i, scan := range scans {
		st.applyOutputNames(i, scan)
	}

	where, err := st.resolveScalar(stmt.Where, "WHERE")
	if err != nil {
		return nil, err
	}

	var root LogicalNode
	if len(scans) == 1 {
		root = scans[0]
		if where != nil {
			root = &LogicalFilter{Child: root, Predicate: where}
		}
	} else {
		root, err = buildMultiTablePlan(scans, where, st)
		if err != nil {
			return nil, err
		}
	}

	items := expandSelectList(stmt.Columns, st)

	aggregating := len(stmt.GroupBy) > 0 || containsAggregate(stmt.Having)
	for _, it := range items {
		if containsAggregate(it.expr) {
			aggregating = true
		}
	}

	var q *selectQuery
	if aggregating {
		q, err = buildAggregation(root, stmt, items, st)
	} else {
		if stmt.Having != nil {
			return nil, fmt.Errorf("planner: HAVING requires GROUP BY or an aggregate function")
		}
		q, err = buildScalarSelect(root, items, st)
	}
	if err != nil {
		return nil, err
	}
	return q.finish(stmt)
}

// ---- Column resolution -------------------------------------------------------

// columnBinding is one column of one FROM table.
type columnBinding struct {
	table  int
	source string // column name in the table file
	out    string // name the column carries through the plan
}

// symbolTable maps qualified "table.col" and unqualified "col" references to
// the columns of the tables in a FROM clause.
type symbolTable struct {
	tableNames []string                    // table name or alias, per FROM position
	qualified  map[string]*columnBinding   // "table.col" → binding
	byName     map[string][]*columnBinding // "col" → every table's binding for it
	tableOfOut map[string]int              // output name → FROM position
	columns    [][]*columnBinding          // per table, in schema order
}

// newSymbolTable builds a symbol table from the given schemas and table refs.
func newSymbolTable(schemas []exec.Schema, tableRefs []sql.TableRef) (*symbolTable, error) {
	st := &symbolTable{
		tableNames: make([]string, len(schemas)),
		qualified:  make(map[string]*columnBinding),
		byName:     make(map[string][]*columnBinding),
		tableOfOut: make(map[string]int),
		columns:    make([][]*columnBinding, len(schemas)),
	}
	for i, ref := range tableRefs {
		name := ref.Alias
		if name == "" {
			name = ref.Name
		}
		for j := 0; j < i; j++ {
			if st.tableNames[j] == name {
				return nil, fmt.Errorf("planner: table name %q specified more than once; give one of them an alias", name)
			}
		}
		st.tableNames[i] = name
		for _, f := range schemas[i].Fields {
			b := &columnBinding{table: i, source: f.Name}
			st.qualified[name+"."+f.Name] = b
			st.byName[f.Name] = append(st.byName[f.Name], b)
			st.columns[i] = append(st.columns[i], b)
		}
	}
	for i := range st.columns {
		for _, b := range st.columns[i] {
			b.out = b.source
			if len(st.byName[b.source]) > 1 {
				b.out = st.tableNames[i] + "." + b.source
			}
			st.tableOfOut[b.out] = i
		}
	}
	return st, nil
}

// applyOutputNames renames scan's schema to the plan-wide output names of its
// table's columns, recording the file column names the scan must read.
func (st *symbolTable) applyOutputNames(table int, scan *LogicalScan) {
	fields := make([]exec.Field, len(scan.Schema.Fields))
	copy(fields, scan.Schema.Fields)
	for i, b := range st.columns[table] {
		if b.out == b.source {
			continue
		}
		if scan.SourceNames == nil {
			scan.SourceNames = make(map[string]string)
		}
		scan.SourceNames[b.out] = b.source
		fields[i].Name = b.out
	}
	scan.Schema = exec.Schema{Fields: fields}
}

// resolve looks up a column reference in the symbol table.
func (st *symbolTable) resolve(ref *sql.ColumnRefExpr) (*columnBinding, error) {
	if ref.Table != "" {
		b, ok := st.qualified[ref.Table+"."+ref.Name]
		if !ok {
			return nil, fmt.Errorf("column %q not found in table %q", ref.Name, ref.Table)
		}
		return b, nil
	}
	switch bs := st.byName[ref.Name]; len(bs) {
	case 0:
		return nil, fmt.Errorf("column %q not found in any table", ref.Name)
	case 1:
		return bs[0], nil
	default:
		return nil, fmt.Errorf("column %q is ambiguous; qualify with table name", ref.Name)
	}
}

// resolveColumns rewrites every column reference in e to its output name. It
// does not look inside aggregates' arguments differently from anything else; the
// caller decides whether aggregates are allowed.
func (st *symbolTable) resolveColumns(e sql.Expr) (sql.Expr, error) {
	return rewriteExpr(e, func(n sql.Expr) (sql.Expr, bool, error) {
		ref, ok := n.(*sql.ColumnRefExpr)
		if !ok {
			return nil, false, nil
		}
		b, err := st.resolve(ref)
		if err != nil {
			return nil, true, fmt.Errorf("planner: %w", err)
		}
		return &sql.ColumnRefExpr{Name: b.out}, true, nil
	})
}

// resolveScalar resolves an expression that is evaluated per input row, where
// aggregate functions are not allowed. clause names the clause for errors.
func (st *symbolTable) resolveScalar(e sql.Expr, clause string) (sql.Expr, error) {
	if e == nil {
		return nil, nil
	}
	if containsAggregate(e) {
		return nil, fmt.Errorf("planner: aggregate functions are not allowed in %s", clause)
	}
	return st.resolveColumns(e)
}

// tablesOf returns the set of FROM positions whose columns a resolved expression
// references.
func (st *symbolTable) tablesOf(e sql.Expr) map[int]bool {
	set := make(map[int]bool)
	walkExpr(e, func(n sql.Expr) {
		if ref, ok := n.(*sql.ColumnRefExpr); ok {
			if t, found := st.tableOfOut[ref.Name]; found {
				set[t] = true
			}
		}
	})
	return set
}

// ---- Joins -------------------------------------------------------------------

// buildMultiTablePlan builds a left-deep join tree from multiple table scans.
// where must already be resolved.
//
// Every WHERE conjunct ends up in exactly one place, so none can be lost:
//
//   - an equality between join-compatible columns of two different tables is a
//     join-key candidate; the ones the join tree uses become join conditions and
//     the rest are applied as filters above the joins;
//   - a term that references one table is pushed into that table's scan;
//   - a term that references no column at all (1 = 0) is pushed into the first
//     scan, which filters the whole inner join equally well;
//   - anything else (a.x < b.y, a.x + 1 = b.y, an OR across tables) is applied
//     as a filter above the joins.
func buildMultiTablePlan(scans []*LogicalScan, where sql.Expr, st *symbolTable) (LogicalNode, error) {
	perTableFilters := make([]sql.Expr, len(scans))
	var edges []joinEdge
	var residual []sql.Expr
	for _, term := range flattenAnd(where) {
		if e, ok := joinEdgeOf(term, scans, st); ok {
			edges = append(edges, e)
			continue
		}
		tables := st.tablesOf(term)
		switch len(tables) {
		case 0:
			perTableFilters[0] = andExpr(perTableFilters[0], term)
		case 1:
			for t := range tables {
				perTableFilters[t] = andExpr(perTableFilters[t], term)
			}
		default:
			residual = append(residual, term)
		}
	}

	for i, f := range perTableFilters {
		if f != nil {
			scans[i].Predicate = f
		}
	}

	root, unused, err := buildJoinTree(scans, edges)
	if err != nil {
		return nil, err
	}
	for _, e := range unused {
		residual = append(residual, e.term)
	}
	if len(residual) > 0 {
		var pred sql.Expr
		for _, term := range residual {
			pred = andExpr(pred, term)
		}
		root = &LogicalFilter{Child: root, Predicate: pred}
	}
	return root, nil
}

// flattenAnd flattens a nested AND tree into a list of terms.
func flattenAnd(e sql.Expr) []sql.Expr {
	if e == nil {
		return nil
	}
	bin, ok := e.(*sql.BinaryExpr)
	if !ok || bin.Op != sql.OpAnd {
		return []sql.Expr{e}
	}
	return append(flattenAnd(bin.Left), flattenAnd(bin.Right)...)
}

// andExpr combines two expressions with AND (nil-safe).
func andExpr(a, b sql.Expr) sql.Expr {
	if a == nil {
		return b
	}
	return &sql.BinaryExpr{Op: sql.OpAnd, Left: a, Right: b}
}

// joinEdge is an equality between a column of one table and a column of
// another that the hash join can use as its key.
type joinEdge struct {
	leftTable, rightTable int
	leftCol, rightCol     string // output names
	term                  sql.Expr
}

// joinEdgeOf reports whether a resolved WHERE term can be a hash-join key:
// column = column across two different tables, both INT64 or both DATE. The
// hash join keys rows by an int64 image of the key, which identifies equal
// values only for those types; a STRING key's image is not its value, so an
// equality on strings is kept as an ordinary predicate instead.
func joinEdgeOf(term sql.Expr, scans []*LogicalScan, st *symbolTable) (joinEdge, bool) {
	bin, ok := term.(*sql.BinaryExpr)
	if !ok || bin.Op != sql.OpEQ {
		return joinEdge{}, false
	}
	l, lok := bin.Left.(*sql.ColumnRefExpr)
	r, rok := bin.Right.(*sql.ColumnRefExpr)
	if !lok || !rok {
		return joinEdge{}, false
	}
	lt, lfound := st.tableOfOut[l.Name]
	rt, rfound := st.tableOfOut[r.Name]
	if !lfound || !rfound || lt == rt {
		return joinEdge{}, false
	}
	ltype := fieldType(scans[lt].Schema, l.Name)
	rtype := fieldType(scans[rt].Schema, r.Name)
	if ltype != rtype || (ltype != exec.TypeInt64 && ltype != exec.TypeDate) {
		return joinEdge{}, false
	}
	return joinEdge{leftTable: lt, rightTable: rt, leftCol: l.Name, rightCol: r.Name, term: term}, true
}

func fieldType(schema exec.Schema, name string) exec.DataType {
	if i := schema.IndexOf(name); i >= 0 {
		return schema.Fields[i].Type
	}
	return 0
}

// buildJoinTree builds a left-deep join tree: starting from the first table, it
// repeatedly joins a table that some edge connects to the tables already
// joined. It returns the edges the tree did not use as join conditions — the
// caller must still apply them — and an error if a table cannot be connected,
// since cross joins are not supported.
func buildJoinTree(scans []*LogicalScan, edges []joinEdge) (LogicalNode, []joinEdge, error) {
	included := map[int]bool{0: true}
	used := make([]bool, len(edges))
	var root LogicalNode = scans[0]

	for len(included) < len(scans) {
		joined := false
		for i, e := range edges {
			if used[i] {
				continue
			}
			var newTable int
			var inTree, inNew string
			switch {
			case included[e.leftTable] && !included[e.rightTable]:
				newTable, inTree, inNew = e.rightTable, e.leftCol, e.rightCol
			case included[e.rightTable] && !included[e.leftTable]:
				newTable, inTree, inNew = e.leftTable, e.rightCol, e.leftCol
			default:
				continue
			}
			cond := &sql.BinaryExpr{
				Op:    sql.OpEQ,
				Left:  &sql.ColumnRefExpr{Name: inTree},
				Right: &sql.ColumnRefExpr{Name: inNew},
			}
			root = &LogicalJoin{Left: root, Right: scans[newTable], Condition: cond}
			included[newTable] = true
			used[i] = true
			joined = true
			break
		}
		if !joined {
			for i := range scans {
				if !included[i] {
					return nil, nil, fmt.Errorf("planner: no join condition connects table %q to the query; cross joins are not supported (join keys must be an equality between INT64 or DATE columns)", scans[i].TableName)
				}
			}
		}
	}

	var unused []joinEdge
	for i, e := range edges {
		if !used[i] {
			unused = append(unused, e)
		}
	}
	return root, unused, nil
}

// ---- SELECT list --------------------------------------------------------------

// selectItem is one SELECT-list entry as written, with * already expanded.
type selectItem struct {
	expr  sql.Expr // unresolved, as parsed
	alias string   // explicit alias, or the header derived from expr
	// userAlias is the explicit AS alias, which ORDER BY and HAVING may refer
	// to. Empty for unaliased items.
	userAlias string
}

func expandSelectList(cols []sql.SelectColumn, st *symbolTable) []selectItem {
	var items []selectItem
	for _, col := range cols {
		if _, ok := col.Expr.(*sql.StarExpr); ok {
			for t := range st.columns {
				for _, b := range st.columns[t] {
					items = append(items, selectItem{
						expr:  &sql.ColumnRefExpr{Table: st.tableNames[t], Name: b.source},
						alias: b.source,
					})
				}
			}
			continue
		}
		alias := col.Alias
		if alias == "" {
			alias = exprName(col.Expr)
		}
		items = append(items, selectItem{expr: col.Expr, alias: alias, userAlias: col.Alias})
	}
	return items
}

// selectQuery is a query whose FROM/WHERE (and, for aggregate queries, GROUP BY
// and HAVING) are planned, with its SELECT list resolved against that plan.
type selectQuery struct {
	child LogicalNode
	items []selectItem
	exprs []sql.Expr // items[i] resolved against child's output

	// orderExpr resolves an ORDER BY expression that is not a reference to a
	// SELECT-list entry against child's output. For aggregate queries it may
	// add aggregates to the aggregate node.
	orderExpr func(sql.Expr) (sql.Expr, error)
}

func buildScalarSelect(child LogicalNode, items []selectItem, st *symbolTable) (*selectQuery, error) {
	q := &selectQuery{child: child, items: items}
	for _, it := range items {
		e, err := st.resolveScalar(it.expr, "the SELECT list of a query without GROUP BY")
		if err != nil {
			return nil, err
		}
		q.exprs = append(q.exprs, e)
	}
	q.orderExpr = func(e sql.Expr) (sql.Expr, error) { return st.resolveScalar(e, "ORDER BY") }
	return q, nil
}

// finish adds the SELECT-list projection, DISTINCT, ORDER BY and LIMIT/OFFSET.
func (q *selectQuery) finish(stmt *sql.SelectStmt) (LogicalNode, error) {
	// Internal names must be unique so that operators above the projection can
	// address each column by name. A duplicate user-facing name (two columns
	// both called "shared") gets an internal name and is renamed back at the top.
	used := make(map[string]bool)
	var items []ProjectItem
	for i, it := range q.items {
		name := it.alias
		if used[name] {
			name = uniqueName("_col", i, used)
		}
		used[name] = true
		items = append(items, ProjectItem{Alias: name, Expr: q.exprs[i]})
	}
	visible := len(items)

	var orderBy []sql.OrderByItem
	for _, ob := range stmt.OrderBy {
		idx, err := q.orderTarget(ob.Expr, items[:visible])
		if err != nil {
			return nil, err
		}
		if idx < 0 {
			resolved, err := q.orderExpr(ob.Expr)
			if err != nil {
				return nil, err
			}
			idx = matchItem(resolved, items)
			if idx < 0 {
				if stmt.Distinct {
					return nil, fmt.Errorf("planner: for SELECT DISTINCT, ORDER BY expression %s must appear in the select list", sql.FormatExpr(ob.Expr))
				}
				name := uniqueName("_order", len(items), used)
				used[name] = true
				items = append(items, ProjectItem{Alias: name, Expr: resolved})
				idx = len(items) - 1
			}
		}
		orderBy = append(orderBy, sql.OrderByItem{
			Expr:       &sql.ColumnRefExpr{Name: items[idx].Alias},
			Descending: ob.Descending,
		})
	}

	// Typed only now: resolving ORDER BY can add aggregates, which changes the
	// aggregate's output schema.
	childSchema := q.child.OutputSchema()
	for i := range items {
		items[i].Type = resolveExprType(items[i].Expr, childSchema)
	}

	var root LogicalNode = q.child
	if !isIdentityProject(items, childSchema) {
		root = &LogicalProject{Child: root, Exprs: items}
	}
	if stmt.Distinct {
		root = &LogicalDistinct{Child: root}
	}
	if len(orderBy) > 0 {
		root = &LogicalSort{Child: root, OrderBy: orderBy}
	}
	if stmt.Limit != nil || stmt.Offset != nil {
		lim := &LogicalLimit{Child: root, Count: -1}
		if stmt.Limit != nil {
			lim.Count = *stmt.Limit
		}
		if stmt.Offset != nil {
			lim.Offset = *stmt.Offset
		}
		root = lim
	}

	renamed := false
	for i := 0; i < visible; i++ {
		if items[i].Alias != q.items[i].alias {
			renamed = true
		}
	}
	if renamed || len(items) > visible {
		outSchema := root.OutputSchema()
		final := make([]ProjectItem, visible)
		for i := 0; i < visible; i++ {
			final[i] = ProjectItem{
				Alias: q.items[i].alias,
				Expr:  &sql.ColumnRefExpr{Name: items[i].Alias},
				Type:  outSchema.Fields[i].Type,
			}
		}
		root = &LogicalProject{Child: root, Exprs: final}
	}
	return root, nil
}

// uniqueName returns prefix_n, or the first prefix_n_k not in used.
func uniqueName(prefix string, n int, used map[string]bool) string {
	name := fmt.Sprintf("%s_%d", prefix, n)
	for k := 1; used[name]; k++ {
		name = fmt.Sprintf("%s_%d_%d", prefix, n, k)
	}
	return name
}

// orderTarget resolves an ORDER BY expression that names a SELECT-list entry:
// a 1-based position, or an unqualified name equal to an output column's
// header. It returns -1 when the expression names no entry and must instead be
// evaluated against the query's input.
func (q *selectQuery) orderTarget(e sql.Expr, items []ProjectItem) (int, error) {
	switch x := e.(type) {
	case *sql.IntLiteral:
		if x.Value < 1 || x.Value > int64(len(items)) {
			return 0, fmt.Errorf("planner: ORDER BY position %d is not in the select list", x.Value)
		}
		return int(x.Value - 1), nil
	case *sql.ColumnRefExpr:
		if x.Table != "" {
			return -1, nil
		}
		found := -1
		for i := range items {
			if q.items[i].alias != x.Name {
				continue
			}
			if found >= 0 && sql.FormatExpr(q.exprs[found]) != sql.FormatExpr(q.exprs[i]) {
				return 0, fmt.Errorf("planner: ORDER BY %q is ambiguous", x.Name)
			}
			if found < 0 {
				found = i
			}
		}
		return found, nil
	}
	return -1, nil
}

// matchItem returns the index of the projection item whose expression is e, or
// -1. Resolved expressions use plan-unique column names, so equal renderings
// mean equal expressions.
func matchItem(e sql.Expr, items []ProjectItem) int {
	want := sql.FormatExpr(e)
	for i, it := range items {
		if sql.FormatExpr(it.Expr) == want {
			return i
		}
	}
	return -1
}

// isIdentityProject reports whether projecting items over a child with the
// given schema would reproduce the child's output exactly, in which case the
// projection is omitted.
func isIdentityProject(items []ProjectItem, schema Schema) bool {
	if len(items) != len(schema.Fields) {
		return false
	}
	for i, it := range items {
		ref, ok := it.Expr.(*sql.ColumnRefExpr)
		if !ok || ref.Table != "" || ref.Name != schema.Fields[i].Name || it.Alias != schema.Fields[i].Name {
			return false
		}
	}
	return true
}

// ---- Aggregation ---------------------------------------------------------------

// aggBuilder plans an aggregate query: it collects the aggregates the query
// computes and rewrites SELECT, HAVING and ORDER BY expressions to read the
// aggregate's output columns.
type aggBuilder struct {
	st        *symbolTable
	agg       *LogicalAggregate
	groupCols map[string]bool // resolved GROUP BY column names
	outNames  map[string]bool // names already used by the aggregate's output
}

func buildAggregation(child LogicalNode, stmt *sql.SelectStmt, items []selectItem, st *symbolTable) (*selectQuery, error) {
	ab := &aggBuilder{
		st:        st,
		agg:       &LogicalAggregate{Child: child},
		groupCols: make(map[string]bool),
		outNames:  make(map[string]bool),
	}

	for _, gb := range stmt.GroupBy {
		if containsAggregate(gb) {
			return nil, fmt.Errorf("planner: aggregate functions are not allowed in GROUP BY")
		}
		cr, ok := gb.(*sql.ColumnRefExpr)
		if !ok {
			return nil, fmt.Errorf("planner: GROUP BY only supports column references")
		}
		resolved, err := st.resolveColumns(cr)
		if err != nil {
			return nil, err
		}
		name := resolved.(*sql.ColumnRefExpr).Name
		ab.agg.GroupBy = append(ab.agg.GroupBy, resolved)
		ab.groupCols[name] = true
		ab.outNames[name] = true
	}

	q := &selectQuery{items: items}
	for _, it := range items {
		var e sql.Expr
		var err error
		if ae, ok := it.expr.(*sql.AggFuncExpr); ok {
			// A top-level aggregate gets its own output column named after the
			// SELECT entry, so the common SELECT group-cols, aggregates shape
			// needs no projection above the aggregate.
			e, err = ab.addAggregate(ae, it.alias)
		} else {
			e, err = ab.rewrite(it.expr, nil)
		}
		if err != nil {
			return nil, err
		}
		q.exprs = append(q.exprs, e)
	}

	var root LogicalNode = ab.agg
	if stmt.Having != nil {
		pred, err := ab.rewrite(stmt.Having, func(name string) (sql.Expr, bool) {
			for i, it := range items {
				if it.userAlias == name {
					return q.exprs[i], true
				}
			}
			return nil, false
		})
		if err != nil {
			return nil, err
		}
		root = &LogicalFilter{Child: root, Predicate: pred}
	}

	q.child = root
	q.orderExpr = func(e sql.Expr) (sql.Expr, error) { return ab.rewrite(e, nil) }
	return q, nil
}

// rewrite resolves e as an expression over the aggregate's output: each
// aggregate becomes a reference to an aggregate output column (added if no
// equal aggregate exists yet), and each column reference must name a GROUP BY
// column. alias, when non-nil, is consulted for an unqualified name that is not
// a column of any table, which is how HAVING refers to SELECT-list aliases.
func (ab *aggBuilder) rewrite(e sql.Expr, alias func(string) (sql.Expr, bool)) (sql.Expr, error) {
	return rewriteExpr(e, func(n sql.Expr) (sql.Expr, bool, error) {
		switch x := n.(type) {
		case *sql.AggFuncExpr:
			out, err := ab.findOrAddAggregate(x)
			return out, true, err
		case *sql.ColumnRefExpr:
			b, err := ab.st.resolve(x)
			if err != nil {
				if alias != nil && x.Table == "" && len(ab.st.byName[x.Name]) == 0 {
					if sub, ok := alias(x.Name); ok {
						return sub, true, nil
					}
				}
				return nil, true, fmt.Errorf("planner: %w", err)
			}
			if !ab.groupCols[b.out] {
				return nil, true, fmt.Errorf("planner: column %q must appear in the GROUP BY clause or be used in an aggregate function", sql.FormatExpr(x))
			}
			return &sql.ColumnRefExpr{Name: b.out}, true, nil
		}
		return nil, false, nil
	})
}

// addAggregate appends an aggregate output column for ae, preferring name as
// its column name, and returns a reference to it.
func (ab *aggBuilder) addAggregate(ae *sql.AggFuncExpr, name string) (sql.Expr, error) {
	fn := strings.ToUpper(ae.Func)
	if ae.Distinct && fn != "COUNT" {
		return nil, fmt.Errorf("planner: %s(DISTINCT ...) is not supported; only COUNT(DISTINCT col) is implemented", ae.Func)
	}
	var arg sql.Expr
	if ae.Arg != nil {
		if _, star := ae.Arg.(*sql.StarExpr); star {
			if fn != "COUNT" || ae.Distinct {
				return nil, fmt.Errorf("planner: %s(*) is not supported", ae.Func)
			}
		} else {
			if containsAggregate(ae.Arg) {
				return nil, fmt.Errorf("planner: aggregate function calls cannot be nested")
			}
			var err error
			if arg, err = ab.st.resolveColumns(ae.Arg); err != nil {
				return nil, err
			}
		}
	}

	if name == "" || ab.outNames[name] {
		name = uniqueName("_agg", len(ab.agg.Aggs), ab.outNames)
	}
	ab.outNames[name] = true

	item := AggItem{Func: fn, Alias: name, Distinct: ae.Distinct}
	switch a := arg.(type) {
	case nil:
		// COUNT(*) — no source column.
	case *sql.ColumnRefExpr:
		item.ColName = a.Name
	default:
		// Computed argument: the physical planner materialises it into this
		// synthetic column before aggregating (buildPreProjection).
		item.ColName = fmt.Sprintf("_agg_arg_%d", len(ab.agg.Aggs))
		item.AggExpr = a
	}
	ab.agg.Aggs = append(ab.agg.Aggs, item)
	return &sql.ColumnRefExpr{Name: name}, nil
}

// findOrAddAggregate returns a reference to an existing aggregate output equal
// to ae, adding a hidden one when there is none.
func (ab *aggBuilder) findOrAddAggregate(ae *sql.AggFuncExpr) (sql.Expr, error) {
	var argKey string
	if ae.Arg != nil {
		if _, star := ae.Arg.(*sql.StarExpr); !star && !containsAggregate(ae.Arg) {
			resolved, err := ab.st.resolveColumns(ae.Arg)
			if err != nil {
				return nil, err
			}
			argKey = sql.FormatExpr(resolved)
		}
	}
	fn := strings.ToUpper(ae.Func)
	for _, a := range ab.agg.Aggs {
		if a.Func != fn || a.Distinct != ae.Distinct {
			continue
		}
		key := a.ColName
		if a.AggExpr != nil {
			key = sql.FormatExpr(a.AggExpr)
		}
		if key == argKey {
			return &sql.ColumnRefExpr{Name: a.Alias}, nil
		}
	}
	return ab.addAggregate(ae, "")
}

func isStarExpr(e sql.Expr) bool {
	_, ok := e.(*sql.StarExpr)
	return ok
}

// resolveExprType infers the logical type of a resolved expression evaluated
// over schema.
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
	case *sql.BoolLiteral, *sql.IsNullExpr, *sql.BetweenExpr, *sql.InExpr, *sql.LikeExpr:
		return TypeBool
	case *sql.UnaryExpr:
		if e.Op == sql.OpNot {
			return TypeBool
		}
		return resolveExprType(e.Expr, schema)
	case *sql.BinaryExpr:
		switch e.Op {
		case sql.OpAdd, sql.OpSub, sql.OpMul, sql.OpDiv:
			l := resolveExprType(e.Left, schema)
			r := resolveExprType(e.Right, schema)
			if l == TypeFloat64 || r == TypeFloat64 {
				return TypeFloat64
			}
			return l
		}
		return TypeBool
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
	case *sql.CaseExpr:
		// Determine result type from the first WHEN result.
		for _, w := range e.Whens {
			t := resolveExprType(w.Result, schema)
			if t != 0 {
				return t
			}
		}
	}
	return TypeInt64
}
