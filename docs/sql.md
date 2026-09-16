# Supported SQL

```sql
SELECT [DISTINCT] expr [[AS] alias], ...        -- or SELECT *
FROM table [[AS] alias] [, table [[AS] alias] ...]
[WHERE condition]
[GROUP BY column, ...]
[HAVING condition]
[ORDER BY expr [ASC|DESC], ...]
[LIMIT n] [OFFSET m]
[;]
```

A query is a single `SELECT`. Anything after it other than one `;` is a parse
error rather than being ignored: `UNION`, `INTERSECT` and `EXCEPT` are rejected
by name, and any other trailing text is reported as unexpected.

## Column references

- Columns may be written unqualified (`col`), qualified by table (`orders.col`),
  or qualified by alias (`o.col`).
- An unqualified name that exists in more than one `FROM` table is an error
  wherever it appears — `SELECT`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, or
  an aggregate argument. So is a reference to a column or table qualifier that
  does not exist.
- Same-named columns from different tables stay distinct:
  `SELECT a.id, b.id FROM a, b WHERE ...` returns both values. Self-joins need
  distinct aliases (`FROM t x, t y`).
- An unaliased column's header is the column name without its qualifier;
  `SELECT *` expands to every column of every `FROM` table, in order.

## Joins

- Inner joins are written as `FROM t1, t2 WHERE ...`. Every table must be
  connected to the others by at least one **join key**: an equality between an
  `INT64` column of one table and an `INT64` column of another (or `DATE` and
  `DATE`). A table with no such connection is rejected; cross joins are not
  supported.
- Every `WHERE` term is applied. Terms over one table are pushed into that
  table's scan; join keys the join tree does not need, equalities on other types,
  non-equality comparisons across tables (`a.x < b.y`), `OR`s spanning tables and
  constant terms (`1 = 0`) are all applied to the joined rows.

## Predicates and expressions

- Comparisons `=, <>, <, <=, >, >=` between numbers (an `INT64`/`FLOAT64` mix is
  compared as `FLOAT64`, so `int_col >= 2.5` is exact), between dates, and
  between a `DATE` and a `'YYYY-MM-DD'` string or an integer day number.
  `STRING` columns support `=` and `<>` against a string literal, `LIKE` and
  `IN`. Any other comparison (strings with `<`, a date with an integer column,
  a string column with a string column) is a planning error.
- `AND`, `OR`, `NOT` follow SQL three-valued logic: `NULL AND FALSE` is `FALSE`,
  `NULL OR FALSE` is `NULL`, and a row is kept only when the whole condition is
  `TRUE`.
- `BETWEEN` / `NOT BETWEEN`, `LIKE` / `NOT LIKE` (with `%` and `_`),
  `IS NULL` / `IS NOT NULL`.
- `IN` / `NOT IN` take a list of literals, including negative numbers and
  `NULL`. List entries are converted to the tested value's type: `int_col IN
  (2.0)` matches 2, `int_col IN (2.5)` matches nothing, and a date column accepts
  `'YYYY-MM-DD'` strings. A `NULL` in the list makes a non-match `NULL`, so
  `x NOT IN (2, NULL)` selects no rows. An entry that is not a literal, or cannot
  be compared with the tested type, is an error.
- Arithmetic `+ - * /` and unary minus. `INT64 / INT64` is integer division;
  division by zero yields `NULL`.
- `CASE WHEN ... THEN ... [ELSE ...] END`; all branches must share a type.

## Aggregation

- Aggregate functions: `COUNT(*)`, `COUNT(expr)`, `COUNT(DISTINCT col)`, `SUM`,
  `AVG`, `MIN`, `MAX`. Arguments may be expressions (`SUM(price * (1 - disc))`).
  `DISTINCT` is supported only for `COUNT`.
- A query aggregates when it has `GROUP BY`, or an aggregate anywhere in its
  `SELECT` list or `HAVING` clause.
- `GROUP BY` takes column references.
- The output has exactly the `SELECT`-list columns, in the order written and
  under their aliases. `SELECT` entries may be expressions over aggregates and
  grouping columns (`SUM(x) + 1`, `k * 100 + COUNT(*)`). A column used outside an
  aggregate must be a `GROUP BY` column; otherwise the query is rejected.
- `HAVING` may use any aggregate (whether or not it is selected), grouping
  columns, and `SELECT`-list aliases, in any expression — including inside
  `CASE`, `IN` and `NOT`. `HAVING` in a query that does not aggregate is an error.
- Aggregates are rejected in `WHERE` and `GROUP BY`, and cannot be nested.

## ORDER BY, LIMIT, OFFSET

- An `ORDER BY` item may be an output column's name or alias, a 1-based column
  position, or an expression over the query's input — in an aggregate query, over
  grouping columns and aggregates. It need not appear in the `SELECT` list, except
  under `SELECT DISTINCT`. `NULL`s sort first.
- `LIMIT n` and `OFFSET m` take non-negative integers and may appear in either
  order or alone.

## Not supported

Each of these is rejected with an error rather than ignored:

- `UNION` / `INTERSECT` / `EXCEPT`, subqueries, explicit `JOIN ... ON` syntax,
  outer and cross joins
- `GROUP BY` expressions or `SELECT`-list aliases (`GROUP BY x + 1`,
  `SELECT c AS k ... GROUP BY k`)
- `DISTINCT` inside aggregates other than `COUNT`
- Scalar functions other than the aggregates above
