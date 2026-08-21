# Supported SQL

```sql
SELECT expr [AS alias], ...
FROM table [alias] [, table2 [alias2], ...]
[WHERE condition]
[GROUP BY col, ...]
[HAVING condition]
[ORDER BY col [ASC|DESC], ...]
[LIMIT n]

-- Joins: implicit inner join via FROM t1, t2 WHERE t1.key = t2.key
-- Column references: unqualified (col), qualified (table.col), aliased (alias.col)
-- Ambiguous columns: error when an unqualified name exists in multiple tables
-- Cross joins: explicitly rejected — every table must connect via an equi-join condition
-- Aggregate functions: COUNT(*), COUNT(col), COUNT(DISTINCT col), SUM, AVG, MIN, MAX
-- HAVING: supports aggregate expressions directly (HAVING COUNT(*) > 5) and output aliases
-- Predicates: =, <>, <, <=, >, >=, AND, OR, NOT, BETWEEN, IN, LIKE, IS NULL
-- Expressions: arithmetic (+, -, *, /), CASE WHEN (string and numeric results), DISTINCT, unary minus
```
