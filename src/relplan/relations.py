"""Bag-preserving relational execution with a metered plan search.

The query entry :func:`run_query` enumerates semantically equivalent *complete*
plans under a bounded search workload, picks one deterministically and executes
it under a separate execution budget.  Two limits are kept apart on purpose:

* ``search_limited``  - enumeration hit its limit but a complete plan was found
  and executed successfully;
* ``no_plan``         - no complete plan was available;
* ``execution_limited`` - the selected plan ran past the execution budget; any
  rows produced before the limit are partial and are never returned.

Join predicates and post-join filters are never rewritten: a join condition is
only evaluated at its original join position and a filter is only pushed when
the conservative inner-join/left-scan rule proves the move is valid.  Left
joins therefore keep unmatched rows with SQL-style NULL padding, including the
case where the right-hand table is empty, and duplicated detail rows take part
in the aggregation as many times as they appear.
"""
import json
from copy import deepcopy

DEFAULT_SEARCH_BUDGET = 256
DEFAULT_EXECUTION_BUDGET = 100000


class ExecutionLimitError(ValueError):
    """Raised when a plan exceeds its execution budget.

    The exception carries the work already performed, the node path where the
    limit triggered and a partial profile.  It deliberately never carries
    result rows: an interrupted plan has no complete answer.
    """

    def __init__(self, message, work, budget, path, profile):
        super().__init__(message)
        self.work = work
        self.budget = budget
        self.path = path
        self.profile = profile


class _SearchLimit(Exception):
    """Internal: enumeration consumed its search budget."""


def predicate(p, row):
    if not isinstance(p, list) or not p:
        raise ValueError("invalid predicate")
    op = p[0]
    if op == "col":
        return row.get(p[1])
    if op == "lit":
        return p[1]
    if op == "and":
        a, b = predicate(p[1], row), predicate(p[2], row)
        return False if a is False or b is False else None if a is None or b is None else bool(a and b)
    if op not in {"eq", "lt", "gt"}:
        raise ValueError("unsupported predicate")
    a, b = predicate(p[1], row), predicate(p[2], row)
    if a is None or b is None:
        return None
    return a == b if op == "eq" else a < b if op == "lt" else a > b


def columns(pred):
    """Columns referenced by a predicate."""
    if pred[0] == "col":
        return {pred[1]}
    if pred[0] == "lit":
        return set()
    return set().union(*(columns(x) for x in pred[1:] if isinstance(x, list)))


def source_tables(node):
    """Base-table names referenced anywhere below a plan node."""
    op = node["op"]
    if op == "scan":
        return {node["table"]}
    if op in ("filter", "project"):
        return source_tables(node["input"])
    if op == "join":
        return source_tables(node["left"]) | source_tables(node["right"])
    raise ValueError("unsupported relational operator")


def _predicate_tables(pred):
    return {c.split(".", 1)[0] for c in columns(pred)}


def canonical(plan):
    """Stable textual key used for de-duplication and deterministic ties."""
    return json.dumps(plan, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Plan enumeration
# --------------------------------------------------------------------------- #

def _push_legal(join_node, pred):
    """The single, conservative pushdown rule inherited from ``optimize``.

    A filter sitting directly above an inner join may move onto the immediate
    left input only when that input is a scan and every referenced column is
    produced by it.  Left joins never qualify: pushing would drop the NULL
    padded rows that the outer join must preserve.
    """
    left = join_node["left"]
    if join_node.get("kind", "inner") != "inner" or left["op"] != "scan":
        return False
    refs = columns(pred)
    return bool(refs) and all(c.startswith(left["table"] + ".") for c in refs)


def _flatten_joins(node):
    """Split a left-deep join tree into a base subtree and ordered steps.

    Returns ``(base, steps, left_deep)`` with ``steps`` ordered top-down: the
    first step is the outermost join's right input, the last one sits nearest
    the base scan.  A chain is left-deep only when every join's right input is
    itself a non-join subtree; rebuilding such a chain reproduces the original
    tree.  Bushy trees report ``left_deep=False`` and are never reordered,
    because flatten/rebuild would invert the nesting and move joins across
    outer-join boundaries.
    """
    collected = []
    left_deep = True
    while node["op"] == "join":
        if node["right"]["op"] == "join":
            left_deep = False
        collected.append((node.get("kind", "inner"), node["predicate"], node["right"]))
        node = node["left"]
    return node, collected[::-1], left_deep


def _rebuild_joins(base, steps):
    """Rebuild a left-deep tree from top-down steps: the base scan is joined
    by the first step, then each following step becomes the next outer node."""
    cur = base
    for kind, pred, right in steps:
        cur = {"op": "join", "kind": kind, "predicate": pred, "left": cur, "right": right}
    return cur


def _legal_permutations(base, segment):
    """All legal top-down orderings of an inner-join segment.

    A step may occupy a position only when, apart from the columns its own
    right subtree supplies, every table its predicate mentions is already
    available on the left side there.  The original ordering always satisfies
    this, so at least one permutation is returned.
    """
    available = source_tables(base)
    found = []

    def backtrack(prefix, avail, remaining):
        if not remaining:
            found.append(list(prefix))
            return
        for idx, step in enumerate(remaining):
            _, pred, right = step
            needed = _predicate_tables(pred) - source_tables(right)
            if needed <= avail:
                backtrack(prefix + [step], avail | source_tables(right),
                          remaining[:idx] + remaining[idx + 1:])

    backtrack([], available, segment)
    found.sort(key=lambda perm: canonical(_rebuild_joins(base, perm)))
    return found


def _join_orderings(base, steps, left_deep):
    """Enumerate step orderings for a flattened join chain.

    Reordering is restricted to left-deep, inner-join-only chains:

    * a non-left-deep (bushy) chain would be rebuilt with inverted nesting,
      which can move joins across outer-join boundaries;
    * any left join freezes the chain because an inner join on either side
      moving across it would change which rows are preserved;
    * within a legal chain an order whose predicate tables are not yet
      available is rejected instead of letting an unknown condition match.
    """
    if not left_deep or any(kind != "inner" for kind, _, _ in steps):
        return [list(steps)]
    return _legal_permutations(base, steps)


def _enumerate(node, meter, budget):
    """Yield every complete semantic variant of ``node`` (post-order)."""
    meter[0] += 1
    if meter[0] > budget:
        raise _SearchLimit()
    op = node["op"]
    if op == "scan":
        yield deepcopy(node)
    elif op == "filter":
        for variant in _enumerate(node["input"], meter, budget):
            # Position rule: the post-join/post-input filter stays on top.
            yield {"op": "filter", "predicate": node["predicate"], "input": variant}
            # The one legal relocation, judged against the original join node.
            if variant["op"] == "join" and _push_legal(variant, node["predicate"]):
                pushed = deepcopy(variant)
                left = pushed["left"]
                pushed["left"] = {"op": "filter", "predicate": node["predicate"], "input": left}
                yield pushed
    elif op == "project":
        for variant in _enumerate(node["input"], meter, budget):
            yield {"op": "project", "columns": node["columns"], "input": variant}
    elif op == "join":
        left_variants = list(_enumerate(node["left"], meter, budget))
        right_variants = list(_enumerate(node["right"], meter, budget))
        seen_keys = set()
        for lv in left_variants:
            for rv in right_variants:
                rebuilt = {"op": "join", "kind": node.get("kind", "inner"),
                           "predicate": node["predicate"], "left": lv, "right": rv}
                base, steps, left_deep = _flatten_joins(rebuilt)
                for ordered in _join_orderings(base, steps, left_deep):
                    candidate = _rebuild_joins(base, ordered)
                    key = canonical(candidate)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        yield candidate
    else:
        raise ValueError("unsupported relational operator")


# --------------------------------------------------------------------------- #
# Cost model (purely table cardinalities; deterministic, no promises of optimum)
# --------------------------------------------------------------------------- #

def _table_size(name, tables):
    rows = tables.get(name)
    return len(rows) if isinstance(rows, list) else None


def estimate(plan, tables):
    """Return ``(total_work, output_rows, basis)``; ``None`` means unknown."""
    basis = []
    op = plan["op"]
    if op == "scan":
        n = _table_size(plan["table"], tables)
        if n is None:
            basis.append("scan %s: cardinality unknown" % plan["table"])
        else:
            basis.append("scan %s: %d rows" % (plan["table"], n))
        return n, n, basis
    if op in ("filter", "project"):
        work, card, sub = estimate(plan["input"], tables)
        basis.extend(sub)
        basis.append("%s: selectivity unknown, cardinality unchanged" % op)
        if work is not None and card is not None:
            work += card
        return work, card, basis
    if op == "join":
        lw, lc, sub = estimate(plan["left"], tables)
        rw, rc, sub2 = estimate(plan["right"], tables)
        basis.extend(sub)
        basis.extend(sub2)
        if lc is None or rc is None:
            basis.append("join: cardinality unknown, comparison work unknown")
            return None, None, basis
        comparisons = lc * rc
        work = comparisons
        if lw is not None:
            work += lw
        if rw is not None:
            work += rw
        # Worst-case bag cardinality; left joins can only add NULL-padded rows.
        card = lc * rc
        basis.append("join: %d x %d = %d comparisons" % (lc, rc, comparisons))
        return work, card, basis
    raise ValueError("unsupported relational operator")


# --------------------------------------------------------------------------- #
# Instrumented execution
# --------------------------------------------------------------------------- #

def execute_profile(plan, tables, budget=DEFAULT_EXECUTION_BUDGET):
    """Execute a plan while metering the actual work.

    Work is charged per emitted output row at every node and per join
    predicate comparison.  A join whose predicate never matches still pays for
    every comparison, which is observable in the returned profile.  Exceeding
    ``budget`` raises :class:`ExecutionLimitError`; partial rows are discarded.
    """
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ValueError("budget must be a non-negative integer")
    work = [0]
    profile = []

    def charge(amount, path, node_op):
        work[0] += amount
        if work[0] > budget:
            raise ExecutionLimitError(
                "execution budget exceeded at %s (%s): %d > %d" % (path, node_op, work[0], budget),
                work[0], budget, path, list(profile))

    def visit(p, depth, path):
        if depth > 30:
            raise ValueError("plan depth limit")
        op = p["op"]
        comparisons = 0
        if op == "scan":
            name = p["table"]
            if name not in tables:
                raise ValueError("unknown table")
            rows = [{name + "." + k: v for k, v in r.items()} for r in tables[name]]
        elif op == "filter":
            rows = [r for r in visit(p["input"], depth + 1, path + "/input")
                    if predicate(p["predicate"], r) is True]
        elif op == "project":
            rows = [{k: r.get(k) for k in p["columns"]} for r in visit(p["input"], depth + 1, path + "/input")]
        elif op == "join":
            if p.get("kind", "inner") not in {"inner", "left"}:
                raise ValueError("unsupported join")
            left = visit(p["left"], depth + 1, path + "/left")
            right = visit(p["right"], depth + 1, path + "/right")
            right_keys = set().union(*(r.keys() for r in right)) if right else set(p.get("right_columns", []))
            rows = []
            join_path = path or "join"
            for l in left:
                hits = []
                for r in right:
                    charge(1, join_path, "join")  # every comparison is work, match or not
                    comparisons += 1
                    if predicate(p["predicate"], dict(l, **r)) is True:
                        hits.append(dict(l, **r))
                rows.extend(hits)
                if not hits and p.get("kind") == "left":
                    rows.append(dict(l, **{k: None for k in right_keys}))
        else:
            raise ValueError("unsupported relational operator")
        charge(len(rows), path or op, op)  # every emitted row is work
        profile.append({"path": path or op, "op": op, "rows_out": len(rows),
                        "comparisons": comparisons, "work": work[0]})
        return rows

    result_rows = visit(plan, 0, "")
    return {"rows": result_rows, "work": work[0], "budget": budget,
            "limit_triggered": False, "profile": profile}


def execute(plan, tables, budget=DEFAULT_EXECUTION_BUDGET):
    """Backward-compatible entry: return the result rows or raise ValueError."""
    return execute_profile(plan, tables, budget)["rows"]


# --------------------------------------------------------------------------- #
# Search and the query entry point
# --------------------------------------------------------------------------- #

def search_plan(plan, search_budget=DEFAULT_SEARCH_BUDGET, tables=None):
    """Enumerate complete plans under a bounded, deterministic search.

    Returns the examined candidates (best first), search accounting and a
    truncation flag.  When the budget runs out mid-enumeration every candidate
    already received is still a complete plan; ``truncated`` reports that more
    candidates may exist but were never examined.
    """
    if not isinstance(search_budget, int) or isinstance(search_budget, bool) or search_budget < 0:
        raise ValueError("search_budget must be a non-negative integer")
    meter = [0]
    examined = {}
    truncated = False
    try:
        for candidate in _enumerate(deepcopy(plan), meter, search_budget):
            key = canonical(candidate)
            if key in examined:
                continue
            work, card, basis = estimate(candidate, tables if tables is not None else {})
            examined[key] = {"plan": candidate, "key": key,
                             "estimated_work": work, "estimated_rows": card, "basis": basis}
            meter[0] += 1
            if meter[0] > search_budget:
                raise _SearchLimit()
    except _SearchLimit:
        truncated = True
    candidates = sorted(examined.values(),
                        key=lambda c: (c["estimated_work"] is None, c["estimated_work"], c["key"]))
    return {
        "candidates": candidates,
        "search": {"budget": search_budget, "work": meter[0], "examined": len(candidates),
                   "truncated": truncated,
                   "remaining_candidates": None if truncated else 0},
    }


def run_query(plan, tables, search_budget=DEFAULT_SEARCH_BUDGET,
              execution_budget=DEFAULT_EXECUTION_BUDGET):
    """Search, select and execute; report what was chosen and where limits hit.

    Status values:

    * ``completed``        - the whole search space was examined and the plan
      finished inside the execution budget;
    * ``search_limited``   - the search budget ran out first, but a complete
      candidate was selected and executed successfully;
    * ``no_plan``          - the search budget allowed no complete candidate;
    * ``execution_limited`` - the selected plan exceeded the execution budget;
      no partial rows are returned.
    """
    found = search_plan(plan, search_budget=search_budget, tables=tables)
    candidates = found["candidates"]
    search_info = found["search"]
    if not candidates:
        return {"status": "no_plan", "selected_plan": None, "selected_key": None,
                "search": search_info, "candidates": []}
    selected = candidates[0]
    report = {
        "status": "search_limited" if search_info["truncated"] else "completed",
        "selected_plan": selected["plan"], "selected_key": selected["key"],
        "search": search_info, "candidates": candidates,
    }
    try:
        executed = execute_profile(selected["plan"], tables, execution_budget)
    except ExecutionLimitError as exc:
        report["status"] = "execution_limited"
        report["executed"] = {
            "rows": None, "work": exc.work, "budget": exc.budget,
            "limit_triggered": True, "limit_path": exc.path, "profile": exc.profile,
        }
        return report
    report["executed"] = executed
    return report


def optimize(plan, limit=256):
    """Legacy single-rule optimizer; behavior and budget exception unchanged."""
    p = deepcopy(plan)
    seen = [0]

    def visit(x):
        seen[0] += 1
        if seen[0] > limit:
            raise ValueError("optimization budget exceeded")
        for k in ["input", "left", "right"]:
            if k in x:
                x[k] = visit(x[k])
        if x["op"] == "filter" and x["input"]["op"] == "join":
            join_node = x["input"]
            if _push_legal(join_node, x["predicate"]):
                left = join_node["left"]
                join_node["left"] = {"op": "filter", "predicate": x["predicate"], "input": left}
                return join_node
        return x

    return visit(p)
