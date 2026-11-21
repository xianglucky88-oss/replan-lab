import json
import os
import subprocess
import sys

import pytest

from relplan import relations, api


def bag(rows):
    return sorted(json.dumps(r, sort_keys=True) for r in rows)


def join_scan(left, right, kind, pred):
    return {"op": "join", "kind": kind, "predicate": pred,
            "left": {"op": "scan", "table": left}, "right": {"op": "scan", "table": right}}


# --------------------------------------------------------------------------- #
# Daily transaction scenario: left joins, empty dimension, duplicate details
# --------------------------------------------------------------------------- #

def daily_tables():
    return {
        "txn": [{"id": 1, "cat": "a"}, {"id": 2, "cat": "b"},
                {"id": 3, "cat": None}, {"id": 4, "cat": "a"}],
        # txn 1 appears twice: duplicated detail rows must count twice
        "detail": [{"txn_id": 1, "amt": 10}, {"txn_id": 1, "amt": 10},
                   {"txn_id": 2, "amt": 5}],
        # empty dimension: left join must still keep every transaction
        "dim_cat": [],
    }


def daily_plan():
    return {"op": "project", "columns": ["txn.id", "detail.amt", "dim_cat.name"],
            "input":
            {"op": "join", "kind": "left",
             "predicate": ["eq", ["col", "txn.cat"], ["col", "dim_cat.code"]],
             "left": join_scan("txn", "detail", "left",
                               ["eq", ["col", "txn.id"], ["col", "detail.txn_id"]]),
             "right": {"op": "scan", "table": "dim_cat"}}}


def test_left_join_keeps_unclassified_transactions_empty_dim_and_duplicates():
    tables = daily_tables()
    report = relations.run_query(daily_plan(), tables)
    assert report["status"] == "completed"
    rows = report["executed"]["rows"]
    ids = [r["txn.id"] for r in rows]
    # 4 transactions, txn 1 duplicated by its two detail rows -> 5 output rows
    assert sorted(ids) == [1, 1, 2, 3, 4]
    # unclassified txn 3 survives both left joins with NULL padding
    unclassified = [r for r in rows if r["txn.id"] == 3][0]
    assert unclassified["detail.amt"] is None and unclassified["dim_cat.name"] is None
    # every dimension value is NULL because the dimension table is empty, but
    # none of the left-side rows disappeared
    assert all(r["dim_cat.name"] is None for r in rows)
    # duplicated detail participates by its actual multiplicity
    assert [r["detail.amt"] for r in rows if r["txn.id"] == 1] == [10, 10]


def test_every_enumerated_plan_preserves_semantics_bags_and_nulls():
    tables = daily_tables()
    plan = daily_plan()
    expected = bag(relations.execute(plan, tables, budget=10 ** 9))
    found = relations.search_plan(plan, search_budget=10 ** 6, tables=tables)
    assert found["search"]["truncated"] is False
    assert found["search"]["remaining_candidates"] == 0
    for cand in found["candidates"]:
        got = bag(relations.execute(cand["plan"], tables, budget=10 ** 9))
        assert got == expected


def test_null_join_key_never_becomes_a_match():
    tables = {"a": [{"id": None}], "b": [{"id": None}]}
    plan = join_scan("a", "b", "inner", ["eq", ["col", "a.id"], ["col", "b.id"]])
    assert relations.execute(plan, tables) == []  # NULL = NULL is unknown, not true
    left = join_scan("a", "b", "left", ["eq", ["col", "a.id"], ["col", "b.id"]])
    out = relations.execute(left, tables)
    assert out == [{"a.id": None, "b.id": None}]  # padded, not matched


# --------------------------------------------------------------------------- #
# Filter placement: post-join filters stay in place over left joins
# --------------------------------------------------------------------------- #

def test_post_join_filter_is_not_pushed_past_left_join():
    tables = {"a": [{"id": 1}, {"id": 2}], "b": [{"id": 1}, {"id": 1}]}
    plan = {"op": "filter", "predicate": ["gt", ["col", "a.id"], ["lit", 0]],
            "input": join_scan("a", "b", "left",
                               ["eq", ["col", "a.id"], ["col", "b.id"]])}
    found = relations.search_plan(plan, tables=tables)
    for cand in found["candidates"]:
        # the filter must remain above the left join
        assert cand["plan"]["op"] == "filter"
        assert cand["plan"]["input"]["op"] == "join"
        assert cand["plan"]["input"].get("kind") == "left"
    expected = bag(relations.execute(plan, tables))
    assert all(bag(relations.execute(c["plan"], tables)) == expected for c in found["candidates"])


def test_filter_pushdown_still_applies_for_inner_join():
    tables = {"a": [{"id": 1}, {"id": 2}, {"id": 3}], "b": [{"id": 2}]}
    plan = {"op": "filter", "predicate": ["gt", ["col", "a.id"], ["lit", 1]],
            "input": join_scan("a", "b", "inner",
                               ["eq", ["col", "a.id"], ["col", "b.id"]])}
    found = relations.search_plan(plan, tables=tables)
    keys = [c["key"] for c in found["candidates"]]
    assert len(keys) == len(set(keys))  # de-duplicated
    # one candidate keeps the filter on top, one pushes it onto the scan
    assert any(c["plan"]["op"] == "filter" for c in found["candidates"])
    assert any(c["plan"]["op"] == "join" and c["plan"]["left"]["op"] == "filter"
               for c in found["candidates"])
    expected = bag(relations.execute(plan, tables))
    assert all(bag(relations.execute(c["plan"], tables)) == expected for c in found["candidates"])
    # legacy optimizer behavior is unchanged
    legacy = relations.optimize(plan)
    assert legacy["op"] == "join" and legacy["left"]["op"] == "filter"


# --------------------------------------------------------------------------- #
# Join reordering: same result, different actual join work
# --------------------------------------------------------------------------- #

def reorder_tables():
    return {
        "t": [{"id": i} for i in range(1, 11)],          # 10 rows
        "b": [{"k": i, "tid": i} for i in range(1, 11)],  # non-selective: 10 matches
        "a": [{"k": 1}, {"k": 1}],                        # selective: only id 1
    }


def reorder_plan():
    # (t join b) join a, both predicates reference t so both orders are legal
    first = join_scan("t", "b", "inner",
                      ["eq", ["col", "t.id"], ["col", "b.tid"]])
    return {"op": "join", "kind": "inner",
            "predicate": ["eq", ["col", "t.id"], ["col", "a.k"]],
            "left": first, "right": {"op": "scan", "table": "a"}}


def test_search_picks_cheaper_order_and_reports_the_choice():
    tables = reorder_tables()
    report = relations.run_query(reorder_plan(), tables)
    assert report["status"] == "completed"
    costs = sorted(c["estimated_work"] for c in report["candidates"])
    assert costs == [242, 322]  # includes emitted rows plus comparison work
    assert report["selected_key"] == report["candidates"][0]["key"]
    assert report["candidates"][0]["estimated_work"] == 242
    # both plans produce the same bag of rows
    rows_by_plan = {c["key"]: tuple(bag(relations.execute(c["plan"], tables, budget=10 ** 9)))
                    for c in report["candidates"]}
    assert len(set(rows_by_plan.values())) == 1
    # actual metered work of the selected plan matches its execution profile
    direct = relations.execute_profile(report["selected_plan"], tables, budget=10 ** 9)
    assert direct["work"] == report["executed"]["work"] == 66
    join_nodes = [n for n in direct["profile"] if n["op"] == "join"]
    assert sorted(n["comparisons"] for n in join_nodes) == [20, 20]
    # the losing order does the same comparisons with a much fatter intermediate
    loser = relations.execute_profile(report["candidates"][1]["plan"], tables, budget=10 ** 9)
    assert loser["work"] == 154


def test_same_rows_but_only_one_plan_fits_the_execution_budget():
    tables = reorder_tables()
    found = relations.search_plan(reorder_plan(), tables=tables)
    cheap, expensive = found["candidates"][0], found["candidates"][1]
    assert cheap["estimated_work"] < expensive["estimated_work"]
    # budget between the two workloads: cheap delivers, expensive hits the limit
    ok = relations.execute_profile(cheap["plan"], tables, budget=120)
    assert ok["limit_triggered"] is False and ok["work"] == 66
    with pytest.raises(relations.ExecutionLimitError) as exc:
        relations.execute_profile(expensive["plan"], tables, budget=120)
    assert isinstance(exc.value, ValueError)  # legacy callers still see ValueError
    assert exc.value.work == 130 and exc.value.budget == 120
    assert exc.value.path == "/left"  # the fat intermediate join is where it trips
    assert exc.value.profile  # partial profile is observable; no rows leak
    assert not hasattr(exc.value, "rows")


def test_zero_match_join_still_charges_every_comparison():
    tables = {"t": [{"id": 1}, {"id": 2}, {"id": 3}], "a": [{"k": 9}, {"k": 8}]}
    plan = join_scan("t", "a", "inner", ["eq", ["col", "t.id"], ["col", "a.k"]])
    prof = relations.execute_profile(plan, tables, budget=10 ** 6)
    assert prof["rows"] == []                      # nothing matched ...
    join_node = [n for n in prof["profile"] if n["op"] == "join"][0]
    assert join_node["comparisons"] == 6          # ... but all comparisons are work
    assert prof["work"] == 3 + 2 + 6              # scan rows + comparisons
    with pytest.raises(relations.ExecutionLimitError) as exc:
        relations.execute_profile(plan, tables, budget=10)
    assert exc.value.work == 11 and exc.value.path == join_node["path"]
    # left join with no matches pads instead, comparisons still charged
    left = join_scan("t", "a", "left", ["eq", ["col", "t.id"], ["col", "a.k"]])
    padded = relations.execute_profile(left, tables, budget=10 ** 6)
    assert len(padded["rows"]) == 3 and all(r["a.k"] is None for r in padded["rows"])


# --------------------------------------------------------------------------- #
# The three limit states
# --------------------------------------------------------------------------- #

def test_search_limited_still_executes_a_complete_plan():
    tables = reorder_tables()
    report = relations.run_query(reorder_plan(), tables, search_budget=6, execution_budget=10 ** 6)
    assert report["status"] == "search_limited"
    assert report["search"]["truncated"] is True
    assert report["search"]["examined"] >= 1
    assert report["search"]["remaining_candidates"] is None  # unchecked candidates remain
    assert report["executed"]["limit_triggered"] is False
    assert report["executed"]["rows"] is not None


def test_no_plan_when_search_budget_allows_no_complete_candidate():
    report = relations.run_query(daily_plan(), daily_tables(), search_budget=0)
    assert report["status"] == "no_plan"
    assert report["selected_plan"] is None and report["candidates"] == []
    assert "executed" not in report


def test_execution_limited_never_returns_partial_rows():
    report = relations.run_query(daily_plan(), daily_tables(),
                                 search_budget=10 ** 6, execution_budget=2)
    assert report["status"] == "execution_limited"
    executed = report["executed"]
    assert executed["rows"] is None
    assert executed["limit_triggered"] is True
    assert executed["work"] > executed["budget"]
    assert executed["limit_path"]
    assert isinstance(executed["profile"], list)


def test_search_and_execution_limits_are_distinct_states():
    tables = reorder_tables()
    limited_search = relations.run_query(reorder_plan(), tables, search_budget=6,
                                         execution_budget=10 ** 6)
    limited_exec = relations.run_query(reorder_plan(), tables, search_budget=10 ** 6,
                                       execution_budget=60)
    assert limited_search["status"] == "search_limited"
    assert limited_exec["status"] == "execution_limited"
    assert limited_search["executed"]["limit_triggered"] is False
    assert limited_exec["executed"]["limit_triggered"] is True


def test_inner_join_cannot_cross_a_left_join_barrier():
    tables = {"t": [{"id": i, "k": i % 2} for i in range(1, 7)],
              "a": [{"k": 0}, {"k": 1}], "b": [{"id": i} for i in range(1, 7)]}
    inner = join_scan("t", "a", "inner", ["eq", ["col", "t.k"], ["col", "a.k"]])
    plan = {"op": "join", "kind": "left",
            "predicate": ["eq", ["col", "t.id"], ["col", "b.id"]],
            "left": inner, "right": {"op": "scan", "table": "b"}}
    found = relations.search_plan(plan, tables=tables)
    assert found["search"]["examined"] == 1
    # the single candidate is the exact original tree, not an inverted one
    assert found["candidates"][0]["key"] == relations.canonical(plan)
    top = found["candidates"][0]["plan"]
    assert top.get("kind") == "left" and top["left"].get("kind") == "inner"


def test_bushy_trees_and_left_deep_round_trip():
    leaf = {"op": "scan", "table": "t"}
    bushy = {"op": "join", "kind": "inner",
             "predicate": ["eq", ["col", "t.id"], ["col", "a.k"]],
             "left": {"op": "join", "kind": "inner",
                      "predicate": ["eq", ["col", "t.id"], ["col", "a.k"]],
                      "left": leaf, "right": {"op": "scan", "table": "a"}},
             "right": {"op": "join", "kind": "inner",
                       "predicate": ["eq", ["col", "t.id"], ["col", "b.k"]],
                       "left": leaf, "right": {"op": "scan", "table": "b"}}}
    _, _, left_deep = relations._flatten_joins(bushy)
    assert left_deep is False
    found = relations.search_plan(bushy,
                                  tables={"t": [{"id": 1}], "a": [{"k": 1}], "b": [{"k": 1}]})
    assert found["search"]["examined"] == 1
    assert found["candidates"][0]["key"] == relations.canonical(bushy)
    # any left-deep tree must flatten and rebuild to itself exactly
    deep = {"op": "join", "kind": "left",
            "predicate": ["eq", ["col", "t.id"], ["col", "b.k"]],
            "left": {"op": "join", "kind": "left",
                     "predicate": ["eq", ["col", "t.id"], ["col", "a.k"]],
                     "left": leaf, "right": {"op": "scan", "table": "a"}},
            "right": {"op": "scan", "table": "b"}}
    base, steps, deep_flag = relations._flatten_joins(deep)
    assert deep_flag is True
    assert relations.canonical(relations._rebuild_joins(base, steps)) == relations.canonical(deep)


def test_legacy_optimize_preserves_original_budget_and_rule():
    tables = {"a": [{"id": 1}, {"id": 1}, {"id": None}, {"id": 2}],
              "b": [{"id": 1}, {"id": 1}, {"id": None}]}
    plan = {"op": "filter", "predicate": ["gt", ["col", "a.id"], ["lit", 0]],
            "input": join_scan("a", "b", "inner",
                               ["eq", ["col", "a.id"], ["col", "b.id"]])}
    assert relations.execute(relations.optimize(plan), tables) == relations.execute(plan, tables)
    plan["input"]["kind"] = "left"
    assert relations.optimize(plan)["op"] == "filter"  # no pushdown past a left join
    with pytest.raises(ValueError):
        relations.optimize({"op": "scan", "table": "a"}, limit=0)


# --------------------------------------------------------------------------- #
# Reporting, determinism and the service entry
# --------------------------------------------------------------------------- #

def test_entry_point_explains_selection_search_and_remaining_candidates():
    report = relations.run_query(daily_plan(), daily_tables())
    assert set(report) >= {"status", "selected_plan", "selected_key", "search",
                           "candidates", "executed"}
    assert set(report["search"]) == {"budget", "work", "examined", "truncated",
                                     "remaining_candidates"}
    assert report["search"]["examined"] == len(report["candidates"])
    cand = report["candidates"][0]
    assert set(cand) >= {"plan", "key", "estimated_work", "estimated_rows", "basis"}
    assert cand["key"] == report["selected_key"]
    executed = report["executed"]
    assert set(executed) >= {"rows", "work", "budget", "limit_triggered", "profile"}
    assert all({"path", "op", "rows_out", "comparisons", "work"} <= set(n)
               for n in executed["profile"])


def test_choice_is_deterministic_for_the_same_input_including_missing_stats():
    tables = reorder_tables()
    first = relations.run_query(reorder_plan(), tables)
    second = relations.run_query(reorder_plan(), tables)
    assert first["selected_key"] == second["selected_key"]
    assert [c["key"] for c in first["candidates"]] == [c["key"] for c in second["candidates"]]
    # skewed keys do not change the deterministic ordering
    skewed = dict(tables, a=[{"k": 1}] * 50)
    r1 = relations.run_query(reorder_plan(), skewed)
    r2 = relations.run_query(reorder_plan(), skewed)
    assert r1["selected_key"] == r2["selected_key"]
    # missing statistics: estimates are unknown but the choice is still fixed
    unknown = relations.search_plan(reorder_plan(), search_budget=10 ** 6, tables={})
    assert all(c["estimated_work"] is None for c in unknown["candidates"])
    again = relations.search_plan(reorder_plan(), search_budget=10 ** 6, tables={})
    assert [c["key"] for c in unknown["candidates"]] == [c["key"] for c in again["candidates"]]


def test_selection_is_stable_under_hash_randomization():
    script = (
        "import sys; sys.path.insert(0, %r);"
        "from relplan import relations;"
        "tables=%r;plan=%r;"
        "print(relations.run_query(plan, tables)['selected_key'])"
        % (os.path.join(os.path.dirname(__file__), "..", "src"), reorder_tables(), reorder_plan())
    )
    env = dict(os.environ, PYTHONHASHSEED="1")
    key1 = subprocess.run([sys.executable, "-c", script], env=env,
                          capture_output=True, text=True, check=True).stdout
    env["PYTHONHASHSEED"] = "2"
    key2 = subprocess.run([sys.executable, "-c", script], env=env,
                          capture_output=True, text=True, check=True).stdout
    assert key1 == key2 and key1.strip()


def test_service_entry_wires_budgets_through():
    payload = {"plan": daily_plan(), "tables": daily_tables(),
               "search_budget": 6, "execution_budget": 10 ** 6}
    result = api.relation_call(payload)
    assert result["status"] == "search_limited"
    assert result["search"]["budget"] == 6 and result["executed"]["budget"] == 10 ** 6
    with pytest.raises(ValueError):
        api.relation_call({"plan": daily_plan(), "tables": daily_tables(), "search_budget": -1})
    with pytest.raises(ValueError):
        api.relation_call({"plan": daily_plan(), "tables": daily_tables(), "execution_budget": 1.5})


def test_legacy_execute_still_raises_plain_value_error_on_budget():
    with pytest.raises(ValueError):
        relations.execute({"op": "scan", "table": "a"}, {"a": [{}, {}]}, budget=1)
    with pytest.raises(ValueError):
        relations.optimize({"op": "scan", "table": "a"}, limit=0)
