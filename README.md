# relplan-lab

Analysis execution primitives.

Python 3.11 or newer. The primary interfaces are importable component libraries and a local JSON HTTP service.

## Install

```sh
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements-dev.txt
./.venv/bin/python -m pip install --no-deps -e .
```

## Run

```sh
./.venv/bin/python -m relplan demo
./.venv/bin/python -m relplan serve --port 8080
./.venv/bin/python -m pytest -q
```

The server binds only to `127.0.0.1`. `GET /health` reports process readiness; `GET /capabilities` lists operations.
Each component has `POST /v1/<operation>` accepting its JSON request object. Responses contain `result` or an explicit error. The service accepts bodies up to 1 MB and at most four concurrent operation handlers. It does not expose a shell, arbitrary file paths or remote URLs.

## Components

| Operation | Behavior |
|---|---|
| `relations` | Bag-preserving Scan/Filter/Project and inner/left joins, SQL-style NULL comparisons, bounded inner-join filter pushdown, metered plan search with join reordering and separate search/execution budgets. |
| `sandbox` | Actual Wasmtime execution with no host imports, fuel accounting, memory limits and integer arguments. |
| `sparse` | Validated CSR structures and sparse products, including duplicate input entries. |
| `autodiff` | Reverse-mode scalar differentiation with shared intermediates, sin and exp. |
| `randomstream` | SHA-256 counter streams with length-delimited split names, jumps and restorable Box-Muller cache. |

## Request examples

Run `python -m relplan.examples` to print valid JSON requests. The output is generated locally and contains no credentials.

```sh
./.venv/bin/python -m relplan.examples > requests.json
curl -s http://127.0.0.1:8080/capabilities
```

Use one operation request from that output as the JSON body for its `/v1/<operation>` endpoint. The `demo` command executes all five operations and prints their computed results.

## Relational plan search

The `relations` operation accepts `search_budget` and `execution_budget`
(non-negative integers, defaults 256 and 100000) and runs `run_query`:

1. enumerate every complete semantic variant under `search_budget` (metered per
   plan node visited); when the budget runs out mid-enumeration the complete
   candidates already found are kept and `search.truncated` is set;
2. rank them deterministically by estimated work and canonical plan text — the
   same input always selects the same plan, even when statistics are missing or
   keys are skewed;
3. execute the selected plan under `execution_budget`, charging one unit per
   emitted row and per join predicate comparison (a zero-match join still pays
   for every comparison).

The transformations never change query meaning. Join reordering is restricted
to left-deep inner-join chains: an order is rejected unless the tables its
predicate needs are already available at that position, bushy trees and any
chain containing a left join are frozen, and the single filter-pushdown rule
only applies to inner joins whose immediate left input is the referenced scan.
Post-join filters above left joins therefore stay in place and unmatched rows
keep their NULL-padded values.

The response `status` distinguishes the four outcomes:

| status | meaning |
|---|---|
| `completed` | the whole search space was examined and execution finished in budget |
| `search_limited` | search hit its limit first, but a complete plan executed successfully |
| `no_plan` | the search budget allowed no complete candidate; nothing is executed |
| `execution_limited` | the selected plan exceeded the execution budget; `rows` is `null` and partial results are never returned |

`search.work` / `search.examined` / `search.remaining_candidates` describe the
search, `selected_plan` / `selected_key` the choice, and `executed.profile`
gives per-node output rows, comparison counts and cumulative work plus
`limit_path` when the execution limit triggered. Calling
`relations.execute_profile(plan, tables, budget)` directly meters any plan with
the same accounting; the legacy `optimize` and `execute` entry points keep
their original signatures and `ValueError` behavior (`ExecutionLimitError`
subclasses `ValueError`).

## Supported scope

Relational optimization uses a conservative rule set and is not a complete SQL optimizer. The Wasm interface has no WASI or host callbacks; module compilation is trusted local work and is not fuel-metered. Differentiation supports scalar primitives only. Random streams are for reproducible numerical use, not security tokens; finite tests do not establish statistical independence.

Runtime state uses explicit temporary directories or caller-provided library objects. Service examples are per-request computations; use the library interfaces for persistent state and long-lived readers.

## Dependencies

Pinned Python dependencies are listed in `requirements.txt`; the development test runner is pinned in `requirements-dev.txt`. System libraries or a native compiler, when required above, are detected at runtime. No accounts, network connection strings, tokens or private-key files are required.
