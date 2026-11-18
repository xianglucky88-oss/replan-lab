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
| `relations` | Bag-preserving Scan/Filter/Project and inner/left joins, SQL-style NULL comparisons, bounded inner-join filter pushdown. |
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

## Supported scope

Relational optimization uses a conservative rule set and is not a complete SQL optimizer. The Wasm interface has no WASI or host callbacks; module compilation is trusted local work and is not fuel-metered. Differentiation supports scalar primitives only. Random streams are for reproducible numerical use, not security tokens; finite tests do not establish statistical independence.

Runtime state uses explicit temporary directories or caller-provided library objects. Service examples are per-request computations; use the library interfaces for persistent state and long-lived readers.

## Dependencies

Pinned Python dependencies are listed in `requirements.txt`; the development test runner is pinned in `requirements-dev.txt`. System libraries or a native compiler, when required above, are detected at runtime. No accounts, network connection strings, tokens or private-key files are required.
