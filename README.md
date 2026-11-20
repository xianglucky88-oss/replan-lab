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
| `expression` | Shared scalar term evaluator: one forward pass and reverse-mode gradients for every term weight. |
| `classification` | Time-windowed binary-classification pipeline: train, select, then evaluate the selected model once on the holdout. |
| `model_load` | Re-validate a serialized model artifact, including its checksum, feature constraints and fitted standardizer. |
| `model_predict` | Score the next batch with a reloaded artifact, separating legal missing values from fields not yet populated. |
| `model_recheck` | Re-evaluate a model expression through the scalar autodiff entry point and return score plus parameter derivatives. |

## Request examples

Run `python -m relplan.examples` to print valid JSON requests. The output is generated locally and contains no credentials.

```sh
./.venv/bin/python -m relplan.examples > requests.json
curl -s http://127.0.0.1:8080/capabilities
```

Use one operation request from that output as the JSON body for its `/v1/<operation>` endpoint. The `demo` command executes every operation and prints its computed results.

### Binary-classification windows

`classification` takes `train`, `validation` and `holdout` datasets and never re-cuts them:

- Each dataset declares `window` (`train`/`select`/`holdout`/`predict`), `role`, optional `period`, a `label_field` and `fields`. A field carries `available_from` (prediction phase 1..3), `outcome_known_at` (for the label) and `missing` (`error` or `mean`).
- Rows whose entity appears in more than one window are excluded as a whole and listed in `excluded_rows`; the window boundary is never moved to keep a sample.
- A label is used only when its outcome was determined by the close of that stage. Rows with an outcome that is still pending are disclosed, never counted as negatives.
- Location and scale statistics are fitted on training rows only and travel inside the model artifact, so a reloaded model scores exactly what it scored at evaluation time.
- `select` ranks only completed fits. A stopped or failed fit exposes no parameter vector: it is listed in `refused_candidates` and can never overwrite a previously validated selection.
- `holdout` reloads the selected artifact, verifies its checksum and evaluates it once. Holdout labels do not influence training or selection.
- `model_predict` distinguishes a legal gap covered by a `missing: "mean"` policy (imputed and reported) from a field that is not populated at the requested `as_of` phase (the row is reported as unavailable).

The score is a sum of scalar terms over conditioned inputs. Supported term operators are `["intercept"]`, `["var", field]`, and `["mul"|"mul2"|"sin"|"exp", earlier_index]`. Fitting and `model_recheck` call the same `autodiff.evaluate_terms`, so the score and every parameter derivative keep one meaning, and an intermediate referenced by several later terms contributes through each reference.

## Supported scope

Relational optimization uses a conservative rule set and is not a complete SQL optimizer. The Wasm interface has no WASI or host callbacks; module compilation is trusted local work and is not fuel-metered. Differentiation supports scalar primitives only. Random streams are for reproducible numerical use, not security tokens; finite tests do not establish statistical independence.

Runtime state uses explicit temporary directories or caller-provided library objects. Service examples are per-request computations; use the library interfaces for persistent state and long-lived readers.

## Dependencies

Pinned Python dependencies are listed in `requirements.txt`; the development test runner is pinned in `requirements-dev.txt`. System libraries or a native compiler, when required above, are detected at runtime. No accounts, network connection strings, tokens or private-key files are required.
