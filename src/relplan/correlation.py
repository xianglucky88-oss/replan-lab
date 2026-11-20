"""Weighted pairwise feature correlation over sparse sample matrices.

The sample matrix is CSR-backed (rows are samples, columns are features).
Missing values are never guessed: the request states one explicit rule:

* ``zero``   - every sample is observed for every feature; entries absent
  from the CSR are literal zeros.
* ``absent`` - a sample is observed for a feature only when that coordinate
  is stored in the CSR (an explicitly stored zero is still an observation).
* ``mask``   - a second CSR of the same shape declares observations; a
  coordinate is observed when its (duplicate-summed) mask value is nonzero.

Duplicate CSR entries are summed, matching :mod:`relplan.sparse` semantics.
Statistics are always computed per feature pair on that pair's own eligible
sample set: one pair never borrows another pair's effective denominator, and
relationships that cannot be estimated are returned as ``null`` with an
explicit reason instead of a fabricated zero correlation.
"""
import hashlib
import json
import math
import os
from pathlib import Path

from . import sparse

# Minimum effective sample size for an estimable correlation.
ESS_MIN = 2.0
# Relative tolerance below which a weighted variance is treated as zero.
VAR_TOL = 1e-12
# Numerical slack around |corr| = 1. Overshoot inside the slack is clamped
# (and reported); overshoot outside it makes the pair non-estimable.
CORR_NEAR_TOL = 1e-10

MISSING_RULES = ("zero", "absent", "mask")
MAX_DIM = 10000
DEFAULT_WORK_BUDGET = 100000


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a number" % name)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    return value


def _as_csr(payload, name):
    if not isinstance(payload, dict):
        raise ValueError("%s must be a CSR object" % name)
    try:
        matrix = sparse.CSR(payload["rows"], payload["cols"],
                            payload["indptr"], payload["indices"], payload["values"])
    except (KeyError, TypeError):
        raise ValueError("%s has invalid CSR fields" % name)
    return matrix.validate()


def _normalize_pairs(pairs, cols):
    if not isinstance(pairs, list) or not pairs or len(pairs) > MAX_DIM:
        raise ValueError("pairs must be a non-empty list of at most %d entries" % MAX_DIM)
    normalized = []
    for pos, pair in enumerate(pairs):
        if isinstance(pair, dict):
            a, b = pair.get("a"), pair.get("b")
        elif isinstance(pair, (list, tuple)) and len(pair) == 2:
            a, b = pair
        else:
            raise ValueError("pair %d must be [a, b] or {'a': a, 'b': b}" % pos)
        if not isinstance(a, int) or isinstance(a, bool) or not 0 <= a < cols:
            raise ValueError("pair %d references invalid feature a=%r" % (pos, a))
        if not isinstance(b, int) or isinstance(b, bool) or not 0 <= b < cols:
            raise ValueError("pair %d references invalid feature b=%r" % (pos, b))
        normalized.append((a, b))
    return normalized


def prepare_request(data):
    """Validate a correlation request and return typed, normalized inputs."""
    matrix = _as_csr(data.get("matrix"), "matrix")
    n_rows, cols = matrix.rows, matrix.cols

    weights_payload = data.get("weights")
    if not isinstance(weights_payload, list) or len(weights_payload) != n_rows:
        raise ValueError("weights must be a list with one entry per sample")
    weights = [_number(w, "weight[%d]" % i) for i, w in enumerate(weights_payload)]
    if any(w < 0.0 for w in weights):
        raise ValueError("weights must be non-negative")

    rule = data.get("missing_rule")
    if rule not in MISSING_RULES:
        raise ValueError("missing_rule must be one of %s" % ", ".join(MISSING_RULES))

    mask = None
    if "mask" in data and data["mask"] is not None:
        if rule != "mask":
            raise ValueError("mask is only accepted with missing_rule='mask'")
        mask = _as_csr(data["mask"], "mask")
        if mask.rows != n_rows or mask.cols != cols:
            raise ValueError("mask shape must match matrix shape")
    elif rule == "mask":
        raise ValueError("missing_rule='mask' requires a mask CSR")

    pairs = _normalize_pairs(data.get("pairs"), cols)

    budget = data.get("work_budget", DEFAULT_WORK_BUDGET)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("work_budget must be a positive integer")

    offset = data.get("offset", 0)
    limit = data.get("limit")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise ValueError("limit must be a positive integer")
    if offset > len(pairs):
        raise ValueError("offset beyond the pair list")

    return {
        "matrix": matrix, "weights": weights, "missing_rule": rule,
        "mask": mask, "pairs": pairs, "work_budget": budget,
        "offset": offset, "limit": limit,
    }


def request_fingerprint(matrix, weights, rule, mask, work_budget):
    """Stable hash of every statistical input; chunked runs must share it."""
    payload = {
        "matrix": {"rows": matrix.rows, "cols": matrix.cols,
                   "indptr": matrix.indptr, "indices": matrix.indices,
                   "values": matrix.values},
        "weights": weights, "missing_rule": rule, "work_budget": work_budget,
        "mask": None if mask is None else {
            "rows": mask.rows, "cols": mask.cols,
            "indptr": mask.indptr, "indices": mask.indices, "values": mask.values},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _row_sums(csr_matrix):
    """Per-row coordinate -> summed value, applying duplicate-entry semantics."""
    rows = []
    for i in range(csr_matrix.rows):
        merged = {}
        for j, value in csr_matrix.row(i):
            merged[j] = merged.get(j, 0.0) + value
        rows.append(merged)
    return rows


def _column_csr(n_rows, row_indices, values):
    """Build an n_rows x 1 CSR from unique ascending row indices."""
    indptr = [0] * (n_rows + 1)
    indices = []
    stored = []
    cursor = 0
    for i in range(n_rows):
        if cursor < len(row_indices) and row_indices[cursor] == i:
            indices.append(0)
            stored.append(float(values[cursor]))
            cursor += 1
        indptr[i + 1] = len(indices)
    return sparse.CSR(n_rows, 1, indptr, indices, stored).validate()


def _transpose_column(column):
    rows = [i for i in range(column.rows)
            if column.indptr[i] < column.indptr[i + 1]]
    return sparse.CSR(1, column.rows, [0, len(rows)], rows, list(column.values)).validate()


def _dot(left, right, budget):
    """Scalar dot product u^T v computed through the CSR multiply primitive."""
    product = sparse.multiply(_transpose_column(left), right, max_work=budget)
    return product.values[0] if product.values else 0.0


def _correlation_from_sums(cross, var_a, var_b):
    """Pure decision step: raw weighted centered sums -> (corr, reasons)."""
    if var_a <= 0.0 or var_b <= 0.0:
        return None, []
    raw = cross / math.sqrt(var_a * var_b)
    if not math.isfinite(raw):
        return None, [{"code": "correlation_out_of_range", "value": None}]
    # Exact unit values (e.g. a self-pair, whose centered columns are
    # identical) are reported as-is; only genuine overshoot gets a reason.
    if raw == 1.0 or raw == -1.0:
        return raw, []
    if raw > 1.0 + CORR_NEAR_TOL or raw < -1.0 - CORR_NEAR_TOL:
        return None, [{"code": "correlation_out_of_range", "value": raw}]
    if raw > 1.0:
        return 1.0, [{"code": "correlation_clamped", "value": raw}]
    if raw < -1.0:
        return -1.0, [{"code": "correlation_clamped", "value": raw}]
    return raw, []


def _analyze_pair(a, b, n_rows, value_rows, observed, weights, budget, work):
    """All statistics for one pair, using only the pair's eligible samples."""
    eligible = []
    for i in range(n_rows):
        work[0] += 1
        if weights[i] <= 0.0:
            continue
        if (a in observed[i]) and (b in observed[i]):
            eligible.append(i)
    k = len(eligible)

    w_col = _column_csr(n_rows, eligible, [weights[i] for i in eligible])
    ones = _column_csr(n_rows, eligible, [1.0] * k)
    x_col = _column_csr(n_rows, eligible, [value_rows[i].get(a, 0.0) for i in eligible])
    y_col = _column_csr(n_rows, eligible, [value_rows[i].get(b, 0.0) for i in eligible])

    total_weight = _dot(w_col, ones, budget)
    sum_w2 = _dot(w_col, w_col, budget)
    sum_x = _dot(w_col, x_col, budget)
    sum_y = _dot(w_col, y_col, budget)
    sum_x2 = _dot(w_col, _column_csr(
        n_rows, eligible, [value_rows[i].get(a, 0.0) ** 2 for i in eligible]), budget)
    sum_y2 = _dot(w_col, _column_csr(
        n_rows, eligible, [value_rows[i].get(b, 0.0) ** 2 for i in eligible]), budget)

    result = {
        "a": a, "b": b, "observed_samples": k,
        "effective_weight": total_weight,
        "effective_sample_size": None,
        "mean_a": None, "mean_b": None,
        "variance_a": None, "variance_b": None,
        "covariance": None, "correlation": None, "reasons": [],
    }
    if total_weight <= 0.0:
        result["reasons"].append({"code": "no_eligible_samples"})
        return result, work[0]

    ess = (total_weight * total_weight) / sum_w2 if sum_w2 > 0.0 else None
    result["effective_sample_size"] = ess
    mean_a = sum_x / total_weight
    mean_b = sum_y / total_weight
    result["mean_a"] = mean_a
    result["mean_b"] = mean_b

    if k < ESS_MIN or ess is None or ess < ESS_MIN:
        result["reasons"].append({
            "code": "insufficient_effective_sample_size",
            "effective_sample_size": ess, "observed_samples": k})

    # Second, centered pass: centered columns and their weighted copies are
    # CSR objects, so every moment still goes through sparse.multiply.
    centered_x = _column_csr(n_rows, eligible,
                             [value_rows[i].get(a, 0.0) - mean_a for i in eligible])
    centered_y = _column_csr(n_rows, eligible,
                             [value_rows[i].get(b, 0.0) - mean_b for i in eligible])
    weighted_cx = _column_csr(n_rows, eligible,
                              [weights[i] * (value_rows[i].get(a, 0.0) - mean_a) for i in eligible])
    weighted_cy = _column_csr(n_rows, eligible,
                              [weights[i] * (value_rows[i].get(b, 0.0) - mean_b) for i in eligible])
    raw_var_a = _dot(centered_x, weighted_cx, budget)
    raw_var_b = _dot(centered_y, weighted_cy, budget)
    raw_cross = _dot(centered_x, weighted_cy, budget)
    result["variance_a"] = raw_var_a / total_weight
    result["variance_b"] = raw_var_b / total_weight
    result["covariance"] = raw_cross / total_weight

    # Scale-invariant zero-variance test on raw weighted sums: the relative
    # tolerance is applied to the weighted second moment (already in the same
    # units), so extreme weight concentration cannot flip the decision.
    if raw_var_a <= VAR_TOL * (1.0 + abs(sum_x2)):
        result["reasons"].append({"code": "zero_variance", "feature": a})
    if raw_var_b <= VAR_TOL * (1.0 + abs(sum_y2)):
        result["reasons"].append({"code": "zero_variance", "feature": b})

    # Zero-variance and ESS reasons are all reported; only the correlation
    # itself is withheld when any blocking reason applies.
    blocking = {reason["code"] for reason in result["reasons"]}
    if not blocking:
        if a == b:
            # Identity pairs are exactly unit correlation regardless of
            # floating-point summation order across the three accumulators.
            result["correlation"] = 1.0
        else:
            corr, corr_reasons = _correlation_from_sums(raw_cross, raw_var_a, raw_var_b)
            result["correlation"] = corr
            result["reasons"].extend(corr_reasons)
    return result, work[0]


def analyze(data):
    """Compute weighted correlation statistics for the requested feature pairs.

    The response contains only the requested pairs, in request order; there is
    no dense matrix and no zero-filled entry for relationships not computed.
    ``offset``/``limit`` slice the pair list without changing any pair's
    numbers, so independent chunks always equal the corresponding slice of a
    full run.
    """
    req = prepare_request(data)
    matrix, weights = req["matrix"], req["weights"]
    rule, mask, budget = req["missing_rule"], req["mask"], req["work_budget"]
    pairs = req["pairs"][req["offset"]:]
    if req["limit"] is not None:
        pairs = pairs[:req["limit"]]

    value_rows = _row_sums(matrix)
    if rule == "zero":
        observed = [set(range(matrix.cols)) for _ in range(matrix.rows)]
    elif rule == "absent":
        observed = [set(row.keys()) for row in value_rows]
    else:
        observed = [set(j for j, value in row.items() if value != 0.0)
                    for row in _row_sums(mask)]

    fingerprint = request_fingerprint(matrix, weights, rule, mask, budget)
    work = [0]
    results = []
    # Each pair performs up to eleven sparse dot products plus one eligibility
    # scan over the rows, so 12*n_rows is a conservative per-pair work bound.
    # Every dot product also passes the same budget as its multiply max_work,
    # keeping the CSR primitive's own work limit in force at every step.
    for a, b in pairs:
        projected = work[0] + 12 * matrix.rows
        if projected > budget:
            raise ValueError("work budget exceeded")
        pair_result, work[0] = _analyze_pair(
            a, b, matrix.rows, value_rows, observed, weights, budget, work)
        results.append(pair_result)
        work[0] += 1
    return {
        "fingerprint": fingerprint,
        "missing_rule": rule,
        "total_pairs": len(req["pairs"]),
        "offset": req["offset"],
        "limit": req["limit"],
        "pairs": results,
    }


class CorrelationWorkspace:
    """Restricted on-disk workspace keeping chunked and full conclusions equal.

    The workspace lives under an explicit caller-owned base directory. A
    content fingerprint pins the matrix, weights, mask and missing rule; any
    chunk presented against different inputs is rejected instead of silently
    mixing statistical populations. Completed pairs are tracked explicitly,
    so relationships not yet computed are reported as missing rather than
    filled with zero correlations.
    """

    VERSION = 1

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        with open(self.directory / "state.json", encoding="utf-8") as handle:
            self.state = json.load(handle)
        if self.state.get("version") != self.VERSION:
            raise ValueError("unsupported workspace version")
        with open(self.directory / "manifest.json", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

    @classmethod
    def create(cls, base_dir, key, request_data):
        base = Path(base_dir).resolve()
        if not isinstance(key, str) or not key or len(key.encode()) > 128:
            raise ValueError("invalid workspace key")
        if "\x00" in key or "/" in key or "\\" in key or key in (".", ".."):
            raise ValueError("invalid workspace key")
        base.mkdir(parents=True, exist_ok=True)
        directory = (base / key).resolve()
        if directory != base and base not in directory.parents:
            raise ValueError("workspace key escapes the base directory")
        if directory.exists():
            raise ValueError("workspace key already exists")
        os.mkdir(directory, mode=0o700)
        os.mkdir(directory / "chunks", mode=0o700)

        req = prepare_request(request_data)
        matrix, weights = req["matrix"], req["weights"]
        fingerprint = request_fingerprint(
            matrix, weights, req["missing_rule"], req["mask"], req["work_budget"])
        state = {
            "version": cls.VERSION, "key": key, "fingerprint": fingerprint,
            "rows": matrix.rows, "cols": matrix.cols,
            "missing_rule": req["missing_rule"],
            "work_budget": req["work_budget"],
            "pairs": [list(pair) for pair in req["pairs"]],
        }
        with open(directory / "state.json", "w", encoding="utf-8") as handle:
            json.dump(state, handle, sort_keys=True)
        manifest = {"version": cls.VERSION, "completed": [], "chunks": []}
        with open(directory / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True)
        return cls(directory)

    @property
    def fingerprint(self):
        return self.state["fingerprint"]

    def status(self):
        total = len(self.state["pairs"])
        completed = sorted(self.manifest["completed"])
        missing = [i for i in range(total) if i not in set(completed)]
        return {
            "key": self.state["key"], "fingerprint": self.fingerprint,
            "pairs_total": total, "pairs_completed": len(completed),
            "complete": not missing, "missing_pair_indices": missing,
        }

    def run_chunk(self, request_data, offset, limit):
        req = prepare_request(request_data)
        fingerprint = request_fingerprint(
            req["matrix"], req["weights"], req["missing_rule"],
            req["mask"], req["work_budget"])
        if fingerprint != self.fingerprint:
            raise ValueError("chunk inputs do not match workspace fingerprint")
        total = len(self.state["pairs"])
        if req["pairs"] != [tuple(pair) for pair in self.state["pairs"]]:
            raise ValueError("chunk pair order does not match workspace declaration")
        if not isinstance(offset, int) or offset < 0 or offset >= total:
            raise ValueError("invalid chunk offset")
        if not isinstance(limit, int) or limit <= 0 or offset + limit > total:
            raise ValueError("invalid chunk limit")
        covered = set(self.manifest["completed"])
        if any(i in covered for i in range(offset, offset + limit)):
            raise ValueError("chunk range overlaps completed work")

        chunk_request = dict(request_data)
        chunk_request["offset"] = offset
        chunk_request["limit"] = limit
        result = analyze(chunk_request)

        chunk_id = len(self.manifest["chunks"])
        name = "chunk-%06d.json" % chunk_id
        with open(self.directory / "chunks" / name, "w", encoding="utf-8") as handle:
            json.dump({"offset": offset, "limit": limit,
                       "fingerprint": fingerprint, "result": result},
                      handle, sort_keys=True)
        self.manifest["chunks"].append({"file": name, "offset": offset, "limit": limit})
        self.manifest["completed"].extend(range(offset, offset + limit))
        with open(self.directory / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, sort_keys=True)
        return result

    def results(self):
        """Concatenate chunk outputs in pair order; refuse to zero-fill gaps."""
        status = self.status()
        if not status["complete"]:
            raise ValueError("workspace is incomplete: pair indices %s not computed"
                             % status["missing_pair_indices"])
        ordered = [None] * status["pairs_total"]
        for chunk in self.manifest["chunks"]:
            with open(self.directory / "chunks" / chunk["file"], encoding="utf-8") as handle:
                payload = json.load(handle)
            for pos, pair_result in enumerate(payload["result"]["pairs"]):
                ordered[chunk["offset"] + pos] = pair_result
        return ordered

    def verify(self, request_data):
        """Recompute every pair in one full run and compare against chunks."""
        req = prepare_request(request_data)
        fingerprint = request_fingerprint(
            req["matrix"], req["weights"], req["missing_rule"],
            req["mask"], req["work_budget"])
        if fingerprint != self.fingerprint:
            raise ValueError("verification inputs do not match workspace fingerprint")
        full = analyze(request_data)
        stored = self.results()
        if len(stored) != len(full["pairs"]):
            raise ValueError("chunk/full pair count mismatch")
        for i, (got, expected) in enumerate(zip(stored, full["pairs"])):
            if got != expected:
                raise ValueError("chunk result for pair index %d disagrees with full run" % i)
        return {"pairs_verified": len(stored), "fingerprint": self.fingerprint}
