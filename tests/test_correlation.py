import json
import math
import os
import random
from pathlib import Path

import pytest

from relplan import sparse, correlation
from relplan.correlation import CorrelationWorkspace, VAR_TOL, ESS_MIN


def dense_csr(matrix):
    """Independent dense expansion of a CSR, summing duplicate coordinates."""
    out = [[0.0] * matrix.cols for _ in range(matrix.rows)]
    for i in range(matrix.rows):
        for j, value in matrix.row(i):
            out[i][j] += value
    return out


def reference_analyze(matrix_payload, weights, missing_rule, pairs, mask_payload=None):
    """Independent per-sample, per-pair dense reference implementation.

    It never imports or calls the module under test; moments are accumulated
    sample by sample from expanded dense tables so the CSR-product path is
    checked against a genuinely different computation.
    """
    matrix = sparse.CSR(matrix_payload["rows"], matrix_payload["cols"],
                        matrix_payload["indptr"], matrix_payload["indices"],
                        matrix_payload["values"]).validate()
    dense = dense_csr(matrix)
    n_rows, cols = matrix.rows, matrix.cols
    if missing_rule == "zero":
        observed = [[True] * cols for _ in range(n_rows)]
    elif missing_rule == "absent":
        observed = []
        for i in range(n_rows):
            coords = set()
            for j, _ in matrix.row(i):
                coords.add(j)
            observed.append([j in coords for j in range(cols)])
    else:
        mask = sparse.CSR(mask_payload["rows"], mask_payload["cols"],
                          mask_payload["indptr"], mask_payload["indices"],
                          mask_payload["values"]).validate()
        mask_dense = dense_csr(mask)
        observed = [[mask_dense[i][j] != 0.0 for j in range(cols)] for i in range(n_rows)]

    results = []
    for a, b in pairs:
        eligible = [i for i in range(n_rows)
                    if weights[i] > 0.0 and observed[i][a] and observed[i][b]]
        k = len(eligible)
        total_weight = sum(weights[i] for i in eligible)
        sum_w2 = sum(weights[i] ** 2 for i in eligible)
        entry = {"a": a, "b": b, "observed_samples": k,
                 "effective_weight": total_weight,
                 "effective_sample_size": None,
                 "mean_a": None, "mean_b": None,
                 "variance_a": None, "variance_b": None,
                 "covariance": None, "correlation": None, "reasons": []}
        if total_weight <= 0.0:
            entry["reasons"].append({"code": "no_eligible_samples"})
            results.append(entry)
            continue
        ess = total_weight ** 2 / sum_w2 if sum_w2 > 0 else None
        entry["effective_sample_size"] = ess
        mean_a = sum(weights[i] * dense[i][a] for i in eligible) / total_weight
        mean_b = sum(weights[i] * dense[i][b] for i in eligible) / total_weight
        entry["mean_a"], entry["mean_b"] = mean_a, mean_b
        if k < ESS_MIN or ess is None or ess < ESS_MIN:
            entry["reasons"].append({
                "code": "insufficient_effective_sample_size",
                "effective_sample_size": ess, "observed_samples": k})
        raw_var_a = sum(weights[i] * (dense[i][a] - mean_a) ** 2 for i in eligible)
        raw_var_b = sum(weights[i] * (dense[i][b] - mean_b) ** 2 for i in eligible)
        raw_cov = sum(weights[i] * (dense[i][a] - mean_a) * (dense[i][b] - mean_b)
                      for i in eligible)
        sum_x2 = sum(weights[i] * dense[i][a] ** 2 for i in eligible)
        sum_y2 = sum(weights[i] * dense[i][b] ** 2 for i in eligible)
        entry["variance_a"], entry["variance_b"] = raw_var_a / total_weight, raw_var_b / total_weight
        entry["covariance"] = raw_cov / total_weight
        if raw_var_a <= VAR_TOL * (1.0 + abs(sum_x2)):
            entry["reasons"].append({"code": "zero_variance", "feature": a})
        if raw_var_b <= VAR_TOL * (1.0 + abs(sum_y2)):
            entry["reasons"].append({"code": "zero_variance", "feature": b})
        blocking = {r["code"] for r in entry["reasons"]}
        if not blocking:
            if a == b:
                # Identity pairs are exactly unit correlation by construction.
                entry["correlation"] = 1.0
            else:
                raw = raw_cov / math.sqrt(raw_var_a * raw_var_b)
                if not math.isfinite(raw) or abs(raw) > 1.0 + correlation.CORR_NEAR_TOL:
                    entry["correlation"] = None
                    entry["reasons"].append({"code": "correlation_out_of_range", "value": raw if math.isfinite(raw) else None})
                elif raw == 1.0 or raw == -1.0:
                    entry["correlation"] = raw
                else:
                    if raw > 1.0:
                        entry["correlation"], note = 1.0, "correlation_clamped"
                    elif raw < -1.0:
                        entry["correlation"], note = -1.0, "correlation_clamped"
                    else:
                        entry["correlation"], note = raw, None
                    if note:
                        entry["reasons"].append({"code": note, "value": raw})
        results.append(entry)
    return results


def assert_reference_match(got_results, expected_results):
    assert len(got_results) == len(expected_results)
    for got, expected in zip(got_results, expected_results):
        assert got["a"] == expected["a"] and got["b"] == expected["b"]
        assert got["observed_samples"] == expected["observed_samples"]
        assert got["effective_weight"] == pytest.approx(expected["effective_weight"])
        if expected["effective_sample_size"] is None:
            assert got["effective_sample_size"] is None
        else:
            assert got["effective_sample_size"] == pytest.approx(expected["effective_sample_size"])
        for field in ("mean_a", "mean_b", "variance_a", "variance_b", "covariance"):
            if expected[field] is None:
                assert got[field] is None
            else:
                assert got[field] == pytest.approx(expected[field], rel=1e-12)
        if expected["correlation"] is None:
            assert got["correlation"] is None
        else:
            assert got["correlation"] == pytest.approx(expected["correlation"], rel=1e-12)
        assert [r["code"] for r in got["reasons"]] == [r["code"] for r in expected["reasons"]]
        assert got["reasons"] == expected["reasons"]


def build_random_case(seed, n_rows=12, cols=5, include_mask=False, duplicate_rate=0.3):
    rng = random.Random(seed)
    indptr = [0]
    indices, values = [], []
    mask_indptr = [0]
    mask_indices, mask_values = [], []
    for _ in range(n_rows):
        for j in range(cols):
            observed = rng.random() < 0.7
            if not observed:
                continue
            indices.append(j)
            values.append(round(rng.uniform(-5, 5), 4))
            if rng.random() < duplicate_rate:  # duplicated coordinate -> sum
                indices.append(j)
                values.append(round(rng.uniform(-2, 2), 4))
            if include_mask:
                mask_indices.append(j)
                mask_values.append(1 if rng.random() < 0.85 else 0.0)
        indptr.append(len(indices))
        if include_mask:
            mask_indptr.append(len(mask_indices))
    matrix = {"rows": n_rows, "cols": cols, "indptr": indptr,
              "indices": indices, "values": values}
    mask = None
    if include_mask:
        mask = {"rows": n_rows, "cols": cols, "indptr": mask_indptr,
                "indices": mask_indices, "values": mask_values}
    weights = [round(rng.uniform(0, 4), 3) if rng.random() > 0.15 else 0.0
               for _ in range(n_rows)]
    pairs = [(a, b) for a in range(cols) for b in range(a, cols)]
    return matrix, weights, pairs, mask


def test_weighted_correlation_matches_independent_reference_zero_rule():
    matrix, weights, pairs, _ = build_random_case(7)
    request = {"matrix": matrix, "weights": weights,
               "missing_rule": "zero", "pairs": pairs}
    got = correlation.analyze(request)["pairs"]
    expected = reference_analyze(matrix, weights, "zero", pairs)
    assert_reference_match(got, expected)


def test_weighted_correlation_matches_independent_reference_absent_rule():
    matrix, weights, pairs, _ = build_random_case(11)
    request = {"matrix": matrix, "weights": weights,
               "missing_rule": "absent", "pairs": pairs}
    got = correlation.analyze(request)["pairs"]
    expected = reference_analyze(matrix, weights, "absent", pairs)
    assert_reference_match(got, expected)


def test_weighted_correlation_matches_independent_reference_mask_rule():
    matrix, weights, pairs, mask = build_random_case(17, include_mask=True)
    request = {"matrix": matrix, "weights": weights, "missing_rule": "mask",
               "mask": mask, "pairs": pairs}
    got = correlation.analyze(request)["pairs"]
    expected = reference_analyze(matrix, weights, "mask", pairs, mask)
    assert_reference_match(got, expected)


def test_sparse_zeros_are_observed_under_absent_but_missing_under_mask():
    # Column 0 stores an explicit zero: observed under 'absent', but the mask
    # excludes sample 1 there, so the pair's denominator must shrink.
    matrix = {"rows": 3, "cols": 2, "indptr": [0, 2, 4, 6],
              "indices": [0, 1, 0, 1, 0, 1],
              "values": [1.0, 2.0, 0.0, 4.0, 3.0, 6.0]}
    mask = {"rows": 3, "cols": 2, "indptr": [0, 2, 3, 5],
            "indices": [0, 1, 1, 0, 1], "values": [1, 1, 1, 1, 1]}
    weights = [1.0, 1.0, 1.0]
    absent = correlation.analyze({"matrix": matrix, "weights": weights,
                                  "missing_rule": "absent", "pairs": [(0, 1)]})["pairs"][0]
    masked = correlation.analyze({"matrix": matrix, "weights": weights,
                                  "missing_rule": "mask", "mask": mask,
                                  "pairs": [(0, 1)]})["pairs"][0]
    assert absent["observed_samples"] == 3
    assert masked["observed_samples"] == 2
    ref_absent = reference_analyze(matrix, weights, "absent", [(0, 1)])[0]
    ref_masked = reference_analyze(matrix, weights, "mask", [(0, 1)], mask)[0]
    assert_reference_match([absent], [ref_absent])
    assert_reference_match([masked], [ref_masked])


def test_missing_rule_is_required_and_mask_shape_checked():
    matrix, weights, pairs, _ = build_random_case(3, n_rows=4, cols=3)
    with pytest.raises(ValueError):
        correlation.analyze({"matrix": matrix, "weights": weights, "pairs": pairs})
    with pytest.raises(ValueError):
        correlation.analyze({"matrix": matrix, "weights": weights,
                             "missing_rule": "mask", "pairs": pairs})
    bad_mask = dict(matrix, rows=5)
    with pytest.raises(ValueError):
        correlation.analyze({"matrix": matrix, "weights": weights,
                             "missing_rule": "mask", "mask": bad_mask, "pairs": pairs})
    with pytest.raises(ValueError):
        correlation.analyze({"matrix": matrix, "weights": weights,
                             "missing_rule": "zero", "mask": bad_mask, "pairs": pairs})


def test_pairwise_denominators_are_independent():
    # Feature 0 missing for sample 2; feature 1 missing for sample 3.
    matrix = {"rows": 4, "cols": 3, "indptr": [0, 3, 5, 7, 9],
              "indices": [0, 1, 2, 0, 1, 1, 2, 0, 2],
              "values": [1.0, 2.0, 3.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0]}
    weights = [1.0, 1.0, 1.0, 1.0]
    # Observed coordinates per sample: row0={0,1,2}, row1={0,1},
    # row2={1,2}, row3={0,2}. Each pair's intersection is its own denominator:
    # (0,1)={0,1}, (0,2)={0,3}, (1,2)={0,2}; none shares another pair's count logic.
    result = correlation.analyze({"matrix": matrix, "weights": weights,
                                  "missing_rule": "absent",
                                  "pairs": [(0, 1), (0, 2), (1, 2)]})
    assert [p["observed_samples"] for p in result["pairs"]] == [2, 2, 2]
    assert [p["effective_weight"] for p in result["pairs"]] == [2.0, 2.0, 2.0]
    # A pair with a fully observed column keeps its own wider denominator.
    full_col = correlation.analyze({"matrix": matrix, "weights": weights,
                                    "missing_rule": "zero",
                                    "pairs": [(0, 1), (0, 2)]})
    assert [p["observed_samples"] for p in full_col["pairs"]] == [4, 4]
    expected = reference_analyze(matrix, weights, "absent", [(0, 1), (0, 2), (1, 2)])
    assert_reference_match(result["pairs"], expected)


def test_extreme_weights_ess_rejection_and_zero_weight_exclusion():
    matrix = {"rows": 5, "cols": 2, "indptr": [0, 2, 4, 6, 8, 10],
              "indices": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
              "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
    weights = [1e9, 1e9, 1.0, 1.0, 0.0]  # last sample excluded; ESS near 2
    result = correlation.analyze({"matrix": matrix, "weights": weights,
                                  "missing_rule": "zero", "pairs": [(0, 1)]})["pairs"][0]
    assert result["observed_samples"] == 4
    assert 2.0 <= result["effective_sample_size"] < 2.001
    assert result["reasons"] == []
    weights_single = [5.0, 0.0, 0.0, 0.0, 0.0]
    single = correlation.analyze({"matrix": matrix, "weights": weights_single,
                                  "missing_rule": "zero", "pairs": [(0, 1)]})["pairs"][0]
    assert single["effective_sample_size"] == pytest.approx(1.0)
    assert single["correlation"] is None
    assert [r["code"] for r in single["reasons"]] == ["insufficient_effective_sample_size",
                                                       "zero_variance", "zero_variance"]


def test_no_eligible_samples_is_not_zero_correlation():
    matrix = {"rows": 2, "cols": 2, "indptr": [0, 1, 1],
              "indices": [1], "values": [3.0]}
    mask = {"rows": 2, "cols": 2, "indptr": [0, 0, 0], "indices": [], "values": []}
    weights = [1.0, 1.0]
    result = correlation.analyze({"matrix": matrix, "weights": weights,
                                  "missing_rule": "mask", "mask": mask,
                                  "pairs": [(0, 1)]})["pairs"][0]
    assert result["observed_samples"] == 0
    assert result["effective_sample_size"] is None
    assert result["correlation"] is None
    assert result["reasons"] == [{"code": "no_eligible_samples"}]


def test_zero_variance_feature_blocks_correlation_but_keeps_covariance():
    matrix = {"rows": 4, "cols": 2, "indptr": [0, 2, 4, 6, 8],
              "indices": [0, 1, 0, 1, 0, 1, 0, 1],
              "values": [3.0, 1.0, 3.0, 2.0, 3.0, 4.0, 3.0, 8.0]}
    weights = [1.0, 2.0, 1.0, 1.0]
    result = correlation.analyze({"matrix": matrix, "weights": weights,
                                  "missing_rule": "zero", "pairs": [(0, 1)]})["pairs"][0]
    assert result["variance_a"] == pytest.approx(0.0)
    assert result["covariance"] == pytest.approx(0.0)
    assert result["correlation"] is None
    assert [r["code"] for r in result["reasons"]] == ["zero_variance"]
    assert result["reasons"][0]["feature"] == 0


def test_near_perfect_correlation_clamp_and_out_of_range():
    # Perfectly affine columns -> exact +/-1, no clamp reason.
    matrix = {"rows": 5, "cols": 2, "indptr": [0, 2, 4, 6, 8, 10],
              "indices": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
              "values": [1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0]}
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    exact = correlation.analyze({"matrix": matrix, "weights": weights,
                                 "missing_rule": "zero", "pairs": [(0, 1)]})["pairs"][0]
    assert exact["correlation"] == pytest.approx(1.0)
    assert exact["reasons"] == []
    # Self pair is 1.0 through the generic path.
    self_pair = correlation.analyze({"matrix": matrix, "weights": weights,
                                     "missing_rule": "zero", "pairs": [(1, 1)]})["pairs"][0]
    assert self_pair["correlation"] == pytest.approx(1.0) and self_pair["reasons"] == []
    # Forced overshoot past the tolerance is reported as non-estimable.
    assert correlation._correlation_from_sums(2.0, 1.0, 1.0) == (
        None, [{"code": "correlation_out_of_range", "value": 2.0}])
    clamped, reasons = correlation._correlation_from_sums(1.0 + 1e-12, 1.0, 1.0)
    assert clamped == 1.0 and reasons[0]["code"] == "correlation_clamped"


def test_chunks_equal_full_run_and_reference():
    matrix, weights, pairs, mask = build_random_case(23, include_mask=True)
    full = correlation.analyze({"matrix": matrix, "weights": weights,
                                "missing_rule": "mask", "mask": mask, "pairs": pairs})
    expected = reference_analyze(matrix, weights, "mask", pairs, mask)
    assert_reference_match(full["pairs"], expected)
    chunked = []
    offsets = list(range(0, len(pairs), 4))
    for offset in offsets:
        part = correlation.analyze({"matrix": matrix, "weights": weights,
                                    "missing_rule": "mask", "mask": mask,
                                    "pairs": pairs, "offset": offset, "limit": 4})
        assert part["fingerprint"] == full["fingerprint"]
        chunked.extend(part["pairs"])
    assert chunked == full["pairs"]
    tail = correlation.analyze({"matrix": matrix, "weights": weights,
                                "missing_rule": "mask", "mask": mask,
                                "pairs": pairs, "offset": len(pairs) - 2})
    assert tail["pairs"] == full["pairs"][-2:]


def test_workspace_chunking_fingerprint_and_verification(tmp_path):
    matrix, weights, pairs, mask = build_random_case(31, include_mask=True)
    request = {"matrix": matrix, "weights": weights, "missing_rule": "mask",
               "mask": mask, "pairs": pairs}
    ws = CorrelationWorkspace.create(tmp_path, "job-a", request)
    assert ws.status() == {"key": "job-a", "fingerprint": ws.fingerprint,
                           "pairs_total": len(pairs), "pairs_completed": 0,
                           "complete": False,
                           "missing_pair_indices": list(range(len(pairs)))}
    for offset in range(0, len(pairs), 3):
        limit = min(3, len(pairs) - offset)
        ws.run_chunk(request, offset, limit)
    assert ws.status()["complete"] is True
    # Full independent recomputation must equal every stored chunk value.
    verification = ws.verify(request)
    assert verification["pairs_verified"] == len(pairs)
    stored = ws.results()
    assert [p["correlation"] for p in stored] == [
        p["correlation"] for p in correlation.analyze(request)["pairs"]]
    # Key sanitization and containment.
    with pytest.raises(ValueError):
        CorrelationWorkspace.create(tmp_path, "../escape", request)
    with pytest.raises(ValueError):
        CorrelationWorkspace.create(tmp_path, "job/a", request)
    with pytest.raises(ValueError):
        CorrelationWorkspace.create(tmp_path, "job-a", request)


def test_workspace_rejects_mixed_inputs_and_overlapping_chunks(tmp_path):
    matrix, weights, pairs, mask = build_random_case(41, include_mask=True)
    request = {"matrix": matrix, "weights": weights, "missing_rule": "mask",
               "mask": mask, "pairs": pairs}
    ws = CorrelationWorkspace.create(tmp_path, "job", request)
    ws.run_chunk(request, 0, 3)
    with pytest.raises(ValueError):
        ws.run_chunk(request, 0, 3)  # overlap
    tampered = dict(request, weights=[w + 0.001 for w in weights])
    with pytest.raises(ValueError):
        ws.run_chunk(tampered, 3, 3)  # fingerprint mismatch
    with pytest.raises(ValueError):
        ws.verify(tampered)
    with pytest.raises(ValueError):
        ws.results()  # incomplete workspace must not zero-fill
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    assert not (other_dir / "job").exists()


def test_uncomputed_relationships_are_never_filled_with_zero(tmp_path):
    matrix, weights, pairs, _ = build_random_case(5, n_rows=4, cols=3)
    request = {"matrix": matrix, "weights": weights,
               "missing_rule": "zero", "pairs": pairs}
    ws = CorrelationWorkspace.create(tmp_path, "k", request)
    ws.run_chunk(request, 0, 2)
    status = ws.status()
    assert status["missing_pair_indices"] == list(range(2, len(pairs)))
    with pytest.raises(ValueError):
        ws.results()
    # The manifest stores only completed ranges; no pair entries on disk for others.
    chunks = list((ws.directory / "chunks").iterdir())
    payload = json.loads(chunks[0].read_text())
    assert len(payload["result"]["pairs"]) == 2


def test_work_budget_is_enforced_per_pair():
    matrix, weights, pairs, _ = build_random_case(1, n_rows=60, cols=4)
    with pytest.raises(ValueError, match="work budget"):
        correlation.analyze({"matrix": matrix, "weights": weights,
                             "missing_rule": "absent", "pairs": pairs,
                             "work_budget": 10})


def test_invalid_weights_and_duplicate_csr_entries():
    matrix = {"rows": 3, "cols": 2, "indptr": [0, 2, 2, 5],
              "indices": [0, 1, 0, 0, 1],
              "values": [1.0, 2.0, 3.0, 4.0, 5.0]}  # row2 stores (0,3)+(0,4)=7
    weights = [1.0, -1.0, 1.0]
    with pytest.raises(ValueError):
        correlation.analyze({"matrix": matrix, "weights": weights,
                             "missing_rule": "zero", "pairs": [(0, 1)]})
    weights = [1.0, float("nan"), 1.0]
    with pytest.raises(ValueError):
        correlation.analyze({"matrix": matrix, "weights": weights,
                             "missing_rule": "zero", "pairs": [(0, 1)]})
    good_weights = [1.0, 1.0, 1.0]
    got = correlation.analyze({"matrix": matrix, "weights": good_weights,
                               "missing_rule": "zero", "pairs": [(0, 1)]})["pairs"]
    expected = reference_analyze(matrix, good_weights, "zero", [(0, 1)])
    assert_reference_match(got, expected)
    dense = dense_csr(sparse.CSR(3, 2, matrix["indptr"], matrix["indices"], matrix["values"]))
    assert dense[2][0] == 7.0


def test_feature_identity_validation():
    matrix, weights, pairs, _ = build_random_case(2, n_rows=3, cols=2)
    base = {"matrix": matrix, "weights": weights, "missing_rule": "zero"}
    with pytest.raises(ValueError):
        correlation.analyze(dict(base, pairs=[(0, 2)]))
    with pytest.raises(ValueError):
        correlation.analyze(dict(base, pairs=[(-1, 0)]))
    with pytest.raises(ValueError):
        correlation.analyze(dict(base, pairs=[]))
    with pytest.raises(ValueError):
        correlation.analyze(dict(base, pairs=[{"a": 0, "c": 1}]))
    # Dict-form pairs are accepted and normalized.
    got = correlation.analyze(dict(base, pairs=[{"a": 0, "b": 1}]))["pairs"]
    assert got[0]["a"] == 0 and got[0]["b"] == 1
