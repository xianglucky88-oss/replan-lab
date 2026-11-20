import math

import pytest

from relplan import autodiff, classification as C


def make_rows(count, seed, prefix, intercept=0.1, slope=1.4, z_slope=-0.7):
    state = C.RandomStream(seed)
    rows = []
    for index in range(count):
        x = state.normal()
        z = state.normal()
        probability = C._sigmoid(intercept + slope * x + z_slope * z)
        label = int(state.uniform() < probability)
        rows.append({"key": f"{prefix}-{index}", "entity": f"{prefix}-{index}",
                     "x": round(x, 6), "z": round(z, 6), "y": label})
    return rows


FIELDS = [
    {"name": "x", "available_from": 1},
    {"name": "z", "available_from": 1},
    {"name": "y", "available_from": 1, "outcome_known_at": 1},
]


def window(name, role, stage, rows, fields=None, period=None):
    return {
        "name": name, "role": role, "window": stage,
        "period": period or {"start": 1, "end": 10},
        "label_field": "y", "fields": fields or FIELDS, "rows": rows,
    }


def standard_request():
    return {
        "task": C.TASK,
        "train": window("train", "train", "train", make_rows(60, 1, "t"),
                        period={"start": 1, "end": 60}),
        "validation": window("validation", "validation", "select", make_rows(40, 2, "v"),
                             period={"start": 61, "end": 100}),
        "holdout": window("holdout", "holdout", "holdout", make_rows(40, 3, "h"),
                          period={"start": 101, "end": 140}),
        "candidates": [
            {"terms": [["intercept"], ["var", "x"]], "max_iter": 100},
            {"terms": [["intercept"], ["var", "x"], ["var", "z"]], "max_iter": 100},
        ],
        "selection": {"metric": "log_loss"},
    }


# --- windows, grouping and the holdout boundary --------------------------

def test_full_pipeline_reports_all_three_windows_and_verifies_handoff():
    result = C.run_pipeline(standard_request())
    assert result["status"] == "completed"
    stages = result["stages"]
    assert [stage["window"] for stage in stages["train"]] == ["train", "train"]
    assert stages["select"]["window"] == "select"
    assert stages["holdout"]["window"] == "holdout"
    assert result["handoff"]["verified_by_holdout"] is True
    assert stages["holdout"]["selected_model_checksum"] == result["selected_model_checksum"]
    assert result["disclosures"]["selection_used_holdout_labels"] is False
    assert result["disclosures"]["final_metrics_influence_selection"] is False
    assert 0.0 <= result["holdout_metrics"]["auc"] <= 1.0
    assert result["holdout_metrics"]["calibration_bins"]


def test_entity_spanning_windows_is_excluded_whole_and_disclosed():
    request = standard_request()
    request["holdout"]["rows"][0]["entity"] = request["train"]["rows"][0]["entity"]
    request["validation"]["rows"][0]["entity"] = request["train"]["rows"][0]["entity"]
    result = C.run_pipeline(request)
    spanning = set(result["disclosures"]["excluded_entities"])
    assert request["train"]["rows"][0]["entity"] in spanning
    for stage in result["stages"]["train"]:
        excluded = {item["key"] for item in stage["excluded_rows"]}
        assert request["train"]["rows"][0]["key"] in excluded
        assert all("spans more than one window" in item["reason"]
                   for item in stage["excluded_rows"] if item["key"] in excluded)
    select_excluded = {item["key"] for item in result["stages"]["select"]["candidates"][0]["metrics"]["excluded_rows"]}
    assert request["validation"]["rows"][0]["key"] in select_excluded
    holdout_excluded = {item["key"] for item in result["stages"]["holdout"]["excluded_rows"]}
    assert request["holdout"]["rows"][0]["key"] in holdout_excluded


def test_duplicate_key_is_rejected_rather_than_relabelled():
    request = standard_request()
    request["train"]["rows"].append(dict(request["train"]["rows"][0]))
    with pytest.raises(C.ConditionError):
        C.run_pipeline(request)


def test_periods_cannot_cross_the_holdout_boundary():
    request = standard_request()
    request["validation"]["period"] = {"start": 61, "end": 120}
    with pytest.raises(C.ConditionError):
        C.run_pipeline(request)


def test_window_labels_must_match_their_stage():
    request = standard_request()
    request["holdout"]["window"] = "select"
    with pytest.raises(C.ConditionError):
        C.run_pipeline(request)


# --- outcome timing ------------------------------------------------------

def test_outcome_not_yet_known_is_never_used_as_a_negative():
    request = standard_request()
    for row in request["train"]["rows"][:5]:
        row["outcome_known_at"] = 2
    record = C.train_model(request["train"], request["candidates"][0], ())
    excluded = {item["key"]: item["reason"] for item in record["excluded_rows"]}
    for row in request["train"]["rows"][:5]:
        assert "not determined" in excluded[row["key"]]
        assert row["key"] not in record["rows_used"]
    assert record["training"]["rows_used"] == len(request["train"]["rows"]) - 5


def test_label_declared_late_blocks_the_stage():
    request = standard_request()
    request["train"]["fields"] = [
        {"name": "x", "available_from": 1},
        {"name": "z", "available_from": 1},
        {"name": "y", "available_from": 1, "outcome_known_at": 3},
    ]
    with pytest.raises(C.ConditionError, match="not final"):
        C.train_model(request["train"], request["candidates"][0], ())


def test_holdout_outcome_read_only_after_selection():
    request = standard_request()
    for row in request["validation"]["rows"]:
        row["y"] = None
    with pytest.raises(C.ConditionError):
        C.run_pipeline(request)


# --- field availability and missingness ---------------------------------

def test_feature_must_be_available_when_the_row_is_scored():
    request = standard_request()
    for row in request["train"]["rows"]:
        row["z"] = None
    request["train"]["rows"][0]["z"] = 1.0
    request["train"]["fields"] = [
        {"name": "x", "available_from": 1},
        {"name": "z", "available_from": 2},
        {"name": "y", "available_from": 1, "outcome_known_at": 1},
    ]
    with pytest.raises(C.ConditionError, match="not populated"):
        C.train_model(request["train"], request["candidates"][1], ())


def test_legal_missing_is_imputed_while_unavailable_field_is_reported():
    request = standard_request()
    for row in request["train"]["rows"]:
        if row["x"] is None:
            row["x"] = 0.5
    request["train"]["fields"] = [
        {"name": "x", "available_from": 1, "missing": "mean"},
        {"name": "y", "available_from": 1, "outcome_known_at": 1},
    ]
    record = C.train_model(request["train"], {"terms": [["intercept"], ["var", "x"]], "max_iter": 100}, ())
    assert record["status"] == "completed"
    model = C.load_model(record["model"])
    batch = {"name": "next", "role": "batch", "window": "predict",
             "rows": [{"key": "known", "x": 0.4}, {"key": "gap", "x": None}]}
    scored = C.score_batch(record["model"], batch)
    by_key = {row["key"]: row for row in scored["rows"]}
    assert by_key["gap"]["imputed"] == ["x"]
    assert by_key["known"]["imputed"] == []
    # A field that is not populated until a later phase is not a legal gap.
    late = dict(batch)
    late["rows"] = [{"key": "late", "x": 0.4}]
    late_model = C.load_model(record["model"])
    late_model.payload["features"]["x"]["available_from"] = 2
    late_model = C.load_model(C.build_payload(
        late_model.payload["terms"], late_model.payload["params"],
        {"x": {"available_from": 2, "missing": "mean"}},
        late_model.payload["standardizer"], late_model.payload["l2"], late_model.payload["training"],
    ))
    result = late_model.predict_rows(late["rows"], as_of=1)
    assert result["rows"] == []
    assert result["unavailable"][0]["fields"] == ["x"]


def test_missing_without_policy_is_excluded_and_disclosed():
    request = standard_request()
    request["train"]["rows"][0]["x"] = None
    record = C.train_model(request["train"], request["candidates"][0], ())
    excluded = {item["key"]: item for item in record["excluded_rows"]}
    assert "no imputation" in excluded[request["train"]["rows"][0]["key"]]["reason"]


# --- standardisation fitted on training only -----------------------------

def test_standardization_uses_training_rows_only_and_survives_reload():
    request = standard_request()
    record = C.train_model(request["train"], request["candidates"][1], ())
    model = C.load_model(record["model"])
    train_x = [row["x"] for row in request["train"]["rows"] if row["x"] is not None]
    assert model.payload["standardizer"]["x"]["mean"] == pytest.approx(sum(train_x) / len(train_x))
    # Out-of-window values must not shift the fitted statistics.
    request["validation"]["rows"][0]["x"] = 1000.0
    again = C.load_model(model.dump())
    assert again.payload["standardizer"] == model.payload["standardizer"]
    first = model.score({"x": 1000.0, "z": 0.0})
    second = again.score({"x": 1000.0, "z": 0.0})
    assert first == second


def test_model_reload_rejects_tampering():
    record = C.train_model(standard_request()["train"], standard_request()["candidates"][0], ())
    payload = record["model"]
    payload["params"][0] += 1.0
    with pytest.raises(C.ConditionError, match="checksum"):
        C.load_model(payload)


# --- expression meaning shared with the analyst -------------------------

def test_recheck_matches_training_score_and_parameter_gradients():
    request = standard_request()
    record = C.train_model(request["train"], {"terms": [["intercept"], ["var", "x"], ["var", "z"]],
                                              "max_iter": 100}, ())
    model = C.load_model(record["model"])
    row = {"x": 0.7, "z": -1.1}
    score = model.score(row)
    gradients = model.parameter_gradients(row)
    rechecked = model.recheck(row)
    assert rechecked["score"] == pytest.approx(score["score"])
    for name, value in rechecked["gradients"].items():
        assert value == pytest.approx(gradients["gradients"][name])


def test_shared_intermediate_contributes_once_per_reference():
    # t0 = w0 * x; t1 = w1 * t0; t2 = w2 * t1 * t1; score = t0 + t1 + t2
    terms = [["var", "x"], ["mul", 0], ["mul2", 1]]
    compiled, _ = C.compile_terms(terms)
    result = autodiff.evaluate_terms(compiled, {"x": 3.0}, [0.5, 1.0, 2.0])
    assert result["value"] == pytest.approx(1.5 + 1.5 + 4.5)
    # d/dw0: dt0/dw0=3, dt1/dw0=3, dt2/dw0 = 2*2*1.5*3 = 18 -> 3+3+18 = 24
    assert result["gradients"][0] == pytest.approx(24.0)
    rechecked = C.recheck({"terms": terms, "params": [0.5, 1.0, 2.0]}, {"x": 3.0})
    assert rechecked["gradients"]["w0"] == pytest.approx(24.0)
    assert rechecked["score"] == pytest.approx(7.5)


def test_expression_accepts_the_project_scalar_primitives():
    compiled, _ = C.compile_terms([["var", "x"], ["sin", 0], ["exp", 0]])
    result = autodiff.evaluate_terms(compiled, {"x": 0.3}, [0.2, -0.4, 0.1])
    base = 0.2 * 0.3
    expected = base + (-0.4) * math.sin(base) + 0.1 * math.exp(base)
    assert result["value"] == pytest.approx(expected)
    chain = 1.0 + (-0.4) * math.cos(base) + 0.1 * math.exp(base)
    assert result["gradients"][0] == pytest.approx(chain * 0.3)


def test_reference_must_point_backwards():
    with pytest.raises(C.ConditionError):
        C.compile_terms([["mul", 0]])
    with pytest.raises(C.ConditionError):
        C.compile_terms([["var", "x"], ["mul", 2]])


# --- extreme imbalance and numerical range --------------------------------

def test_extreme_imbalance_keeps_training_signal_and_stable_probabilities():
    rows = [{"key": f"n{i}", "entity": f"n{i}", "x": -(i % 7) / 5.0, "y": 0, "outcome_known_at": 1}
            for i in range(60)]
    rows.append({"key": "p0", "entity": "p0", "x": 6.0, "y": 1, "outcome_known_at": 1})
    dataset = window("imbalanced", "train", "train", rows,
                     fields=[{"name": "x", "available_from": 1},
                             {"name": "y", "available_from": 1, "outcome_known_at": 1}])
    record = C.train_model(dataset, {"terms": [["intercept"], ["var", "x"]], "max_iter": 100}, ())
    assert record["status"] == "completed"
    model = C.load_model(record["model"])
    confident = model.score({"x": -40.0})
    assert 0.0 <= confident["probability"] < 1e-12
    assert math.isfinite(confident["score"]) and confident["score"] < -20.0
    wrong = model.score({"x": 40.0})
    assert 1.0 - 1e-12 < wrong["probability"] <= 1.0
    # A confidently misjudged sample keeps its full training signal: the
    # logistic derivative is ~1 away from the label even at p underflow, and
    # no clipping truncates the loss.
    assert C._sigmoid(-800.0) == 0.0
    assert C._bce(-800.0, 1.0) == pytest.approx(800.0)
    residual = C._sigmoid(-800.0) - 1.0
    assert residual == pytest.approx(-1.0)
    assert C.metrics([0, 0, 0, 1], [-800.0, -700.0, -40.0, 30.0])["auc"] == pytest.approx(1.0)
    assert C.metrics([0, 0, 0, 1], [-800.0, -700.0, -40.0, 30.0])["log_loss"] < 1e-9


# --- incomplete fits never enter selection --------------------------------

def test_stopped_or_failed_fit_cannot_be_selected_or_replace_a_validated_model():
    request = standard_request()
    fresh = C.train_model(request["train"], request["candidates"][0], ())
    assert fresh["status"] == "completed"
    blocked = C.train_model(request["train"],
                            {"terms": [["intercept"], ["var", "x"]], "max_iter": 1}, ())
    assert blocked["status"] == "stopped" and blocked["model"] is None
    selection = C.select_model([blocked, fresh], request["validation"], {"metric": "log_loss"})
    refused = selection["refused_candidates"]
    assert any("did not complete" in item["reason"] for item in refused)
    assert selection["selected_model_checksum"] == fresh["model_checksum"]
    assert len(selection["candidates"]) == 1


def test_all_incomplete_candidates_keep_the_previous_validated_model():
    request = standard_request()
    previous = C.run_pipeline(request)["stages"]["select"]
    blocked = C.train_model(request["train"],
                            {"terms": [["intercept"], ["var", "x"]], "max_iter": 1}, ())
    assert blocked["status"] == "stopped"
    selection = C.select_model([blocked], request["validation"], {"metric": "log_loss"},
                              previous=previous)
    assert selection["retained_previous"] is True
    assert selection["selected_model_checksum"] == previous["selected_model_checksum"]
    assert selection["candidates"] == []


def test_evaluation_failure_makes_a_candidate_ineligible():
    request = standard_request()
    record = C.train_model(request["train"], request["candidates"][0], ())
    degenerate = {"name": "one-class", "role": "validation", "window": "select",
                  "period": {"start": 61, "end": 100}, "label_field": "y",
                  "fields": request["validation"]["fields"],
                  "rows": [{**row, "y": 1} for row in request["validation"]["rows"]]}
    with pytest.raises(C.ConditionError, match="no candidate"):
        C.select_model([record], degenerate, {"metric": "log_loss"})
    previous = C.run_pipeline(standard_request())["stages"]["select"]
    retained = C.select_model([record], degenerate, {"metric": "log_loss"}, previous=previous)
    assert retained["retained_previous"] is True
    assert retained["candidates"] == []
    assert any("validation failed" in item["reason"] for item in retained["refused_candidates"])


def test_holdout_refuses_an_unfinished_selection():
    request = standard_request()
    selection = C.run_pipeline(request)["stages"]["select"]
    failed = dict(selection)
    failed["status"] = "stopped"
    with pytest.raises(C.ConditionError):
        C.run_holdout(failed, request["holdout"])


def test_holdout_metrics_do_not_change_the_selection():
    request = standard_request()
    first = C.run_pipeline(request)
    second = C.run_pipeline(request)
    assert first["selected_model_checksum"] == second["selected_model_checksum"]
    # Re-running with reordered holdout rows must not change the chosen model.
    request2 = standard_request()
    request2["holdout"]["rows"] = list(reversed(request2["holdout"]["rows"]))
    third = C.run_pipeline(request2)
    assert third["selected_model_checksum"] == first["selected_model_checksum"]


def test_selection_metric_direction_is_respected():
    request = standard_request()
    by_loss = C.run_pipeline(request)
    request_auc = standard_request()
    request_auc["selection"]["metric"] = "auc"
    by_auc = C.run_pipeline(request_auc)
    assert by_loss["stages"]["select"]["lower_is_better"] is True
    assert by_auc["stages"]["select"]["lower_is_better"] is False
