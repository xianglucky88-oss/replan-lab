"""Temporal binary-classification pipeline with an auditable model hand-off.

Three windows are always distinct and are never re-cut:

``train``
    Fit parameters from labelled rows whose outcome was already determined by
    the close of the training window, using only fields that existed when the
    row was scored.
``select``
    Compare completed fits on a validation split.  Nothing about the model
    changes except which candidate is chosen.
``holdout``
    Score the *already selected* artifact once.  Its labels never feed back
    into fitting or selection.

A model is a scalar expression over conditioned inputs.  Its terms are built
from the primitives already supported by :mod:`relplan.autodiff` (addition,
multiplication, ``sin`` and ``exp``), and both fitting and analyst
re-evaluation call :func:`relplan.autodiff.evaluate_terms`.  A term may be
referenced by several later terms; reverse-mode accumulation therefore keeps
every contribution of a shared intermediate.
"""

from __future__ import annotations

import hashlib
import json
import math

from . import autodiff
from .randomstream import RandomStream

TASK = "binary_classification"
FORMAT = "relplan.binary-classification-model"
MODEL_VERSION = 1

PHASE = {"train": 1, "select": 2, "holdout": 3, "predict": 3}
STAGES = ("train", "select", "holdout")
WINDOWS = ("train", "select", "holdout", "predict")

MAX_ROWS = 5000
MAX_TERMS = 64
MAX_ITER = 5000
MAX_EVALUATIONS = 500_000
CALIBRATION_BINS = 5

DEFAULT_L2 = 1e-3
DEFAULT_MAX_ITER = 1000
DEFAULT_MAX_EVALUATIONS = 200_000
DEFAULT_GRAD_TOL = 1e-6
DEFAULT_DAMPING = 1e-8
DEFAULT_SELECTION_METRIC = "log_loss"

MAXIMISE = {"auc", "average_precision"}
METRICS = {"auc", "average_precision", "log_loss", "brier", "calibration_error"}


class ConditionError(ValueError):
    """A declared condition of the pipeline is not met."""


class ExpressionError(ConditionError):
    """A scalar expression is malformed or leaves its numerical domain."""


class UnavailableFieldError(ConditionError):
    """A model field is not yet populated at the requested prediction phase."""

    def __init__(self, fields, as_of):
        self.fields = list(fields)
        self.as_of = as_of
        super().__init__(
            "fields " + ", ".join(repr(name) for name in self.fields)
            + f" are not populated at prediction phase {as_of}"
        )


# ---------------------------------------------------------------------------
# scalar expression language
# ---------------------------------------------------------------------------

def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExpressionError(f"{label} must be a real number")
    out = float(value)
    if not math.isfinite(out):
        raise ExpressionError(f"{label} must be finite")
    return out


def compile_terms(terms):
    """Validate a term list and return the canonical compiled form.

    Each term is a list in evaluation order:

    ``["var", field]``
        ``w_i * field``
    ``["intercept"]``
        ``w_i``
    ``[op, index]`` for ``op`` in ``mul``/``mul2``/``sin``/``exp``
        ``w_i * op(term_values[index])``; ``index`` must be strictly earlier.

    The resulting tuples are exactly what :func:`autodiff.evaluate_terms`
    consumes, so the analyst and the trainer share one evaluator.
    """
    if not isinstance(terms, list) or not 1 <= len(terms) <= MAX_TERMS:
        raise ExpressionError(f"terms must be a list of 1..{MAX_TERMS} entries")
    compiled, canonical = [], []
    for index, raw in enumerate(terms):
        if not isinstance(raw, list) or not raw or not isinstance(raw[0], str):
            raise ExpressionError(f"term {index} must be a list starting with an operator")
        op = raw[0]
        if op == "var":
            if len(raw) != 2 or not isinstance(raw[1], str) or not raw[1].strip():
                raise ExpressionError(f"term {index}: var requires exactly one field name")
            compiled.append(("var", raw[1].strip()))
            canonical.append(["var", raw[1].strip()])
        elif op == "intercept":
            if len(raw) != 1:
                raise ExpressionError(f"term {index}: intercept takes no arguments")
            compiled.append(("const",))
            canonical.append(["intercept"])
        elif op in ("mul", "mul2", "sin", "exp"):
            if len(raw) != 2 or not isinstance(raw[1], int) or isinstance(raw[1], bool):
                raise ExpressionError(f"term {index}: {op} requires one earlier term index")
            if not 0 <= raw[1] < index:
                raise ExpressionError(f"term {index}: reference must point to an earlier term")
            compiled.append((op, raw[1]))
            canonical.append([op, raw[1]])
        else:
            raise ExpressionError(f"term {index}: unsupported operator {op!r}")
    return tuple(compiled), canonical


def feature_fields(compiled):
    """Input fields referenced by ``var`` terms, in first-use order."""
    seen, out = set(), []
    for term in compiled:
        if term[0] == "var" and term[1] not in seen:
            seen.add(term[1])
            out.append(term[1])
    return out


def parameter_names(compiled):
    return [f"w{index}" for index in range(len(compiled))]


def _evaluate(compiled, params, values):
    """Single shared entry point for scoring, fitting and re-checking."""
    try:
        result = autodiff.evaluate_terms(compiled, values, params)
    except autodiff.ExpressionError as error:
        raise ExpressionError(str(error)) from error
    return result


def recheck(expression, inputs, standardizer=None):
    """Re-evaluate the same expression through the scalar autodiff entry point.

    ``inputs`` are raw field values.  When a standardizer is supplied the
    values are conditioned exactly as the model does; otherwise they are used
    as already-conditioned scalars.  The returned score and parameter
    derivatives are the training-time quantities, and a shared intermediate
    referenced twice contributes through every path.
    """
    if not isinstance(expression, dict):
        raise ConditionError("expression must be an object with terms and params")
    compiled, _ = compile_terms(expression.get("terms"))
    params = expression.get("params")
    if not isinstance(params, list) or len(params) != len(compiled):
        raise ConditionError("expression params must align with its terms")
    params = [_finite(value, f"parameter {index}") for index, value in enumerate(params)]
    values, imputed = _condition(inputs, compiled, standardizer)
    result = _evaluate(compiled, params, values)
    return {
        "score": result["value"],
        "gradients": dict(zip(parameter_names(compiled), result["gradients"])),
        "term_values": dict(enumerate(result["term_values"])),
        "imputed": imputed,
    }


def _condition(inputs, compiled, standardizer):
    if not isinstance(inputs, dict):
        raise ConditionError("prediction row must be an object")
    fields = feature_fields(compiled)
    if standardizer is None:
        values = {}
        for field in fields:
            if inputs.get(field) is None:
                raise ConditionError(f"field {field!r} must be provided")
            values[field] = _finite(inputs[field], f"field {field!r}")
        return values, []
    std = Standardizer.load(standardizer, fields)
    return std.condition(fields, inputs)


# ---------------------------------------------------------------------------
# conditioning
# ---------------------------------------------------------------------------

class Standardizer:
    """Location/scale statistics fitted on the training window only."""

    def __init__(self, stats):
        self.stats = stats

    @classmethod
    def fit(cls, rows, fields, policies):
        stats = {}
        for field in fields:
            present = [float(row[field]) for row in rows if row.get(field) is not None]
            if not present:
                raise ConditionError(f"field {field!r} has no observed value in training")
            mean = _finite(math.fsum(present) / len(present), f"mean of {field!r}")
            variance = math.fsum((value - mean) ** 2 for value in present) / len(present)
            scale = math.sqrt(variance)
            if not math.isfinite(scale):
                raise ConditionError(f"field {field!r} has a non-finite spread")
            scale = scale or 1.0
            low, high = min(present), max(present)
            policy = policies[field]
            stats[field] = {
                "mean": mean,
                "scale": scale,
                "min": low,
                "max": high,
                "impute": mean if policy == "mean" else None,
            }
        return cls(stats)

    def dump(self):
        return json.loads(json.dumps(self.stats, sort_keys=True))

    @classmethod
    def load(cls, payload, fields):
        if not isinstance(payload, dict) or set(payload) != set(fields):
            raise ConditionError("standardizer fields do not match the model feature fields")
        stats = {}
        for field in fields:
            raw = payload[field]
            if not isinstance(raw, dict) or set(raw) != {"mean", "scale", "min", "max", "impute"}:
                raise ConditionError(f"standardizer entry for {field!r} is malformed")
            mean = _finite(raw["mean"], f"mean of {field!r}")
            scale = _finite(raw["scale"], f"scale of {field!r}")
            low = _finite(raw["min"], f"minimum of {field!r}")
            high = _finite(raw["max"], f"maximum of {field!r}")
            if scale <= 0.0:
                raise ConditionError(f"scale of {field!r} must be positive")
            if low > high:
                raise ConditionError(f"observed range of {field!r} is inverted")
            impute = raw["impute"]
            if impute is not None:
                impute = _finite(impute, f"imputation of {field!r}")
                if not low <= impute <= high:
                    raise ConditionError(f"imputation of {field!r} lies outside its observed range")
            stats[field] = {
                "mean": mean, "scale": scale, "min": low, "max": high, "impute": impute
            }
        return cls(stats)

    def condition(self, fields, inputs):
        values, imputed = {}, []
        for field in fields:
            raw = inputs.get(field)
            if raw is None:
                spec = self.stats[field]
                if spec["impute"] is None:
                    raise ConditionError(f"field {field!r} is required and must not be missing")
                raw = spec["impute"]
                imputed.append(field)
            scalar = _finite(raw, f"field {field!r}")
            spec = self.stats[field]
            values[field] = (scalar - spec["mean"]) / spec["scale"]
            if not math.isfinite(values[field]):
                raise ExpressionError(f"field {field!r} produced a non-finite value")
        return values, imputed


# ---------------------------------------------------------------------------
# dataset contract
# ---------------------------------------------------------------------------

def _hashable(value, label):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ConditionError(f"{label} must be a string or an integer")
    if isinstance(value, str) and not value.strip():
        raise ConditionError(f"{label} must not be empty")
    return (type(value).__name__, value)


def _field_declarations(dataset, where):
    declarations = dataset.get("fields")
    if not isinstance(declarations, list) or not declarations:
        raise ConditionError(f"{where}: fields must be a non-empty list")
    specs = {}
    for raw in declarations:
        if not isinstance(raw, dict):
            raise ConditionError(f"{where}: every field declaration must be an object")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip() or name in specs:
            raise ConditionError(f"{where}: field names must be non-empty and unique")
        available = raw.get("available_from", 1)
        known = raw.get("outcome_known_at")
        missing = raw.get("missing", "error")
        if not isinstance(available, int) or isinstance(available, bool) or not 1 <= available <= 3:
            raise ConditionError(f"{where}: field {name!r} available_from must be in 1..3")
        if known is not None and (not isinstance(known, int) or isinstance(known, bool) or not 1 <= known <= 3):
            raise ConditionError(f"{where}: field {name!r} outcome_known_at must be in 1..3 or null")
        if missing not in ("error", "mean"):
            raise ConditionError(f"{where}: field {name!r} missing policy must be 'error' or 'mean'")
        specs[name] = {"name": name, "available_from": available, "outcome_known_at": known, "missing": missing}
    return specs


def _period(dataset, where):
    period = dataset.get("period")
    if period is None:
        return None
    if not isinstance(period, dict) or set(period) != {"start", "end"}:
        raise ConditionError(f"{where}: period must have exactly start and end")
    start, end = period["start"], period["end"]
    for value, label in ((start, "start"), (end, "end")):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConditionError(f"{where}: period {label} must be an integer")
    if start > end:
        raise ConditionError(f"{where}: period start must not follow its end")
    return (start, end)


def validate_dataset(dataset, where, stage=None):
    """Validate a window: declarations, keys, entities and row values."""
    if not isinstance(dataset, dict):
        raise ConditionError(f"{where}: dataset must be an object")
    name = dataset.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConditionError(f"{where}: dataset name must be a non-empty string")
    role = dataset.get("role")
    if role not in ("train", "validation", "holdout", "batch"):
        raise ConditionError(f"{where}: role must be one of train/validation/holdout/batch")
    window = dataset.get("window")
    if window not in WINDOWS:
        raise ConditionError(f"{where}: window must be one of {WINDOWS}")
    if stage is not None and window != stage:
        raise ConditionError(f"{where}: this stage requires window={stage!r}, got {window!r}")
    if window == "predict" and not dataset.get("fields"):
        specs = {}
    else:
        specs = _field_declarations(dataset, where)
    label_field = dataset.get("label_field")
    if label_field is not None and label_field not in specs:
        raise ConditionError(f"{where}: label_field must name a declared field")
    entity_field = dataset.get("entity_field")
    if entity_field is not None and entity_field not in specs:
        raise ConditionError(f"{where}: entity_field must name a declared field")
    rows = dataset.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise ConditionError(f"{where}: rows must be a list of 1..{MAX_ROWS} objects")
    keys, entities = set(), set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ConditionError(f"{where}: row {index} must be an object")
        if "key" not in row:
            raise ConditionError(f"{where}: row {index} is missing its key")
        key = _hashable(row["key"], f"{where}: row {index} key")
        if key in keys:
            raise ConditionError(
                f"{where}: row key {row['key']!r} appears twice; the same record may not be relabelled"
            )
        keys.add(key)
        entity = row.get("entity", row["key"]) if entity_field is None else row.get(entity_field)
        if entity is None:
            raise ConditionError(f"{where}: row {index} is missing its entity")
        entities.add(_hashable(entity, f"{where}: row {index} entity"))
        for field in specs:
            value = row.get(field)
            if value is not None:
                _finite(value, f"{where}: row {index} field {field!r}")
    return {
        "name": name,
        "role": role,
        "window": window,
        "period": _period(dataset, where),
        "fields": specs,
        "label_field": label_field,
        "entity_field": entity_field,
        "rows": rows,
        "keys": keys,
        "entities": entities,
        "where": where,
    }


def _require_role(data, role, stage):
    if data["role"] != role:
        raise ConditionError(f"{stage} requires a dataset role={role!r}, got {data['role']!r}")


def _entity_of(row, data):
    if data["entity_field"] is None:
        return _hashable(row.get("entity", row["key"]), "entity")
    return _hashable(row[data["entity_field"]], "entity")


def _check_periods(train, validation, holdout):
    if train["period"] and validation["period"] and train["period"][1] > validation["period"][0]:
        raise ConditionError("the training period must not extend past the selection period")
    if validation["period"] and holdout["period"] and validation["period"][1] > holdout["period"][0]:
        raise ConditionError("the selection period must not extend past the holdout boundary")


def _spanning_entities(datasets):
    membership = {}
    for position, data in enumerate(datasets):
        for row in data["rows"]:
            membership.setdefault(_entity_of(row, data), set()).add(position)
    return {entity for entity, windows in membership.items() if len(windows) > 1}


def _label_status(row, data):
    """Why a row's label cannot be used at this stage, or ``None``."""
    if data["label_field"] is None:
        return "dataset declares no label field"
    label_spec = data["fields"][data["label_field"]]
    known = row.get("outcome_known_at", label_spec["outcome_known_at"])
    if known is None:
        return "outcome time is not declared"
    if not isinstance(known, int) or isinstance(known, bool) or not 1 <= known <= 3:
        raise ConditionError(f"row {row['key']!r} outcome_known_at must be in 1..3")
    if known > PHASE[data["window"]]:
        return f"outcome is not determined until phase {known}"
    if row.get(data["label_field"]) is None:
        return "outcome has not been observed"
    return None


def prepare_rows(data, stage, spanning, features, need_label):
    """Split rows into usable ones and disclosed exclusions.

    ``features`` maps each model input to its declared policy.  Exclusions, in
    order: the entity touches another window (the group is kept whole), a
    field is not yet populated at this stage, or the outcome is not yet
    determined.  Nothing is ever dropped merely to make the counts look
    better.
    """
    usable, excluded = [], []
    for row in data["rows"]:
        entity = _entity_of(row, data)
        if entity in spanning:
            excluded.append(
                {"key": row["key"], "entity": entity[1],
                 "reason": "entity spans more than one window; the group is excluded whole rather than split"}
            )
            continue
        unavailable = sorted(
            field for field, spec in features.items()
            if max(spec["available_from"],
                   data["fields"].get(field, spec)["available_from"]) > PHASE[stage]
        )
        if unavailable:
            excluded.append(
                {"key": row["key"], "reason": f"fields are not populated at the {stage} stage",
                 "fields": unavailable}
            )
            continue
        still_missing = sorted(
            field for field, spec in features.items()
            if row.get(field) is None and spec["missing"] == "error"
        )
        if still_missing:
            excluded.append(
                {"key": row["key"], "reason": "required field is missing and no imputation is declared",
                 "fields": still_missing}
            )
            continue
        if need_label:
            reason = _label_status(row, data)
            if reason is not None:
                excluded.append({"key": row["key"], "reason": reason})
                continue
        usable.append(row)
    return usable, excluded


# ---------------------------------------------------------------------------
# model artifact
# ---------------------------------------------------------------------------

def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _checksum(payload):
    body = {key: value for key, value in payload.items() if key != "checksum"}
    return hashlib.sha256(_canonical(body)).hexdigest()


def _features_from_specs(fields, specs):
    return {field: {"available_from": specs[field]["available_from"], "missing": specs[field]["missing"]}
            for field in fields}


def build_payload(terms, params, fields, standardizer, l2, training):
    payload = {
        "format": FORMAT,
        "version": MODEL_VERSION,
        "task": TASK,
        "terms": terms,
        "params": [float(value) for value in params],
        "features": fields,
        "standardizer": standardizer,
        "l2": float(l2),
        "training": training,
    }
    payload["checksum"] = _checksum(payload)
    return payload


class BinaryModel:
    """A validated classifier that reloads its own conditioning state."""

    def __init__(self, payload):
        if not isinstance(payload, dict):
            raise ConditionError("model must be an object")
        if payload.get("format") != FORMAT or payload.get("version") != MODEL_VERSION:
            raise ConditionError("unsupported model format")
        if payload.get("task") != TASK:
            raise ConditionError("model task is not binary classification")
        compiled, canonical = compile_terms(payload.get("terms"))
        features = feature_fields(compiled)
        declared = payload.get("features")
        if not isinstance(declared, dict) or set(declared) != set(features):
            raise ConditionError("model feature declarations must match its var terms exactly")
        parsed_fields = {}
        for field in features:
            spec = declared[field]
            if not isinstance(spec, dict) or set(spec) != {"available_from", "missing"}:
                raise ConditionError(f"feature declaration for {field!r} is malformed")
            available = spec["available_from"]
            if not isinstance(available, int) or isinstance(available, bool) or not 1 <= available <= 3:
                raise ConditionError(f"feature {field!r} available_from must be in 1..3")
            if spec["missing"] not in ("error", "mean"):
                raise ConditionError(f"feature {field!r} missing policy is invalid")
            parsed_fields[field] = {"available_from": available, "missing": spec["missing"]}
        params = payload.get("params")
        if not isinstance(params, list) or len(params) != len(compiled):
            raise ConditionError("model params must align with its terms")
        params = [_finite(value, f"model parameter {index}") for index, value in enumerate(params)]
        standardizer = Standardizer.load(payload.get("standardizer"), features)
        for field in features:
            policy = parsed_fields[field]["missing"]
            has_impute = standardizer.stats[field]["impute"] is not None
            if (policy == "mean") != has_impute:
                raise ConditionError(f"feature {field!r} missing policy disagrees with its standardizer")
        training = payload.get("training")
        if not isinstance(training, dict) or training.get("status") != "completed":
            raise ConditionError("only a completed fit may be handed off")
        l2 = _finite(payload.get("l2"), "l2")
        if l2 < 0.0:
            raise ConditionError("l2 must be non-negative")
        checksum = payload.get("checksum")
        if not isinstance(checksum, str) or checksum != _checksum(payload):
            raise ConditionError("model checksum does not match its contents")
        self.payload = json.loads(json.dumps(payload))
        self._compiled = compiled
        self._terms = json.loads(json.dumps(canonical))
        self._params = params
        self._features = features
        self._fields = parsed_fields
        self._standardizer = standardizer
        self._training = training
        self._l2 = l2
        self._names = parameter_names(compiled)

    # -- introspection ----------------------------------------------------
    @property
    def terms(self):
        return json.loads(json.dumps(self._terms))

    @property
    def expression(self):
        return {"terms": self.terms, "features": list(self._features), "params": list(self._params)}

    @property
    def checksum(self):
        return self.payload["checksum"]

    @property
    def features(self):
        return list(self._features)

    @property
    def parameter_names(self):
        return list(self._names)

    @property
    def training(self):
        return dict(self._training)

    def dump(self):
        return json.loads(json.dumps(self.payload))

    # -- inference --------------------------------------------------------
    def _availability(self, as_of):
        if not isinstance(as_of, int) or isinstance(as_of, bool) or not 1 <= as_of <= 3:
            raise ConditionError("as_of must be a prediction phase in 1..3")
        unavailable = [field for field in self._features if self._fields[field]["available_from"] > as_of]
        if unavailable:
            raise UnavailableFieldError(unavailable, as_of)
        return as_of

    def _values(self, row, as_of):
        self._availability(as_of)
        return self._standardizer.condition(self._features, row)

    def score(self, row, as_of=3):
        values, imputed = self._values(row, as_of)
        result = _evaluate(self._compiled, self._params, values)
        return {"score": result["value"], "probability": _sigmoid(result["value"]), "imputed": imputed}

    def parameter_gradients(self, row, as_of=3):
        values, imputed = self._values(row, as_of)
        result = _evaluate(self._compiled, self._params, values)
        return {
            "score": result["value"],
            "probability": _sigmoid(result["value"]),
            "gradients": dict(zip(self._names, result["gradients"])),
            "term_values": dict(enumerate(result["term_values"])),
            "imputed": imputed,
        }

    def recheck(self, row, as_of=3):
        """Analyst cross-check that goes through the scalar autodiff entry."""
        self._availability(as_of)
        return recheck(self.expression, row, self.payload["standardizer"])

    def predict_rows(self, rows, as_of=3):
        """Score a batch, separating legal missing values from unavailable fields."""
        if not isinstance(rows, list) or not rows:
            raise ConditionError("rows must be a non-empty list")
        if not isinstance(as_of, int) or isinstance(as_of, bool) or not 1 <= as_of <= 3:
            raise ConditionError("as_of must be a prediction phase in 1..3")
        scored, unavailable, invalid = [], [], []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ConditionError(f"prediction row {index} must be an object")
            key = row.get("key", index)
            try:
                result = self.score(row, as_of)
            except UnavailableFieldError as error:
                unavailable.append({"key": key, "fields": error.fields, "reason": "not populated at prediction time"})
                continue
            except ConditionError as error:
                invalid.append({"key": key, "reason": str(error)})
                continue
            scored.append(
                {"key": key, "score": result["score"], "probability": result["probability"],
                 "imputed": result["imputed"]}
            )
        return {"rows": scored, "unavailable": unavailable, "invalid": invalid}


def _sigmoid(value):
    """Stable logistic; still defined (and usable) at either extreme."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    scaled = math.exp(value)
    return scaled / (1.0 + scaled)


def load_model(payload):
    return BinaryModel(payload)


# ---------------------------------------------------------------------------
# losses and metrics
# ---------------------------------------------------------------------------

def _bce(score, label):
    """Binary cross entropy straight from the logit; no clipping of p."""
    return max(score, 0.0) - label * score + math.log1p(math.exp(-abs(score)))


def _auc(labels, probabilities):
    order = sorted(range(len(labels)), key=lambda i: probabilities[i])
    ranks, position = [0.0] * len(order), 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and probabilities[order[end + 1]] == probabilities[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for slot in range(position, end + 1):
            ranks[order[slot]] = average
        position = end + 1
    positives = sum(labels)
    negatives = len(labels) - positives
    rank_sum = math.fsum(ranks[i] for i, label in enumerate(labels) if label == 1.0)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(labels, probabilities):
    order = sorted(range(len(labels)), key=lambda i: -probabilities[i])
    positives = sum(labels)
    running, total = 0.0, 0.0
    for position, index in enumerate(order, start=1):
        running += labels[index]
        if labels[index] == 1.0:
            total += running / position
    return total / positives


def _calibration(labels, probabilities):
    order = sorted(range(len(labels)), key=lambda i: (probabilities[i], i))
    bins, size = [], max(1, math.ceil(len(order) / CALIBRATION_BINS))
    for start in range(0, len(order), size):
        group = order[start:start + size]
        if group:
            bins.append(group)
    rows, error = [], 0.0
    for group in bins:
        mean_probability = math.fsum(probabilities[i] for i in group) / len(group)
        observed = math.fsum(labels[i] for i in group) / len(group)
        gap = mean_probability - observed
        weight = len(group) / len(labels)
        error += abs(gap) * weight
        rows.append({"count": len(group), "mean_probability": mean_probability,
                     "observed_rate": observed, "gap": gap})
    return error, rows


def metrics(labels, scores):
    """Discrimination and calibration from logits (overflow-free)."""
    if not labels or len(labels) != len(scores):
        raise ConditionError("metrics need aligned labels and scores")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ConditionError("an evaluation slice must contain both classes")
    probabilities = [_sigmoid(score) for score in scores]
    calibration_error, bins = _calibration(labels, probabilities)
    return {
        "count": len(labels),
        "positives": int(positives),
        "negatives": int(negatives),
        "auc": _auc(labels, probabilities),
        "average_precision": _average_precision(labels, probabilities),
        "log_loss": math.fsum(_bce(score, label) for score, label in zip(scores, labels)) / len(labels),
        "brier": math.fsum((p - label) ** 2 for p, label in zip(probabilities, labels)) / len(labels),
        "calibration_error": calibration_error,
        "calibration_bins": bins,
    }


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

class FitOutcome:
    __slots__ = ("params", "status", "reason", "iterations", "evaluations", "loss", "gradient_norm")

    def __init__(self, params, status, reason, iterations, evaluations, loss, gradient_norm):
        self.params = params
        self.status = status
        self.reason = reason
        self.iterations = iterations
        self.evaluations = evaluations
        self.loss = loss
        self.gradient_norm = gradient_norm


def _objective(compiled, prepared, labels, weights, l2):
    """Return the penalised loss and gradient at ``weights``.

    The per-row derivative is exactly the one the shared scalar evaluator
    reports, so what fitting optimises and what an analyst re-derives are the
    same quantities.  Clipping the sigmoid probability here would erase the
    signal for confidently wrong rows on imbalanced data; working from the
    logit keeps that signal without ever overflowing.
    """
    count = len(compiled)
    gradient = [2.0 * l2 * weight for weight in weights]
    loss = math.fsum(l2 * weight * weight for weight in weights)
    for values, label in zip(prepared, labels):
        result = _evaluate(compiled, weights, values)
        score = result["value"]
        residual = _sigmoid(score) - label
        loss += _bce(score, label)
        derivatives = result["gradients"]
        for index in range(count):
            gradient[index] += residual * derivatives[index]
    return loss, gradient


def _fit(compiled, prepared, labels, l2, options):
    """Damped Newton (Gauss-Newton on the logistic Hessian) with line search.

    Returns ``status == 'completed'`` only when the gradient tolerance was
    met.  A step budget yields ``'stopped'`` and a domain failure yields
    ``'failed'``; in both cases the parameter vector is withheld, so no
    partial fit can enter selection or overwrite a validated candidate.
    """
    count = len(compiled)
    weights = [0.0] * count
    max_iter = options["max_iter"]
    max_evaluations = options["max_evaluations"]
    tolerance = options["grad_tol"]
    damping = options["damping"]
    evaluations = 0
    last_loss, last_norm = float("inf"), float("inf")
    try:
        loss, gradient = _objective(compiled, prepared, labels, weights, l2)
        evaluations += len(prepared)
        for iteration in range(1, max_iter + 1):
            norm = math.sqrt(math.fsum(value * value for value in gradient))
            last_loss, last_norm = loss, norm
            if not math.isfinite(norm) or not math.isfinite(loss):
                return FitOutcome(None, "failed", "objective became non-finite",
                                  iteration, evaluations, last_loss, last_norm)
            if norm <= tolerance:
                return FitOutcome(list(weights), "completed", "gradient tolerance met",
                                  iteration - 1, evaluations, loss, norm)
            if evaluations + 3 * len(prepared) > max_evaluations:
                return FitOutcome(None, "stopped",
                                  f"evaluation budget {max_evaluations} cannot finish another step",
                                  iteration - 1, evaluations, last_loss, last_norm)
            hessian = [[0.0] * count for _ in range(count)]
            for values, _ in zip(prepared, labels):
                result = _evaluate(compiled, weights, values)
                evaluations += 1
                curvature = _sigmoid(result["value"]) * (1.0 - _sigmoid(result["value"]))
                derivatives = result["gradients"]
                for i in range(count):
                    di = curvature * derivatives[i]
                    if di == 0.0:
                        continue
                    row = hessian[i]
                    for j in range(i + 1):
                        row[j] += di * derivatives[j]
            for i in range(count):
                hessian[i][i] += 2.0 * l2 + damping
                for j in range(i):
                    hessian[j][i] = hessian[i][j]
            step = _solve(hessian, [-value for value in gradient])
            if step is None:
                return FitOutcome(None, "failed", "Newton system is singular",
                                  iteration, evaluations, last_loss, last_norm)
            accepted = False
            scale = 1.0
            for _ in range(40):
                trial = [weights[i] + scale * step[i] for i in range(count)]
                if all(math.isfinite(value) for value in trial):
                    try:
                        trial_loss, trial_gradient = _objective(
                            compiled, prepared, labels, trial, l2
                        )
                    except ExpressionError:
                        trial_loss = float("inf")
                    evaluations += len(prepared)
                    if trial_loss < loss:
                        weights, loss, gradient = trial, trial_loss, trial_gradient
                        accepted = True
                        break
                scale *= 0.5
            if not accepted:
                return FitOutcome(None, "stopped", "line search could not reduce the objective",
                                  iteration, evaluations, last_loss, last_norm)
    except ExpressionError as error:
        return FitOutcome(None, "failed", f"evaluation failed: {error}",
                          iteration, evaluations, last_loss, last_norm)
    return FitOutcome(None, "stopped", f"iteration limit {max_iter} reached without convergence",
                      max_iter, evaluations, last_loss, last_norm)


def _solve(matrix, rhs):
    """Gaussian elimination with partial pivoting; ``None`` if singular."""
    size = len(rhs)
    augmented = [list(matrix[row]) + [rhs[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        factor = augmented[column][column]
        for entry in range(column, size + 1):
            augmented[column][entry] /= factor
        for row in range(size):
            if row != column and augmented[row][column]:
                multiple = augmented[row][column]
                for entry in range(column, size + 1):
                    augmented[row][entry] -= multiple * augmented[column][entry]
    return [augmented[row][size] for row in range(size)]


def _options(config):
    if not isinstance(config, dict):
        raise ConditionError("training configuration must be an object")
    def integer(key, default, limit):
        value = config.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= limit:
            raise ConditionError(f"{key} must be an integer within 1..{limit}")
        return value
    l2 = _finite(config.get("l2", DEFAULT_L2), "l2")
    if l2 < 0.0:
        raise ConditionError("l2 must be non-negative")
    tolerance = _finite(config.get("grad_tol", DEFAULT_GRAD_TOL), "grad_tol")
    if tolerance <= 0.0:
        raise ConditionError("grad_tol must be positive")
    damping = _finite(config.get("damping", DEFAULT_DAMPING), "damping")
    if damping <= 0.0:
        raise ConditionError("damping must be positive")
    return {
        "l2": l2,
        "grad_tol": tolerance,
        "damping": damping,
        "max_iter": integer("max_iter", DEFAULT_MAX_ITER, MAX_ITER),
        "max_evaluations": integer("max_evaluations", DEFAULT_MAX_EVALUATIONS, MAX_EVALUATIONS),
    }


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def train_model(dataset, config, spanning=()):
    """Fit a candidate on the training window only."""
    data = validate_dataset(dataset, "training dataset", stage="train")
    _require_role(data, "train", "training")
    options = _options(config)
    compiled, canonical = compile_terms(config.get("terms"))
    fields = feature_fields(compiled)
    for field in fields:
        if field not in data["fields"]:
            raise ConditionError(f"feature field {field!r} has no declaration in the training window")
        if data["fields"][field]["available_from"] > PHASE["train"]:
            raise ConditionError(
                f"feature field {field!r} is not populated at the row's prediction time"
            )
    label_spec = data["fields"].get(data["label_field"]) if data["label_field"] else None
    if label_spec is None:
        raise ConditionError("training dataset must declare its label field")
    declared_known = label_spec["outcome_known_at"]
    if declared_known is None or declared_known > PHASE["train"]:
        raise ConditionError(
            "the training outcome is not final at the close of the training window; "
            "rows whose outcome is not yet observed must not be used as negatives"
        )
    train_features = {field: {"available_from": data["fields"][field]["available_from"],
                              "missing": data["fields"][field]["missing"]} for field in fields}
    usable, excluded = prepare_rows(data, "train", set(spanning), train_features, need_label=True)
    if len(usable) < 2:
        raise ConditionError("fewer than two training rows remain after exclusions")
    labels = [float(row[data["label_field"]]) for row in usable]
    if len(set(labels)) < 2:
        raise ConditionError("training rows contain only one class")
    policies = {field: data["fields"][field]["missing"] for field in fields}
    standardizer = Standardizer.fit(usable, fields, policies)
    prepared = []
    for row in usable:
        values, _ = standardizer.condition(fields, row)
        prepared.append(values)
    outcome = _fit(compiled, prepared, labels, options["l2"], options)
    training = {
        "status": outcome.status,
        "reason": outcome.reason,
        "iterations": outcome.iterations,
        "evaluations": outcome.evaluations,
        "loss": None if not math.isfinite(outcome.loss) else outcome.loss,
        "gradient_norm": None if not math.isfinite(outcome.gradient_norm) else outcome.gradient_norm,
        "window": "train",
        "dataset": data["name"],
        "rows_used": len(usable),
        "options": {key: options[key] for key in ("l2", "grad_tol", "damping", "max_iter", "max_evaluations")},
    }
    record = {
        "task": TASK,
        "stage": "train",
        "window": "train",
        "dataset": data["name"],
        "term_id": json.dumps(canonical, separators=(",", ":")),
        "model_checksum": None,
        "model": None,
        "training": training,
        "status": outcome.status,
        "feature_fields": fields,
        "rows_used": [row["key"] for row in usable],
        "excluded_rows": excluded,
        "imputed_fields": _imputed_summary(usable, fields, standardizer),
    }
    if outcome.status != "completed":
        return record
    payload = build_payload(
        canonical,
        outcome.params,
        _features_from_specs(fields, data["fields"]),
        standardizer.dump(),
        options["l2"],
        training,
    )
    model = BinaryModel(payload)
    record["model"] = model.dump()
    record["model_checksum"] = model.checksum
    record["status"] = "completed"
    return record


def _imputed_summary(rows, fields, standardizer):
    counts = {}
    for row in rows:
        _, imputed = standardizer.condition(fields, row)
        for field in imputed:
            counts[field] = counts.get(field, 0) + 1
    return counts


def evaluate(model, dataset, stage, spanning=()):
    """Metrics of ``model`` on a labelled window, excluding unusable rows."""
    data = validate_dataset(dataset, f"{stage} dataset", stage=stage)
    usable, excluded = prepare_rows(
        data, stage, set(spanning),
        {field: model.payload["features"][field] for field in model.features},
        need_label=True,
    )
    if not usable:
        raise ConditionError(f"no scorable row with a determined outcome remains in {stage}")
    records, scores, labels = [], [], []
    for row in usable:
        try:
            result = model.score(row, as_of=PHASE[stage])
        except UnavailableFieldError as error:
            excluded.append({"key": row["key"], "reason": str(error), "fields": error.fields})
            continue
        records.append(
            {"key": row["key"], "score": result["score"], "probability": result["probability"],
             "label": float(row[data["label_field"]]), "imputed": result["imputed"]}
        )
        scores.append(result["score"])
        labels.append(float(row[data["label_field"]]))
    if not records:
        raise ConditionError(f"no scorable row remains in {stage}")
    result = metrics(labels, scores)
    result["dataset"] = data["name"]
    result["window"] = stage
    result["excluded_rows"] = excluded
    result["records"] = records
    result["model_checksum"] = model.checksum
    return result


def select_model(records, validation_dataset, config=None, previous=None, spanning=()):
    """Choose among completed fits using validation evidence only.

    Incomplete fits are refused outright: their parameters are never scored,
    never ranked, and never replace a previously validated candidate.
    """
    config = config or {}
    if not isinstance(config, dict):
        raise ConditionError("selection configuration must be an object")
    metric = config.get("metric", DEFAULT_SELECTION_METRIC)
    if metric not in METRICS:
        raise ConditionError(f"selection metric must be one of {sorted(METRICS)}")
    if not isinstance(records, list):
        records = [records]
    data = validate_dataset(validation_dataset, "selection dataset", stage="select")
    _require_role(data, "validation", "selection")
    candidates, refusals = [], []
    for record in records:
        if not isinstance(record, dict) or record.get("stage") != "train":
            raise ConditionError("selection takes training records")
        if record.get("status") != "completed" or record.get("model") is None:
            refusals.append(
                {"term_id": record.get("term_id"), "reason": "training did not complete; parameters withheld"}
            )
            continue
        model = BinaryModel(record["model"])
        if model.checksum != record.get("model_checksum"):
            raise ConditionError("training record model does not match its checksum")
        try:
            evaluation = evaluate(model, validation_dataset, "select", spanning)
        except ConditionError as error:
            refusals.append({"term_id": record["term_id"], "reason": f"validation failed: {error}"})
            continue
        candidates.append(
            {
                "term_id": record["term_id"],
                "model_checksum": model.checksum,
                "model": model.dump(),
                "metrics": {key: value for key, value in evaluation.items() if key not in ("records",)},
            }
        )
    chosen = _best(candidates, metric)
    retained = None
    if chosen is None:
        if previous is not None:
            retained = BinaryModel(previous["selected_model"])
            chosen = {"term_id": previous.get("selected_term_id"),
                      "model_checksum": retained.checksum,
                      "model": retained.dump(), "metrics": previous.get("selected_metrics")}
        else:
            raise ConditionError("no candidate produced a usable validation evaluation")
    status = "completed"
    if isinstance(config.get("stop_after"), int) and not isinstance(config.get("stop_after"), bool):
        if config["stop_after"] < len(candidates):
            status = "stopped"
    return {
        "task": TASK,
        "stage": "select",
        "status": status,
        "window": "select",
        "dataset": data["name"],
        "metric": metric,
        "lower_is_better": metric not in MAXIMISE,
        "selected_model": chosen["model"],
        "selected_model_checksum": chosen["model_checksum"],
        "selected_term_id": chosen["term_id"],
        "selected_metrics": chosen["metrics"],
        "candidates": [
            {"term_id": item["term_id"], "model_checksum": item["model_checksum"], "metrics": item["metrics"]}
            for item in candidates
        ],
        "refused_candidates": refusals,
        "retained_previous": retained is not None,
        "selection_evidence": "validation split only; holdout labels are not consulted",
    }


def _best(candidates, metric):
    valid = [item for item in candidates if isinstance(item.get("metrics"), dict)
             and isinstance(item["metrics"].get(metric), (int, float))]
    if not valid:
        return None
    if metric in MAXIMISE:
        return max(valid, key=lambda item: item["metrics"][metric])
    return min(valid, key=lambda item: item["metrics"][metric])


def run_holdout(selection, holdout_dataset, spanning=()):
    """Evaluate the already selected artifact once; never reselect."""
    if not isinstance(selection, dict) or selection.get("stage") != "select":
        raise ConditionError("final evaluation requires a selection record")
    if selection.get("status") != "completed":
        raise ConditionError("final evaluation requires a completed selection")
    model = BinaryModel(selection["selected_model"])
    if model.checksum != selection.get("selected_model_checksum"):
        raise ConditionError("the selected model failed its integrity check")
    checked = validate_dataset(holdout_dataset, "holdout dataset", stage="holdout")
    _require_role(checked, "holdout", "final evaluation")
    evaluation = evaluate(model, holdout_dataset, "holdout", spanning)
    return {
        "task": TASK,
        "stage": "holdout",
        "status": "completed",
        "window": "holdout",
        "dataset": evaluation.get("dataset", holdout_dataset.get("name")),
        "selected_model_checksum": model.checksum,
        "selected_term_id": selection["selected_term_id"],
        "metrics": {key: value for key, value in evaluation.items() if key not in ("records",)},
        "records": [
            {"key": item["key"], "score": item["score"], "probability": item["probability"],
             "label": item["label"]}
            for item in evaluation["records"]
        ],
        "excluded_rows": evaluation["excluded_rows"],
        "locked": {
            "selection_metric": selection["metric"],
            "selection_dataset": selection["dataset"],
            "holdout_cannot_reselect": True,
            "metrics_used_only_for_evaluation": True,
        },
    }


def run_pipeline(request):
    """Train, select, then evaluate the selected model on the final holdout."""
    if not isinstance(request, dict):
        raise ConditionError("request must be an object")
    if request.get("task") != TASK:
        raise ConditionError(f"task must be {TASK!r}")
    train_data = validate_dataset(request.get("train"), "training dataset")
    validation_data = validate_dataset(request.get("validation"), "selection dataset")
    holdout_data = validate_dataset(request.get("holdout"), "holdout dataset")
    if train_data["window"] != "train" or validation_data["window"] != "select" or holdout_data["window"] != "holdout":
        raise ConditionError("train, select and holdout windows must be distinct and correctly assigned")
    _check_periods(train_data, validation_data, holdout_data)
    _require_role(train_data, "train", "training")
    _require_role(validation_data, "validation", "selection")
    _require_role(holdout_data, "holdout", "final evaluation")
    for data, label in ((train_data, "training"), (validation_data, "selection"), (holdout_data, "holdout")):
        if data["label_field"] != train_data["label_field"]:
            raise ConditionError("all windows must label the same outcome field")
    spanning = _spanning_entities([train_data, validation_data, holdout_data])
    configs = request.get("candidates", [request.get("model", {})])
    if not isinstance(configs, list) or not configs:
        raise ConditionError("candidates must be a non-empty list")
    training_records = [train_model(request["train"], config, spanning) for config in configs]
    selection = select_model(
        training_records, request["validation"], request.get("selection", {}),
        previous=request.get("previous_selection"), spanning=spanning,
    )
    holdout = run_holdout(selection, request["holdout"], spanning)
    return {
        "task": TASK,
        "status": "completed",
        "stages": {"train": training_records, "select": selection, "holdout": holdout},
        "selected_model_checksum": selection["selected_model_checksum"],
        "holdout_metrics": holdout["metrics"],
        "handoff": {
            "model": selection["selected_model"],
            "model_checksum": selection["selected_model_checksum"],
            "verified_by_holdout": holdout["selected_model_checksum"] == selection["selected_model_checksum"],
        },
        "disclosures": {
            "windows": {"train": "train", "select": "select", "holdout": "holdout"},
            "grouping": "records from one entity are excluded whole; no group is split across windows",
            "holdout_boundary": "fixed by the declared windows and periods and never moved",
            "standardization": "location and scale are fitted from training rows only",
            "selection_used_holdout_labels": False,
            "final_metrics_influence_selection": False,
            "excluded_entities": sorted(entity[1] for entity in spanning),
        },
    }


# ---------------------------------------------------------------------------
# a runnable demonstration request
# ---------------------------------------------------------------------------

def demonstration_model():
    """A tiny fitted model for the service examples."""
    return build_payload(
        [["intercept"], ["var", "x"], ["mul", 1]],
        [0.1, 0.6, -0.2],
        {"x": {"available_from": 1, "missing": "mean"}},
        {"x": {"mean": 0.0, "scale": 1.0, "min": -3.0, "max": 3.0, "impute": 0.0}},
        1e-3,
        {"status": "completed", "reason": "gradient tolerance met", "iterations": 4, "evaluations": 40,
         "loss": 0.6, "gradient_norm": 1e-8, "window": "train", "dataset": "demo", "rows_used": 10},
    )


def demonstration_request():
    """A small deterministic three-window request used by the service demo."""
    def window(role, stage, seed, period):
        state = RandomStream(seed)
        rows = []
        for index in range(40):
            x = state.normal()
            z = state.normal()
            probability = _sigmoid(1.4 * x - 0.7 * z + 0.1)
            label = int(state.uniform() < probability)
            rows.append({"key": f"{role}-{index}", "entity": f"{role}-group-{index}",
                         "x": round(x, 6), "z": round(z, 6), "y": label,
                         "outcome_known_at": PHASE[stage]})
        return {
            "name": f"{role}-{stage}",
            "role": role,
            "window": stage,
            "period": period,
            "label_field": "y",
            "fields": [
                {"name": "x", "available_from": 1},
                {"name": "z", "available_from": 1},
                {"name": "y", "available_from": 1, "outcome_known_at": PHASE[stage]},
            ],
            "rows": rows,
        }

    return {
        "task": TASK,
        "train": window("train", "train", 11, {"start": 1, "end": 90}),
        "validation": window("validation", "select", 12, {"start": 91, "end": 160}),
        "holdout": window("holdout", "holdout", 13, {"start": 161, "end": 230}),
        "candidates": [
            {"terms": [["intercept"], ["var", "x"]], "max_iter": 100},
            {"terms": [["intercept"], ["var", "x"], ["var", "z"]], "max_iter": 100},
        ],
        "selection": {"metric": "log_loss"},
    }


# ---------------------------------------------------------------------------
# hand-off scoring
# ---------------------------------------------------------------------------

def score_batch(model_payload, dataset, as_of=PHASE["predict"]):
    """Score the next batch with the reloaded artifact.

    The model's own field constraints decide what is legal: a missing value
    covered by an imputation policy is filled and reported, while a field that
    is not yet populated at ``as_of`` makes the row unavailable rather than
    silently imputed.
    """
    model = BinaryModel(model_payload)
    data = validate_dataset(dataset, "prediction batch")
    if data["role"] != "batch":
        raise ConditionError("batch scoring expects a dataset role='batch'")
    result = model.predict_rows(data["rows"], as_of)
    return {
        "task": TASK,
        "stage": "predict",
        "status": "completed",
        "model_checksum": model.checksum,
        "as_of": as_of,
        **result,
    }
