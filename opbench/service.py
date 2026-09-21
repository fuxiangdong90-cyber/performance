"""SQLite persistence and comparison domain. One connection per request."""
import datetime as dt
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from .catalog import canonical, normalize_case, workload

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);
INSERT INTO schema_version SELECT 1 WHERE NOT EXISTS(SELECT 1 FROM schema_version);
CREATE TABLE IF NOT EXISTS devices(id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs(
 id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
 device_id TEXT NOT NULL REFERENCES devices(id), metadata TEXT NOT NULL,
 synthetic INTEGER NOT NULL DEFAULT 0, archived INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS cases(case_key TEXT PRIMARY KEY, operator TEXT NOT NULL, category TEXT NOT NULL, definition TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS measurements(
 run_id TEXT NOT NULL REFERENCES runs(id), case_key TEXT NOT NULL REFERENCES cases(case_key),
 status TEXT NOT NULL, result TEXT NOT NULL, PRIMARY KEY(run_id,case_key));
CREATE INDEX IF NOT EXISTS measurements_case ON measurements(case_key);
CREATE INDEX IF NOT EXISTS cases_operator ON cases(operator);
"""


@contextmanager
def connect(path):
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    try:
        with db:
            yield db
    finally:
        db.close()


def initialize(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)


def validate_report(report):
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("expected OpBench schema_version=1; see docs/report-format.md")
    run = report.get("run", {})
    if not isinstance(run, dict) or not isinstance(run.get("name"), str) or not 1 <= len(run["name"]) <= 300:
        raise ValueError("run.name is required (1..300 characters)")
    if not isinstance(run.get("device"), dict) or not run["device"].get("name"):
        raise ValueError("run.device.name is required")
    if run.get("timing_method") != "synchronized_wall_per_iteration":
        raise ValueError("timing_method must be synchronized_wall_per_iteration")
    records = report.get("results")
    if not isinstance(records, list) or not 1 <= len(records) <= 20000:
        raise ValueError("results must contain 1..20000 cases")
    normalized, seen = [], set()
    for raw in records:
        case = normalize_case(raw)
        key = case["case_key"]
        if key in seen:
            raise ValueError(f"duplicate case {key}")
        seen.add(key)
        status = raw.get("status", "pass")
        if status not in ("pass", "failed", "unsupported", "oom"):
            raise ValueError("invalid result status")
        result = {**case, **workload(case), "status": status, "error": str(raw.get("error", ""))[:4000],
                  "correctness": raw.get("correctness", "not_checked")}
        for metric in ("wall_us", "gpu_us", "cpu_us", "p95_us", "stddev_us", "peak_allocated_bytes", "peak_delta_bytes"):
            value = raw.get(metric)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f"invalid metric {metric}")
            result[metric] = value
        if status == "pass" and (not result["wall_us"] or (result["gpu_us"] is not None and result["gpu_us"] <= 0)):
            raise ValueError("passing result requires positive wall_us and positive gpu_us if present")
        samples = raw.get("samples_us", [])
        if not isinstance(samples, list) or len(samples) > 10000 or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError("invalid samples_us")
        result["samples_us"] = samples
        if raw.get("reference") is not None:
            ref = raw["reference"]
            if not isinstance(ref, dict) or ref.get("device") != "cpu" or ref.get("implementation") != "native":
                raise ValueError("invalid reference metadata")
            fields = ("rtol", "atol", "max_abs_error", "max_scaled_error")
            if any(type(ref.get(k)) not in (int, float) or not math.isfinite(ref[k]) or ref[k] < 0 for k in fields):
                raise ValueError("invalid reference error/tolerance")
            checked_pass = raw.get("correctness") == "cpu_reference" and status == "pass" and ref["max_scaled_error"] <= 1
            checked_fail = raw.get("correctness") == "cpu_reference_failed" and status == "failed" and ref["max_scaled_error"] > 1
            if ref["atol"] <= 0 or not (checked_pass or checked_fail):
                raise ValueError("reference check and result status disagree")
            result["reference"] = {"device": "cpu", "implementation": "native", **{k: ref[k] for k in fields}}
            if "compute_dtype" in ref:
                if ref["compute_dtype"] != "float64" or ref.get("input_dtype") != case["dtype"]:
                    raise ValueError("invalid high precision reference dtype")
                result["reference"].update(compute_dtype="float64",input_dtype=ref["input_dtype"])
        elif raw.get("correctness") in ("cpu_reference", "cpu_reference_failed"):
            raise ValueError("cpu_reference requires reference metadata")
        result["tflops"] = result["flops"] / result["wall_us"] / 1e6 if status == "pass" and result["flops"] is not None else None
        result["bandwidth_gbs"] = result["bytes"] / result["wall_us"] / 1000 if status == "pass" and result["bytes"] is not None else None
        normalized.append(result)
    # Ensure metadata is JSON serializable and contains no NaN.
    canonical(run)
    return run, normalized


def import_report(path, report):
    run, records = validate_report(report)
    rid = str(uuid.uuid4())
    import hashlib
    device_id = hashlib.sha256(canonical(run["device"]).encode()).hexdigest()[:24]
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    with connect(path) as db:
        db.execute("INSERT OR IGNORE INTO devices VALUES (?,?)", (device_id, canonical(run["device"])))
        db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,0)", (rid, run["name"], created, device_id, canonical(run), int(bool(run.get("synthetic", False)))))
        for result in records:
            case = normalize_case(result)
            db.execute("INSERT OR IGNORE INTO cases VALUES (?,?,?,?)", (case["case_key"], case["operator"], case["category"], canonical(case)))
            db.execute("INSERT INTO measurements VALUES (?,?,?,?)", (rid, case["case_key"], result["status"], canonical(result)))
    return {"id": rid, "count": len(records)}


def list_runs(path, archived=False):
    with connect(path) as db:
        rows = db.execute("SELECT r.*, COUNT(m.case_key) count, SUM(m.status!='pass') failures FROM runs r LEFT JOIN measurements m ON m.run_id=r.id WHERE r.archived=? GROUP BY r.id ORDER BY r.created_at DESC", (int(archived),)).fetchall()
    return [{**dict(row), "metadata": json.loads(row["metadata"])} for row in rows]


def export_report(path, rid):
    with connect(path) as db:
        run = db.execute("SELECT metadata FROM runs WHERE id=?", (rid,)).fetchone()
        if not run:
            raise KeyError("run not found")
        results = [json.loads(r[0]) for r in db.execute("SELECT result FROM measurements WHERE run_id=? ORDER BY case_key", (rid,))]
    return {"schema_version": 1, "run": json.loads(run[0]), "results": results}


def archive_run(path, rid, archived):
    with connect(path) as db:
        cursor = db.execute("UPDATE runs SET archived=? WHERE id=?", (int(archived), rid))
        if cursor.rowcount == 0:
            raise KeyError("run not found")


def compare(path, left_id, right_id):
    left_report, right_report = export_report(path, left_id), export_report(path, right_id)
    left = {r["case_key"]: r for r in left_report["results"]}
    right = {r["case_key"]: r for r in right_report["results"]}
    rows = []
    for key in sorted(left.keys() | right.keys()):
        l, r = left.get(key), right.get(key)
        c = l or r
        paired = bool(l and r and l["status"] == r["status"] == "pass")
        row = {k: c[k] for k in ("case_key", "name", "operator", "category", "params", "dtype", "stage", "module_mode", "execution", "flops", "bytes", "arithmetic_intensity", "workload_note")}
        row["implementation"] = c.get("implementation", "native")
        row.update(left=l, right=r, paired=paired, speedup=l["wall_us"]/r["wall_us"] if paired else None,
                   gpu_speedup=l["gpu_us"]/r["gpu_us"] if paired and l["gpu_us"] and r["gpu_us"] else None)
        rows.append(row)
    warnings = []
    for field in ("torch_version", "backend_version", "precision_policy", "precision_settings", "input_initialization", "operator_implementations", "reference_method", "timing_method", "cpu_threads", "warmup", "iterations"):
        if left_report["run"].get(field) != right_report["run"].get(field):
            warnings.append(f"{field}: {left_report['run'].get(field, 'unknown')} → {right_report['run'].get(field, 'unknown')}")
    if left_report["run"].get("synthetic") or right_report["run"].get("synthetic"):
        warnings.insert(0, "SYNTHETIC: 演示数据，不代表任何硬件实测性能")
    return {"rows": rows, "warnings": warnings, "left": left_report["run"], "right": right_report["run"]}
