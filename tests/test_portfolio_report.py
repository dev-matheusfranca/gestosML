from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import sqlite3

import pytest

from scripts.export_portfolio_report import build_report, main


@pytest.fixture
def report_db(tmp_path):
    """Synthetic data only. Free-text sentinels must never leave the exporter."""
    path = tmp_path / "coleta privada #1.sqlite3"
    labels = ["private-label-z", "private-label-a"]
    stage = {
        "n": 4, "accuracy": 0.75, "macro_f1": 0.7333333333333334, "coverage": 0.5,
        "accepted_accuracy": 1.0, "accepted_macro_f1": 1.0,
        "classes": labels, "confusion_matrix": [[2, 0], [1, 1]],
        "per_class": {
            labels[0]: {"precision": 2 / 3, "recall": 1.0, "f1-score": 0.8, "support": 2},
            labels[1]: {"precision": 1.0, "recall": 0.5, "f1-score": 2 / 3, "support": 2},
        },
        "private_extra": "private-stage-value",
    }
    metrics = {
        "validation": stage, "final": stage, "split": {"train": 4, "validation": 4, "test_reserved": 4},
        "latency_ms": 0.5, "challenge": None, "warnings": ["private-warning"],
    }
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE experiments (id, model, status, dataset_id, lineage_id, strategy, protocol,
                seed, metrics_json, threshold, duration_s, artifact_path, environment_json);
            CREATE TABLE snapshots (id, rows);
            CREATE TABLE final_evaluations (lineage_id, experiment_id, report_json);
            CREATE TABLE samples (label, session_id, landmarks_json);
            CREATE TABLE sessions (id, participant);
            CREATE TABLE difficult (status, predicted_label);
        """)
        db.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            "private-run", "forest", "completed", "private-dataset", "private-lineage", "normalized",
            "session", 42, json.dumps(metrics), 0.8, 1.2, "private-path", "private-environment",
        ))
        db.execute("INSERT INTO snapshots VALUES (?,?)", ("private-dataset", 12))
        db.execute("INSERT INTO final_evaluations VALUES (?,?,?)", ("private-lineage", "private-run", json.dumps(stage)))
        db.execute("INSERT INTO sessions VALUES (?,?)", ("private-session", "private-participant"))
        db.execute("INSERT INTO sessions VALUES (?,?)", ("private-empty", "private-participant"))
        db.execute("INSERT INTO samples VALUES (?,?,?)", (labels[0], "private-session", "private-coordinates"))
        db.execute("INSERT INTO difficult VALUES (?,?)", ("pending", "private-predicted"))
    return path


def mutate_metrics(path, change):
    with sqlite3.connect(path) as db:
        metrics = json.loads(db.execute("SELECT metrics_json FROM experiments LIMIT 1").fetchone()[0])
        change(metrics)
        db.execute("UPDATE experiments SET metrics_json=?", (json.dumps(metrics),))


def test_public_report_is_readonly_sanitized_and_reorders_both_matrix_axes(report_db, capsys):
    before = hashlib.sha256(report_db.read_bytes()).hexdigest()
    assert main(["--database", str(report_db)]) == 0
    output = capsys.readouterr().out
    assert "private-" not in output
    for field in ("participant", "landmarks_json", "artifact_path", "environment_json", "dataset_id", "lineage_id"):
        assert field not in output
    report = json.loads(output)
    assert report["current_collection"]["empty_sessions"] == 1
    validation = report["comparison"]["experiments"][0]["validation"]
    assert validation["confusion_matrix"] == [[1, 1], [0, 2]]
    assert validation["per_class"]["class_1"]["recall"] == 0.5
    assert (validation["accepted"], validation["rejected"], validation["accepted_correct"]) == (2, 2, 2)
    assert hashlib.sha256(report_db.read_bytes()).hexdigest() == before


def test_readonly_uri_disallows_writes(report_db, monkeypatch):
    real_connect = sqlite3.connect

    def readonly_connect(database_uri, **kwargs):
        assert database_uri.endswith("?mode=ro") and kwargs["uri"] is True
        connection = real_connect(database_uri, **kwargs)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM samples")
        connection.rollback()
        return connection

    monkeypatch.setattr(sqlite3, "connect", readonly_connect)
    assert build_report(report_db)["schema_version"] == 1


def test_missing_database_is_not_created(tmp_path, capsys):
    path = tmp_path / "missing.sqlite3"
    assert main(["--database", str(path)]) == 2
    assert not path.exists()
    assert str(path) not in capsys.readouterr().err


@pytest.mark.parametrize("column,value", [
    ("seed", 99), ("dataset_id", "other"), ("lineage_id", "other"),
    ("protocol", "participant"), ("strategy", "raw"), ("model", "forest"),
])
def test_incompatible_or_duplicate_runs_are_rejected(report_db, column, value):
    with sqlite3.connect(report_db) as db:
        db.execute("CREATE TEMP TABLE another AS SELECT * FROM experiments")
        db.execute("UPDATE another SET id='another', model='logistic'")
        db.execute(f"UPDATE another SET {column}=?", (value,))  # fixed parameterized test cases only
        db.execute("INSERT INTO experiments SELECT * FROM another")
    with pytest.raises(ValueError, match="ambígua"):
        build_report(report_db)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1, 1.2, True, "private-text"])
def test_invalid_metrics_never_reach_public_output(report_db, invalid, capsys):
    mutate_metrics(report_db, lambda m: m["validation"].update(accuracy=invalid))
    assert main(["--database", str(report_db)]) == 2
    output = capsys.readouterr()
    assert not output.out and "private-" not in output.err


def test_inconsistent_split_is_rejected(report_db):
    mutate_metrics(report_db, lambda m: m["split"].update(train=99))
    with pytest.raises(ValueError, match="snapshot"):
        build_report(report_db)


@pytest.mark.parametrize("statement", [
    "UPDATE final_evaluations SET experiment_id='other'",
    "UPDATE final_evaluations SET report_json='{}'",
    "DELETE FROM final_evaluations",
])
def test_final_must_agree_with_persisted_claim(report_db, statement):
    with sqlite3.connect(report_db) as db:
        db.execute(statement)
    with pytest.raises(ValueError):
        build_report(report_db)


def test_inconsistent_matrix_is_rejected(report_db):
    mutate_metrics(report_db, lambda m: m["validation"].update(confusion_matrix=[[99, 0], [0, 0]]))
    with pytest.raises(ValueError, match="matriz"):
        build_report(report_db)


@pytest.mark.parametrize("value", [None, {}])
def test_claimed_final_cannot_be_empty_even_when_both_records_match(report_db, value):
    mutate_metrics(report_db, lambda m: m.update(final=value))
    with sqlite3.connect(report_db) as db:
        db.execute("UPDATE final_evaluations SET report_json=?", (json.dumps(value),))
    with pytest.raises(ValueError, match="sem métricas"):
        build_report(report_db)


def test_duplicate_final_claims_are_rejected_even_with_nonstandard_schema(report_db):
    # The application schema has a primary key. Defend against malformed external copies too.
    with sqlite3.connect(report_db) as db:
        db.execute("INSERT INTO final_evaluations SELECT * FROM final_evaluations")
    with pytest.raises(ValueError, match="Mais de uma"):
        build_report(report_db)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_zero_coverage_cannot_claim_conditional_accuracy(report_db, value):
    mutate_metrics(report_db, lambda m: m["validation"].update(
        coverage=0, accepted_accuracy=value, accepted_macro_f1=value,
    ))
    with pytest.raises(ValueError, match="sem nenhuma"):
        build_report(report_db)


def test_zero_coverage_exports_null_conditional_metrics(report_db):
    mutate_metrics(report_db, lambda m: m["validation"].update(
        coverage=0, accepted_accuracy=None, accepted_macro_f1=None,
    ))
    stage = build_report(report_db)["comparison"]["experiments"][0]["validation"]
    assert stage["accepted_accuracy"] is None and stage["accepted_macro_f1"] is None
    assert stage["accepted"] == stage["accepted_correct"] == 0


def test_comparison_before_final_evaluation_can_be_exported(report_db):
    mutate_metrics(report_db, lambda m: m.update(final=None))
    with sqlite3.connect(report_db) as db:
        db.execute("DELETE FROM final_evaluations")
    assert build_report(report_db)["comparison"]["experiments"][0]["final"] is None


def test_accepted_correct_cannot_exceed_global_correct(report_db):
    mutate_metrics(report_db, lambda m: m["validation"].update(coverage=1.0))
    with pytest.raises(ValueError, match="excedem"):
        build_report(report_db)


def test_check_mode_detects_stale_report_without_overwriting(report_db, tmp_path, capsys):
    report = build_report(report_db)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert main(["--database", str(report_db), "--check", str(path)]) == 0
    stale = deepcopy(report)
    stale["current_collection"]["samples"] += 1
    path.write_text(json.dumps(stale), encoding="utf-8")
    before = path.read_bytes()
    assert main(["--database", str(report_db), "--check", str(path)]) == 1
    assert path.read_bytes() == before
    assert "não corresponde" in capsys.readouterr().err
