"""Export saved, allowlisted metrics without opening models or modifying the database."""

from __future__ import annotations

import argparse
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import sys


MODELS = ("dummy", "logistic", "forest", "mlp")
STRATEGIES = ("raw", "normalized", "distances", "angles")
PROTOCOLS = ("session", "participant")


def number(value, *, ratio=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Métrica numérica inválida.")
    if value < 0 or (ratio and value > 1) or (integer and int(value) != value):
        raise ValueError("Métrica fora do intervalo permitido.")
    return int(value) if integer else value


def measured_count(value):
    rounded = round(value)
    if not math.isclose(value, rounded, abs_tol=1e-7):
        raise ValueError("Proporção incompatível com a contagem de exemplos.")
    return rounded


def evaluation(raw, labels):
    """Replace free-text labels with stable aliases, including both matrix axes."""
    classes = raw["classes"]
    if len(classes) != len(labels) or set(classes) != set(labels):
        raise ValueError("As classes dos relatórios não coincidem.")
    n = number(raw["n"], integer=True)
    if not n:
        raise ValueError("Avaliação sem exemplos.")
    result = {key: number(raw[key], ratio=True) for key in ("accuracy", "macro_f1", "coverage")}
    accepted = measured_count(result["coverage"] * n)
    for key in ("accepted_accuracy", "accepted_macro_f1"):
        if accepted == 0:
            if raw[key] is not None:
                raise ValueError("Métrica condicional definida sem nenhuma previsão aceita.")
            result[key] = None
        else:
            result[key] = number(raw[key], ratio=True)
    result.update(n=n, accepted=accepted, rejected=n - accepted)
    result["accepted_correct"] = (
        measured_count(result["accepted_accuracy"] * accepted) if accepted else 0
    )
    matrix = raw["confusion_matrix"]
    if len(matrix) != len(labels) or any(len(row) != len(labels) for row in matrix):
        raise ValueError("Dimensões inválidas na matriz de confusão.")
    order = [classes.index(label) for label in labels]
    result["classes"] = [f"class_{index + 1}" for index in range(len(labels))]
    result["confusion_matrix"] = [[number(matrix[i][j], integer=True) for j in order] for i in order]
    if sum(map(sum, result["confusion_matrix"])) != n:
        raise ValueError("A matriz de confusão não corresponde ao total avaliado.")
    correct = sum(result["confusion_matrix"][i][i] for i in range(len(labels)))
    if correct != measured_count(result["accuracy"] * n):
        raise ValueError("A acurácia não corresponde à matriz de confusão.")
    if result["accepted_correct"] > correct or accepted - result["accepted_correct"] > n - correct:
        raise ValueError("Acertos e erros aceitos excedem os totais da matriz de confusão.")
    result["per_class"] = {}
    for alias, label, row in zip(result["classes"], labels, result["confusion_matrix"], strict=True):
        metrics = raw["per_class"][label]
        values = {key: number(metrics[key], ratio=True) for key in ("precision", "recall", "f1-score")}
        values["support"] = number(metrics["support"], integer=True)
        if values["support"] != sum(row):
            raise ValueError("Suporte por classe incompatível com a matriz de confusão.")
        result["per_class"][alias] = values
    return result


def build_report(database: Path):
    database = database.resolve(strict=True)
    # Do not instantiate Store: its constructor creates directories and runs migrations.
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")  # one coherent read snapshot, without changing stored data
        rows = connection.execute(
            "SELECT id, model, dataset_id, lineage_id, strategy, protocol, seed, metrics_json, "
            "threshold, duration_s FROM experiments WHERE status='completed' ORDER BY model"
        ).fetchall()
        if not rows:
            raise ValueError("Não há experimentos concluídos para exportar.")
        cohorts = {tuple(row[key] for key in ("dataset_id", "lineage_id", "strategy", "protocol", "seed"))
                   for row in rows}
        if len(cohorts) != 1 or len({row["model"] for row in rows}) != len(rows):
            raise ValueError("Comparação ambígua: requer uma única versão, configuração e execução por modelo.")
        first = rows[0]
        if (any(row["model"] not in MODELS for row in rows) or first["strategy"] not in STRATEGIES
                or first["protocol"] not in PROTOCOLS):
            raise ValueError("Modelo, estratégia ou protocolo não permitido no relatório público.")
        metrics = {row["model"]: json.loads(row["metrics_json"]) for row in rows}
        labels = sorted(metrics[first["model"]]["validation"]["classes"])
        if not labels or len(set(labels)) != len(labels) or not all(isinstance(label, str) for label in labels):
            raise ValueError("Classes inválidas.")
        split = {key: number(metrics[first["model"]]["split"][key], integer=True)
                 for key in ("train", "validation", "test_reserved")}
        if any(metrics[row["model"]]["split"] != split for row in rows):
            raise ValueError("As partições dos experimentos não coincidem.")
        snapshot = connection.execute("SELECT rows FROM snapshots WHERE id=?", (first["dataset_id"],)).fetchone()
        if not snapshot or number(snapshot["rows"], integer=True) != sum(split.values()):
            raise ValueError("A comparação deve usar um snapshot de desenvolvimento compatível com a divisão.")
        finals = connection.execute(
            "SELECT experiment_id, report_json FROM final_evaluations WHERE lineage_id=? LIMIT 2",
            (first["lineage_id"],),
        ).fetchall()
        if len(finals) > 1:
            raise ValueError("Mais de uma avaliação final registrada para a mesma linhagem.")
        final = finals[0] if finals else None
        if final and final["experiment_id"] not in {row["id"] for row in rows}:
            raise ValueError("A avaliação final pertence a uma execução fora desta comparação.")
        experiments = []
        for row in sorted(rows, key=lambda row: MODELS.index(row["model"])):
            saved = metrics[row["model"]]
            validation = evaluation(saved["validation"], labels)
            if validation["n"] != split["validation"]:
                raise ValueError("Validação incompatível com a divisão.")
            has_final = final is not None and final["experiment_id"] == row["id"]
            final_raw = json.loads(final["report_json"]) if has_final else None
            if has_final and (not isinstance(final_raw, dict) or not final_raw):
                raise ValueError("Avaliação final registrada sem métricas válidas.")
            if saved["final"] != final_raw:
                raise ValueError("O registro final não coincide com o relatório do experimento.")
            final_metrics = evaluation(final_raw, labels) if has_final else None
            if final_metrics and final_metrics["n"] != split["test_reserved"]:
                raise ValueError("Avaliação final incompatível com a divisão.")
            experiments.append({
                "model": row["model"], "threshold": number(row["threshold"], ratio=True),
                "training_duration_s": number(row["duration_s"]),
                "classifier_latency_ms": number(saved["latency_ms"]),
                "validation": validation, "final": final_metrics,
                "challenge_recorded": saved.get("challenge") is not None,
            })
        # Current collection is deliberately separate from the historical experiment snapshot.
        def scalar(query):
            return connection.execute(query).fetchone()[0]
        return {
            "schema_version": 1,
            "source": "saved_local_metrics_not_rerun",
            "current_collection": {
                "samples": scalar("SELECT COUNT(*) FROM samples"),
                "classes": scalar("SELECT COUNT(DISTINCT label) FROM samples"),
                "populated_sessions": scalar("SELECT COUNT(DISTINCT session_id) FROM samples"),
                "empty_sessions": scalar("SELECT COUNT(*) FROM sessions WHERE id NOT IN "
                                         "(SELECT session_id FROM samples)"),
                "difficult": {status: connection.execute("SELECT COUNT(*) FROM difficult WHERE status=?", (status,))
                              .fetchone()[0] for status in ("pending", "reviewed", "discarded")},
            },
            "comparison": {
                "same_dataset_and_lineage": True,
                "snapshot_samples": snapshot["rows"], "class_count": len(labels),
                "strategy": first["strategy"], "protocol": first["protocol"],
                "seed": number(first["seed"], integer=True), "split": split,
                "experiments": experiments,
            },
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Exporta métricas salvas, sem dados pessoais nem reavaliação.")
    parser.add_argument("--database", type=Path, default=Path("data/gesturelab.sqlite3"))
    parser.add_argument("--check", type=Path, help="Compara com um JSON já exportado, sem sobrescrevê-lo.")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.database)
        if args.check:
            if json.loads(args.check.read_text(encoding="utf-8")) != report:
                print("O relatório não corresponde ao estado atual; revise antes de atualizar a documentação.", file=sys.stderr)
                return 1
            print("Relatório agregado confere com as métricas salvas.")
        else:
            print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
        return 0
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, IndexError):
        # Do not print raw database errors, free-text values, paths or user identifiers.
        print("Não foi possível conferir o relatório: verifique banco existente, esquema, "
              "métricas consistentes e uma única comparação concluída compatível.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
