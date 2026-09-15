"""Experimento controlado opcional: orçamento igual de aleatórios e difíceis.

Não escolhe parâmetros nem consulta teste/desafio. A seleção explícita usa IDs
de um snapshot, mantendo as mesmas linhas de validação para todos os braços.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

from .features import training_matrix
from .ml import _choose_threshold, _env, _metrics, _pipeline, _probabilities, _spec, split_dataset


def equal_budget(store, dataset_id: str, base_ids: list[str], random_ids: list[str],
                 difficult_ids: list[str], budget: int, seeds=(11, 42, 73),
                 strategy="normalized", model="logistic", protocol="session", cancel=None) -> dict:
    if budget < 1 or not seeds or not base_ids:
        raise ValueError("Informe base de treino, orçamento positivo e ao menos uma semente.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("As sementes devem ser distintas.")
    pools = [list(base_ids), list(random_ids), list(difficult_ids)]
    if any(len(set(pool)) != len(pool) for pool in pools):
        raise ValueError("Há IDs repetidos nos conjuntos.")
    if set(base_ids) & (set(random_ids) | set(difficult_ids)):
        raise ValueError("Exemplos adicionais não podem pertencer à base inicial.")
    if min(len(random_ids), len(difficult_ids)) < budget:
        raise ValueError("Ambos os conjuntos adicionais precisam atender ao mesmo orçamento.")
    frame, parts = split_dataset(store, dataset_id, protocol, 42)
    train_ids = set(frame.loc[parts == "train", "id"])
    if not set().union(*map(set, pools)) <= train_ids:
        raise ValueError("Use apenas IDs de treino do snapshot; avaliação e teste não podem orientar adições.")
    index = frame.set_index("id", drop=False)
    classes = set(frame["label"])
    if set(index.loc[base_ids, "label"]) != classes:
        raise ValueError("A base inicial precisa conter todas as classes.")
    for ident in difficult_ids:
        quality = index.loc[ident, "quality"]
        if isinstance(quality, str):
            quality = json.loads(quality)
        if quality.get("source") != "difficult_review":
            raise ValueError("O conjunto difícil deve conter somente exemplos revisados por uma pessoa.")
    validation = frame.loc[parts == "validation"]
    Xval = training_matrix(validation["landmarks"], validation["handedness"], strategy)
    yval = validation["label"].astype(str).to_numpy()
    _, grid = _spec(model, 42)
    parameters = grid[0]  # configuração fixa, sem escolher o braço pela avaliação
    report = {"id": "improvement_" + uuid.uuid4().hex, "dataset": store.snapshot_info(dataset_id),
        "protocol": protocol, "strategy": strategy, "model": model, "parameters": parameters,
        "environment": _env(), "budget": budget, "seeds": list(seeds), "base_ids": base_ids,
        "validation_ids": validation["id"].tolist(), "final_test_consulted": False,
        "note": "Hipótese de desenvolvimento; não demonstra melhoria real sem coleta independente.", "runs": []}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        additions = {"base": [], "random": rng.choice(random_ids, budget, replace=False).tolist(),
                     "difficult": rng.choice(difficult_ids, budget, replace=False).tolist()}
        for arm, extra in additions.items():
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Experimento cancelado; relatório incompleto não publicado.")
            selected = index.loc[base_ids + extra]
            X = training_matrix(selected["landmarks"], selected["handedness"], strategy)
            pipeline = _pipeline(model, strategy, int(seed), parameters)
            started = time.perf_counter()
            pipeline.fit(X, selected["label"].astype(str))
            duration = time.perf_counter() - started
            model_classes, probabilities = _probabilities(pipeline, Xval)
            threshold, _ = _choose_threshold(model_classes, probabilities, yval)
            report["runs"].append({"seed": int(seed), "arm": arm, "added_ids": extra,
                "train_n": len(selected), "duration_s": duration, "threshold": threshold,
                "metrics": _metrics(pipeline, Xval, yval, threshold)})
    if cancel is not None and cancel.is_set():
        raise InterruptedError("Experimento cancelado; relatório incompleto não publicado.")
    output = store.root / "improvement-reports"
    output.mkdir(exist_ok=True)
    target = output / (report["id"] + ".json")
    temporary = target.with_suffix(".writing")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return {**report, "report_path": str(target)}


def main():
    import argparse
    from .store import Store
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("definition", type=Path, help="JSON com dataset_id, base_ids, random_ids, difficult_ids, budget e seeds")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    try:
        specification = json.loads(args.definition.read_text(encoding="utf-8"))
        result = equal_budget(Store(args.data_dir), **specification)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Não foi possível executar o experimento: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
