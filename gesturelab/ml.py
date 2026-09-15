"""Treinamento reproduzível e inferência local do GestureLab."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_VERSION, LandmarkFeatures, as_landmarks, canonical_landmarks, training_matrix
from .store import Store


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _env() -> dict[str, str]:
    def installed(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "unavailable"
    return {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "scipy": installed("scipy"), "joblib": installed("joblib"), "mediapipe": installed("mediapipe"), "feature_version": FEATURE_VERSION}


def _group_column(frame: pd.DataFrame, protocol: str) -> pd.Series:
    if protocol == "session":
        return frame["session_id"].astype(str)
    if protocol == "participant":
        participants = frame["participant"].fillna("").astype(str).str.strip()
        if (participants == "").any():
            raise ValueError("O protocolo por participante exige participante pseudônimo em todas as amostras.")
        return participants
    raise ValueError("protocol deve ser session ou participant.")


def _has_all_classes(frame: pd.DataFrame, partitions: pd.Series, labels: set[str]) -> bool:
    return all(set(frame.loc[partitions == part, "label"]) == labels for part in ("train", "validation", "test"))


def _sample_fingerprint(row: pd.Series) -> str:
    payload = {"id": str(row["id"]), "session_id": str(row["session_id"]), "participant": str(row["participant"]), "purpose": str(row["purpose"]), "label": str(row["label"]), "handedness": str(row["handedness"]), "extractor_version": str(row["extractor_version"]), "landmarks": np.asarray(row["landmarks"], dtype=float).reshape(-1).tolist()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _new_group_assignments(frame: pd.DataFrame, groups: pd.Series, seed: int) -> dict[str, str]:
    """Encontra split estrito por grupo; nunca substitui por divisão por frames."""
    unique = groups.unique()
    labels = set(frame["label"].astype(str))
    if len(unique) < 4:
        raise ValueError("São necessários pelo menos 4 grupos independentes para treino, validação e teste final.")
    for attempt in range(96):
        random = seed + attempt
        outer = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random)
        dev_idx, test_idx = next(outer.split(frame, frame["label"], groups))
        dev = frame.iloc[dev_idx]
        dev_groups = groups.iloc[dev_idx]
        if dev_groups.nunique() < 2:
            continue
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random + 1009)
        train_local, validation_local = next(inner.split(dev, dev["label"], dev_groups))
        parts = pd.Series("", index=frame.index, dtype="object")
        parts.iloc[test_idx] = "test"
        parts.iloc[dev_idx[train_local]] = "train"
        parts.iloc[dev_idx[validation_local]] = "validation"
        if _has_all_classes(frame, parts, labels):
            return {str(group): str(parts.loc[groups == group].iloc[0]) for group in unique}
    present = ", ".join(sorted(labels))
    raise ValueError(f"Não foi possível reservar grupos com todas as classes ({present}) em treino, validação e teste. Colete cada gesto em mais sessões independentes.")


def split_dataset(store: Store, dataset_id: str, protocol: str, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Retorna snapshot e partições persistentes, mantendo o teste reservado da linhagem."""
    frame = store.load_snapshot(dataset_id)
    # Sessões challenge medem gestos/condições fora do desenvolvimento e não
    # participam de escolha de modelo, parâmetros ou limiar.
    if "purpose" in frame:
        frame = frame.loc[frame["purpose"].astype(str) == "development"].reset_index(drop=True)
    if frame.empty or frame["label"].nunique() < 2:
        raise ValueError("O treinamento exige ao menos duas categorias com exemplos.")
    info = store.snapshot_info(dataset_id)
    groups = _group_column(frame, protocol)
    locked_protocol = store.split_protocol(info["lineage_id"])
    if locked_protocol is not None and locked_protocol != protocol:
        raise ValueError("Esta linhagem já reservou o teste final com outro protocolo; crie uma nova Store/linhagem para trocar o protocolo.")
    saved = store.split_rows(info["lineage_id"], protocol)
    saved_samples = store.split_sample_rows(info["lineage_id"], protocol)
    missing = [str(group) for group in groups.unique() if str(group) not in saved]
    if not saved:
        store.save_split_rows(info["lineage_id"], protocol, _new_group_assignments(frame, groups, seed))
        saved = store.split_rows(info["lineage_id"], protocol)
    elif missing:
        # A reserva do teste já criada nunca é movida. Novos grupos entram no
        # desenvolvimento; a validação original também permanece fixa.
        store.save_split_rows(info["lineage_id"], protocol, {group: "train" for group in missing})
        saved = store.split_rows(info["lineage_id"], protocol)
    store.lock_split_protocol(info["lineage_id"], protocol)
    current = {str(row["id"]): _sample_fingerprint(row) for _, row in frame.iterrows()}
    # Validação e teste são congelados por ID e conteúdo. Remover, relabel ou
    # acrescentar frames em seus grupos exigiria uma nova linhagem, pois mudaria
    # a evidência usada para escolher/medir o modelo.
    for sample_id, (partition, fingerprint) in saved_samples.items():
        if partition in {"validation", "test"} and (sample_id not in current or current[sample_id] != fingerprint):
            raise ValueError("O snapshot altera ou remove uma amostra congelada de validação/teste. Crie uma nova Store/linhagem para preservar o histórico metodológico.")
    parts_by_id: dict[str, str] = {}
    new_members: list[tuple[str, str, str]] = []
    for index, row in frame.iterrows():
        sample_id = str(row["id"])
        fingerprint = current[sample_id]
        group = str(groups.iloc[index])
        if sample_id in saved_samples:
            partition = saved_samples[sample_id][0]
        else:
            partition = saved[group]
            if saved_samples and partition in {"validation", "test"}:
                raise ValueError("O snapshot adiciona amostras a um grupo congelado de validação/teste. Colete a nova variação em uma sessão independente.")
            new_members.append((sample_id, partition, fingerprint))
        parts_by_id[sample_id] = partition
    if new_members:
        store.save_split_samples(info["lineage_id"], protocol, new_members)
    partitions = frame["id"].astype(str).map(parts_by_id)
    labels = set(frame["label"].astype(str))
    if not _has_all_classes(frame, partitions, labels):
        raise ValueError("O snapshot atual não contém todas as classes em cada partição persistida. Crie uma coleta independente por gesto antes de avaliar.")
    group_sets = {part: set(groups[partitions == part]) for part in ("train", "validation", "test")}
    if group_sets["train"] & group_sets["validation"] or group_sets["train"] & group_sets["test"] or group_sets["validation"] & group_sets["test"]:
        raise RuntimeError("Violação interna: grupos sobrepostos entre partições.")
    return frame, partitions


def _group_cv_score(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, groups: pd.Series, seed: int) -> dict | None:
    """Relata CV por grupos no desenvolvimento sem alterar a seleção por validação."""
    n_groups = int(groups.nunique())
    if n_groups < 3:
        return None
    folds = min(3, n_groups)
    results: list[float] = []
    try:
        for train_index, test_index in GroupKFold(n_splits=folds).split(X, y, groups):
            # Cada fold precisa conter as mesmas classes no treino e na dobra.
            if set(y[train_index]) != set(y) or set(y[test_index]) != set(y):
                return None
            from sklearn.base import clone
            candidate = clone(pipeline)
            candidate.fit(X[train_index], y[train_index])
            results.append(float(f1_score(y[test_index], candidate.predict(X[test_index]), average="macro", zero_division=0)))
    except ValueError:
        return None
    return {"folds": folds, "macro_f1_mean": float(np.mean(results)), "macro_f1_std": float(np.std(results)), "scores": results}


def _spec(name: str, seed: int) -> tuple[object, list[dict]]:
    if name == "dummy":
        return DummyClassifier(strategy="prior"), [{}]
    if name == "logistic":
        return LogisticRegression(max_iter=800, random_state=seed), [{"classifier__C": value} for value in (0.3, 1.0, 3.0)]
    if name == "forest":
        return RandomForestClassifier(random_state=seed, n_jobs=1, class_weight="balanced"), [{"classifier__n_estimators": 120, "classifier__max_depth": depth, "classifier__min_samples_leaf": leaf} for depth in (None, 10) for leaf in (1, 3)]
    if name == "mlp":
        # early_stopping criaria uma divisão interna por frames. A validação
        # externa já é agrupada e é a única usada neste protocolo.
        return MLPClassifier(max_iter=350, random_state=seed, early_stopping=False), [{"classifier__hidden_layer_sizes": hidden, "classifier__alpha": alpha} for hidden in ((24,), (48,)) for alpha in (0.0001, 0.001)]
    raise ValueError(f"Modelo desconhecido: {name}")


def _pipeline(name: str, strategy: str, seed: int, parameters: dict) -> Pipeline:
    classifier, _ = _spec(name, seed)
    pipeline = Pipeline([("features", LandmarkFeatures(strategy)), ("scaler", StandardScaler()), ("classifier", classifier)])
    pipeline.set_params(**parameters)
    return pipeline


def _probabilities(pipeline: Pipeline, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classifier = pipeline.named_steps["classifier"]
    classes = np.asarray(classifier.classes_, dtype=str)
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)
    else:  # não usado pelos modelos atuais, mas mantém contrato explícito
        predicted = pipeline.predict(X)
        probabilities = np.zeros((len(predicted), len(classes)))
        for index, label in enumerate(predicted):
            probabilities[index, np.where(classes == str(label))[0][0]] = 1.0
    return classes, np.asarray(probabilities, dtype=float)


def _choose_threshold(classes: np.ndarray, probabilities: np.ndarray, y: Iterable[str]) -> tuple[float, dict]:
    truth = np.asarray(list(y), dtype=str)
    best: tuple[float, float, dict] | None = None
    for threshold in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8):
        scores = probabilities.max(axis=1)
        accepted = scores >= threshold
        coverage = float(accepted.mean())
        if coverage < 0.60:
            continue
        predicted = classes[probabilities.argmax(axis=1)]
        accepted_f1 = float(f1_score(truth[accepted], predicted[accepted], labels=classes, average="macro", zero_division=0)) if accepted.any() else 0.0
        candidate = {"coverage": coverage, "accepted_macro_f1": accepted_f1, "accepted_accuracy": float(accuracy_score(truth[accepted], predicted[accepted])) if accepted.any() else 0.0}
        key = (accepted_f1, coverage)
        if best is None or key > (best[1], best[2]["coverage"]):
            best = (threshold, accepted_f1, candidate)
    if best is None:
        return 0.0, {"coverage": 1.0, "accepted_macro_f1": 0.0, "accepted_accuracy": 0.0, "warning": "Validação sem cobertura mínima para escolher limiar."}
    return best[0], best[2]


def _metrics(pipeline: Pipeline, X: np.ndarray, y: Iterable[str], threshold: float) -> dict:
    truth = np.asarray(list(y), dtype=str)
    classes, probabilities = _probabilities(pipeline, X)
    predicted = classes[probabilities.argmax(axis=1)]
    scores = probabilities.max(axis=1)
    accepted = scores >= threshold
    report = classification_report(truth, predicted, labels=classes, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=classes, average="macro", zero_division=0)),
        "coverage": float(accepted.mean()),
        "accepted_accuracy": float(accuracy_score(truth[accepted], predicted[accepted])) if accepted.any() else None,
        "accepted_macro_f1": float(f1_score(truth[accepted], predicted[accepted], labels=classes, average="macro", zero_division=0)) if accepted.any() else None,
        "per_class": report,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=classes).tolist(),
        "classes": classes.tolist(),
        "n": int(len(truth)),
    }


def _challenge_metrics(pipeline: Pipeline, frame: pd.DataFrame, strategy: str, threshold: float) -> dict | None:
    if frame.empty:
        return None
    X = training_matrix(frame["landmarks"], frame["handedness"], strategy)
    classes, probabilities = _probabilities(pipeline, X)
    scores = probabilities.max(axis=1)
    result: dict = {"n": int(len(frame)), "coverage": float((scores >= threshold).mean()), "mean_score": float(scores.mean()), "note": "Conjunto challenge não participou de seleção de modelo nem de limiar."}
    labels = set(frame["label"].astype(str))
    if labels.issubset(set(classes)):
        result["known_label_metrics"] = _metrics(pipeline, X, frame["label"].astype(str), threshold)
    else:
        result["note"] += " Há rótulos fora das classes treinadas; cobertura não prova detecção confiável de gestos desconhecidos."
    return result


def _latency_ms(pipeline: Pipeline, X: np.ndarray) -> float:
    probe = X[:1]
    started = time.perf_counter()
    for _ in range(20):
        pipeline.predict_proba(probe)
    return (time.perf_counter() - started) * 1000 / 20


def _artifact_atomic(store: Store, experiment_id: str, bundle: dict) -> tuple[str, str]:
    target = store.artifacts_dir / f"{experiment_id}.joblib"
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=".writing-", suffix=".joblib", dir=store.artifacts_dir)
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(bundle, temporary)
        digest = _sha256(temporary)
        os.replace(temporary, target)
        return target.name, digest
    finally:
        temporary.unlink(missing_ok=True)


def _cancelled(cancel: object | None) -> bool:
    return bool(cancel is not None and getattr(cancel, "is_set", lambda: False)())


def train(store: Store, dataset_id: str, strategy: str = "normalized", models: list[str] | None = None, seed: int = 42, protocol: str = "session", preset: str = "quick", cancel: object | None = None, progress: Callable[[str], None] | None = None) -> list[str]:
    """Treina somente com treino/validação. O teste final fica reservado para evaluate_final."""
    if preset not in {"quick", "single"}:
        raise ValueError("preset deve ser quick ou single.")
    names = models or ["dummy", "logistic", "forest", "mlp"]
    unknown = set(names) - {"dummy", "logistic", "forest", "mlp"}
    if unknown:
        raise ValueError(f"Modelos inválidos: {', '.join(sorted(unknown))}")
    frame, partitions = split_dataset(store, dataset_id, protocol, seed)
    snapshot_frame = store.load_snapshot(dataset_id)
    challenge_frame = snapshot_frame.loc[snapshot_frame["purpose"].astype(str) == "challenge"].reset_index(drop=True) if "purpose" in snapshot_frame else pd.DataFrame()
    info = store.snapshot_info(dataset_id)
    X = training_matrix(frame["landmarks"], frame["handedness"], strategy)
    y = frame["label"].astype(str).to_numpy()
    train_rows, validation_rows = (partitions == "train").to_numpy(), (partitions == "validation").to_numpy()
    completed: list[str] = []
    for number, name in enumerate(names, start=1):
        experiment = store.create_experiment(model=name, dataset_id=dataset_id, lineage_id=info["lineage_id"], strategy=strategy, protocol=protocol, seed=seed, parameters={}, environment=_env())
        started = time.perf_counter()
        try:
            if _cancelled(cancel):
                store.finish_experiment(experiment, "cancelled", duration_s=time.perf_counter() - started, error="Cancelado antes do ajuste.")
                continue
            _, grid = _spec(name, seed)
            if preset == "single":
                grid = grid[:1]
            best_pipeline: Pipeline | None = None
            best_parameters: dict = {}
            best_score = -1.0
            fit_warnings = []
            for variant, parameters in enumerate(grid, start=1):
                if _cancelled(cancel):
                    store.finish_experiment(experiment, "cancelled", duration_s=time.perf_counter() - started, error="Cancelado entre ajustes; nenhum artefato foi publicado.")
                    break
                if progress:
                    progress(f"Treinando {name} ({number}/{len(names)}), configuração {variant}/{len(grid)}")
                pipeline = _pipeline(name, strategy, seed, parameters)
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    pipeline.fit(X[train_rows], y[train_rows])
                for warning in captured:
                    message = f"{warning.category.__name__}: {warning.message}"
                    fit_warnings.append({"configuration": variant, "message": message})
                    if progress:
                        progress(f"Aviso em {name}: {message}")
                if _cancelled(cancel):
                    store.finish_experiment(experiment, "cancelled", duration_s=time.perf_counter() - started, error="Cancelado após ajuste; artefato não foi publicado.")
                    break
                predicted = pipeline.predict(X[validation_rows])
                score = float(f1_score(y[validation_rows], predicted, average="macro", zero_division=0))
                if score > best_score:
                    best_pipeline, best_parameters, best_score = pipeline, parameters, score
            else:
                assert best_pipeline is not None
                classes, probabilities = _probabilities(best_pipeline, X[validation_rows])
                threshold, threshold_metrics = _choose_threshold(classes, probabilities, y[validation_rows])
                if _cancelled(cancel):
                    store.finish_experiment(experiment, "cancelled", duration_s=time.perf_counter() - started, error="Cancelado antes da publicação; artefato não foi publicado.")
                    continue
                validation_metrics = _metrics(best_pipeline, X[validation_rows], y[validation_rows], threshold)
                cv = _group_cv_score(best_pipeline, X[train_rows], y[train_rows], _group_column(frame.loc[train_rows], protocol).reset_index(drop=True), seed)
                challenge = _challenge_metrics(best_pipeline, challenge_frame, strategy, threshold)
                # Os campos no topo são os resultados de validação, nunca uma
                # consulta antecipada ao teste final reservado.
                metrics = {"accuracy": validation_metrics["accuracy"], "macro_f1": validation_metrics["macro_f1"], "validation": validation_metrics, "threshold_selection": threshold_metrics, "group_cv_train": cv, "challenge": challenge, "latency_ms": _latency_ms(best_pipeline, X[validation_rows]), "final": None, "split": {"train": int(train_rows.sum()), "validation": int(validation_rows.sum()), "test_reserved": int((partitions == "test").sum())}}
                metrics["warnings"] = fit_warnings
                if _cancelled(cancel):
                    store.finish_experiment(experiment, "cancelled", duration_s=time.perf_counter() - started, error="Cancelado antes da publicação; artefato não foi publicado.")
                    continue
                bundle = {"pipeline": best_pipeline, "threshold": threshold, "classes": classes.tolist(), "strategy": strategy, "feature_version": FEATURE_VERSION, "dataset_id": dataset_id, "lineage_id": info["lineage_id"], "environment": _env()}
                artifact_path, artifact_hash = _artifact_atomic(store, experiment, bundle)
                complete_parameters = {"grid": best_parameters, "classifier": best_pipeline.named_steps["classifier"].get_params(deep=True), "strategy": strategy, "preset": preset}
                store.finish_experiment(experiment, "completed", parameters=complete_parameters, metrics=metrics, threshold=threshold, duration_s=time.perf_counter() - started, artifact_path=artifact_path, artifact_sha256=artifact_hash)
                completed.append(experiment)
        except Exception as error:
            store.finish_experiment(experiment, "failed", duration_s=time.perf_counter() - started, error=str(error))
            if progress:
                progress(f"Falha em {name}: {error}")
    return completed


def load_model(store: Store, experiment_id: str) -> dict:
    experiment = store.experiment(experiment_id)
    if experiment["status"] != "completed" or not experiment["artifact_path"]:
        raise ValueError("O experimento não possui um modelo concluído para carregar.")
    path = (store.artifacts_dir / experiment["artifact_path"]).resolve()
    if path.parent != store.artifacts_dir.resolve() or not path.is_file():
        raise ValueError("Caminho de artefato local inválido.")
    if _sha256(path) != experiment["artifact_sha256"]:
        raise ValueError("Hash do artefato não confere; o arquivo não será carregado.")
    expected = experiment["environment"]
    if expected.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Versão dos atributos incompatível; artefato não será desserializado.")
    current = _env()
    if any(expected.get(key) != current.get(key) for key in ("sklearn", "scipy", "joblib")) or expected.get("python", "").split(".")[:2] != platform.python_version().split(".")[:2]:
        raise ValueError("Versões de Python ou scikit-learn incompatíveis com o artefato local confiável.")
    # joblib/pickle só é aceito depois de validar caminho interno e hash gravado
    # pelo próprio Store; arquivos arbitrários não fazem parte deste contrato.
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Artefato incompatível com a versão atual dos atributos.")
    return bundle


def predict(bundle: dict, landmarks: object, handedness: str = "Right") -> dict:
    started = time.perf_counter()
    strategy = str(bundle.get("strategy", "normalized"))
    values = as_landmarks(landmarks) if strategy == "raw" else canonical_landmarks(landmarks, handedness)
    pipeline = bundle["pipeline"]
    classes, probabilities = _probabilities(pipeline, values.reshape(1, -1))
    index = int(probabilities[0].argmax())
    score = float(probabilities[0, index])
    return {"label": str(classes[index]), "score": score, "accepted": score >= float(bundle["threshold"]), "latency_ms": (time.perf_counter() - started) * 1000}


def evaluate_final(store: Store, experiment_id: str) -> dict:
    experiment = store.experiment(experiment_id)
    if experiment["status"] != "completed":
        raise ValueError("Somente experimento concluído pode ser avaliado.")
    metrics = experiment["metrics"]
    if metrics.get("final") is not None:
        return metrics["final"]
    stored = store.claim_final_evaluation(experiment["lineage_id"], experiment_id)
    if stored is not None:
        # Recupera uma interrupção rara entre gravar a reserva final e atualizar
        # o histórico do experimento, sem consultar o teste novamente.
        metrics["final"] = stored
        store.finish_experiment(experiment_id, "completed", parameters=experiment["parameters"], metrics=metrics, threshold=experiment["threshold"], duration_s=experiment["duration_s"], artifact_path=experiment["artifact_path"], artifact_sha256=experiment["artifact_sha256"])
        return stored
    try:
        frame, partitions = split_dataset(store, experiment["dataset_id"], experiment["protocol"], int(experiment["seed"]))
        bundle = load_model(store, experiment_id)
        X = training_matrix(frame["landmarks"], frame["handedness"], experiment["strategy"])
        mask = (partitions == "test").to_numpy()
        final = _metrics(bundle["pipeline"], X[mask], frame.loc[mask, "label"].astype(str), float(bundle["threshold"]))
        metrics["final"] = final
        store.complete_final_evaluation(experiment["lineage_id"], experiment_id, final)
        store.finish_experiment(experiment_id, "completed", parameters=experiment["parameters"], metrics=metrics, threshold=experiment["threshold"], duration_s=experiment["duration_s"], artifact_path=experiment["artifact_path"], artifact_sha256=experiment["artifact_sha256"])
        return final
    except Exception:
        store.release_final_evaluation(experiment["lineage_id"], experiment_id)
        raise


def compare(store: Store, ids: Iterable[str]) -> dict:
    experiments = [store.experiment(ident) for ident in ids]
    if len(experiments) < 2:
        raise ValueError("Selecione ao menos dois experimentos para comparar.")
    required = ("protocol", "lineage_id")
    incompatible = {field: sorted({str(item[field]) for item in experiments}) for field in required if len({str(item[field]) for item in experiments}) != 1}
    variables = {field: sorted({str(item[field]) for item in experiments}) for field in ("dataset_id", "strategy", "model", "seed") if len({str(item[field]) for item in experiments}) != 1}
    return {"compatible": not incompatible, "differences": incompatible, "controlled_variables": variables, "experiments": experiments, "message": "Comparação metodologicamente compatível: mesma linhagem, protocolo e validação/teste congelados." if not incompatible else "Não compare métricas como equivalentes: linhagem ou protocolo diferem."}
