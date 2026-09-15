"""Contratos integrados com dados sintéticos: não são resultados de reconhecimento."""
import json
import threading

import numpy as np
import pytest

from gesturelab import ml
from gesturelab.improvement import equal_budget
from gesturelab.store import Store


def test_database_handles_closed_after_each_operation(tmp_path):
    store = Store(tmp_path)
    for _ in range(10):
        store.categories()
        store.active()
    # Windows não permite apagar o banco com handles de conexões ainda abertos.
    store.db_path.unlink()
    assert not store.db_path.exists()


def build_dataset(store):
    labels = ["palma", "punho"]
    for label in labels:
        store.add_category(label)
    rng = np.random.default_rng(12)
    for group in range(4):
        session = store.create_session(f"p{group}")
        for kind, label in enumerate(labels):
            for example in range(8):
                points = np.full((21, 3), 0.3) + rng.normal(0, 0.012, (21, 3))
                points[:, 1] = np.linspace(0.3, 0.7, 21)
                points[9] = (0.3, 0.6, 0)
                points[4, 0] += kind * 0.25
                store.add_sample(session, label, points, quality={"source": "difficult_review" if example >= 5 else "capture"})
    return store.snapshot()


def test_all_models_reload_and_match_batch_pipeline(tmp_path):
    store = Store(tmp_path)
    dataset = build_dataset(store)
    ids = ml.train(store, dataset, models=["dummy", "logistic", "forest", "mlp"])
    assert len(ids) == 4, [(e["model"], e["error"]) for e in store.experiments()]
    frame, _ = ml.split_dataset(store, dataset, "session", 42)
    for ident in ids:
        bundle = ml.load_model(store, ident)
        sample = frame.iloc[0]
        result = ml.predict(bundle, sample.landmarks, sample.handedness)
        from gesturelab.features import training_matrix
        matrix = training_matrix([sample.landmarks], [sample.handedness], bundle["strategy"])
        assert result["label"] == bundle["pipeline"].predict(matrix)[0]
        assert ml.compare(store, [ids[0], ident])["compatible"]
        exp = store.experiment(ident)
        assert exp["metrics"]["final"] is None
        assert exp["metrics"]["validation"]["coverage"] >= 0.6
    # Corrupção nunca pode chegar ao desserializador.
    file = store.artifacts_dir / store.experiment(ids[0])["artifact_path"]
    file.write_bytes(b"invalid-model")
    with pytest.raises(ValueError, match="Hash"):
        ml.load_model(store, ids[0])


def test_cancel_during_last_fit_never_publishes(tmp_path, monkeypatch):
    store = Store(tmp_path)
    dataset = build_dataset(store)
    cancelled = threading.Event()
    original = ml._pipeline

    def pipeline(*args, **kwargs):
        model = original(*args, **kwargs)
        fit = model.fit

        def cancel_after_fit(*args, **kwargs):
            result = fit(*args, **kwargs)
            cancelled.set()
            return result

        model.fit = cancel_after_fit
        return model

    monkeypatch.setattr(ml, "_pipeline", pipeline)
    assert ml.train(store, dataset, models=["dummy"], cancel=cancelled) == []
    assert store.experiments()[0]["status"] == "cancelled"
    assert not list(store.artifacts_dir.glob("*.joblib"))


def test_equal_budget_uses_frozen_validation_and_reviewed_pool(tmp_path):
    store = Store(tmp_path)
    dataset = build_dataset(store)
    frame, parts = ml.split_dataset(store, dataset, "session", 42)
    train = frame.loc[parts == "train"]
    pools = {"base": [], "random": [], "difficult": []}
    for _, grouped in train.groupby("label"):
        ordinary = [row.id for row in grouped.itertuples() if json.loads(row.quality).get("source") == "capture"]
        difficult = [row.id for row in grouped.itertuples() if json.loads(row.quality).get("source") == "difficult_review"]
        pools["base"].extend(ordinary[:2])
        pools["random"].extend(ordinary[2:])
        pools["difficult"].extend(difficult)
    report = equal_budget(store, dataset, pools["base"], pools["random"], pools["difficult"], 2, seeds=[11, 42])
    assert report["final_test_consulted"] is False
    assert len(report["runs"]) == 6
    assert {row["train_n"] for row in report["runs"] if row["arm"] != "base"} == {len(pools["base"]) + 2}
    assert set(report["validation_ids"]) == set(frame.loc[parts == "validation", "id"])
    with pytest.raises(ValueError, match="apenas IDs de treino"):
        equal_budget(store, dataset, pools["base"], frame.loc[parts == "test", "id"].tolist(), pools["difficult"], 1)
