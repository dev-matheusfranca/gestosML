from __future__ import annotations

import threading

import numpy as np
import pytest

from gesturelab.features import canonical_landmarks, feature_vector
from gesturelab.ml import evaluate_final, load_model, predict, split_dataset, train
from gesturelab.store import Store


def _points(kind: int, variation: float = 0.0) -> np.ndarray:
    points = np.zeros((21, 3), dtype=float)
    for index in range(1, 21):
        points[index] = (index / 25, index / 80, 0.01 * index)
    points[9] = (0.0, 1.0, 0.0)  # escala canônica previsível
    points[4, 0] = 0.15 + kind * 0.45 + variation
    points[8, 1] = 0.25 + kind * 0.35
    return points


def _dataset(store: Store) -> str:
    store.add_category("palma")
    store.add_category("punho")
    for session_number in range(4):
        session = store.create_session(f"p{session_number}")
        for kind, label in enumerate(("palma", "punho")):
            for copy in range(2):
                store.add_sample(session, label, _points(kind, 0.001 * (session_number + copy)))
    return store.snapshot()


def test_normalization_rejects_invalid_and_canonicalizes_left() -> None:
    right = _points(0)
    left = right.copy()
    left[:, 0] *= -1
    assert np.allclose(canonical_landmarks(right, "Right"), canonical_landmarks(left, "Left"))
    assert feature_vector(right, "normalized").shape == (63,)
    assert feature_vector(right, "distances").shape == (9,)
    assert feature_vector(right, "angles").shape == (15,)
    with pytest.raises(ValueError):
        canonical_landmarks(np.zeros((21, 3)))
    with pytest.raises(ValueError):
        canonical_landmarks(np.full((21, 3), np.nan))


def test_store_keeps_snapshots_immutable_and_validates_samples(tmp_path) -> None:
    store = Store(tmp_path / "data")
    dataset = _dataset(store)
    before = store.load_snapshot(dataset)
    sample_id = store.samples().iloc[0]["id"]
    store.relabel_sample(sample_id, "punho")
    after = store.load_snapshot(dataset)
    assert before.iloc[0]["label"] == after.iloc[0]["label"]
    assert store.validate()["valid"] is True
    with pytest.raises(ValueError):
        store.add_sample("missing", "palma", _points(0))


def test_group_split_has_no_overlap_and_final_reservation_is_stable(tmp_path) -> None:
    store = Store(tmp_path / "data")
    dataset = _dataset(store)
    frame, parts = split_dataset(store, dataset, "session", 42)
    groups = frame["session_id"]
    grouped = {part: set(groups[parts == part]) for part in ("train", "validation", "test")}
    assert not (grouped["train"] & grouped["validation"])
    assert not (grouped["train"] & grouped["test"])
    test_groups = grouped["test"]
    # Outro snapshot da mesma Store não permite que a nova seed libere o teste.
    session = store.create_session("extra")
    store.add_sample(session, "palma", _points(0, 0.01))
    store.add_sample(session, "punho", _points(1, 0.01))
    later = store.snapshot()
    second_frame, second_parts = split_dataset(store, later, "session", 999)
    assert set(second_frame.loc[second_parts == "test", "session_id"]) == test_groups


def test_frozen_validation_and_test_samples_cannot_be_changed(tmp_path) -> None:
    store = Store(tmp_path / "data")
    dataset = _dataset(store)
    frame, parts = split_dataset(store, dataset, "session", 42)
    frozen = frame.loc[parts == "test"].iloc[0]
    frozen_id = frozen["id"]
    store.relabel_sample(frozen_id, "punho" if frozen["label"] != "punho" else "palma")
    with pytest.raises(ValueError, match="congelada"):
        split_dataset(store, store.snapshot(), "session", 42)


def test_protocol_cannot_change_after_holdout_reservation(tmp_path) -> None:
    store = Store(tmp_path / "data")
    dataset = _dataset(store)
    split_dataset(store, dataset, "session", 42)
    with pytest.raises(ValueError, match="outro protocolo"):
        split_dataset(store, dataset, "participant", 42)


def test_train_load_predict_and_evaluate_final_once(tmp_path) -> None:
    store = Store(tmp_path / "data")
    dataset = _dataset(store)
    ids = train(store, dataset, models=["logistic"], seed=12)
    assert len(ids) == 1
    model = load_model(store, ids[0])
    result = predict(model, _points(0))
    assert {"label", "score", "accepted", "latency_ms"} <= set(result)
    first = evaluate_final(store, ids[0])
    second = evaluate_final(store, ids[0])
    assert first == second
    assert store.experiment(ids[0])["metrics"]["final"] == first


def test_cancelled_training_does_not_publish_artifact(tmp_path) -> None:
    store = Store(tmp_path / "data")
    dataset = _dataset(store)
    cancelled = threading.Event()
    cancelled.set()
    assert train(store, dataset, models=["logistic"], cancel=cancelled) == []
    experiment = store.experiments()[0]
    assert experiment["status"] == "cancelled"
    assert experiment["artifact_path"] is None


def test_final_holdout_is_claimed_once_per_lineage(tmp_path) -> None:
    store = Store(tmp_path / "data")
    dataset = _dataset(store)
    first = train(store, dataset, models=["dummy"], seed=5)[0]
    evaluate_final(store, first)
    second = train(store, dataset, models=["logistic"], seed=5)[0]
    with pytest.raises(ValueError, match="já foi consultado"):
        evaluate_final(store, second)


def test_reviewing_difficult_creates_one_human_labeled_sample(tmp_path) -> None:
    store = Store(tmp_path / "data")
    session = store.create_session("revisor")
    difficult = store.add_difficult(session, _points(0), "Right", predicted_label="punho", score=0.1)
    count = len(store.samples())
    store.review_difficult(difficult, "Palma aberta")
    assert len(store.samples()) == count + 1
    assert store.difficult()[0]["incorporated_sample_id"]
    with pytest.raises(ValueError):
        store.review_difficult(difficult, "Palma aberta")
