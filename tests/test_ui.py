"""Testes de fumaça da interface sem abrir webcam ou treinar modelos."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from gesturelab.ui import MainWindow


class EmptyStore:
    def categories(self):
        return []

    def samples(self, session_id=None):
        import pandas as pd

        return pd.DataFrame(columns=["id", "session_id", "participant", "label", "handedness", "quality", "landmarks"])

    def sessions(self):
        return []

    def experiments(self):
        return []

    def active(self):
        return None

    def snapshots(self):
        return []

    def difficult(self):
        return []


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_main_window_starts_with_empty_store_and_all_pages(app):
    window = MainWindow(EmptyStore())
    assert window.stack.count() == 8
    assert window.windowTitle().startswith("GestureLab")
    assert window.home_samples.value.text() == "0"
    for index in range(8):
        window.show_page(index)
        assert window.stack.currentIndex() == index
    window.close()


def test_window_does_not_start_camera_on_construction(app):
    window = MainWindow(EmptyStore())
    assert window.camera_worker is None
    assert "Câmera" in window.collect_preview.text()
    window.close()


def test_no_hand_clears_prediction_and_rearms_action(app):
    from gesturelab.inference import ActionGate, TemporalStabilizer
    window = MainWindow(EmptyStore())
    window.temporal_stabilizer = TemporalStabilizer(window=1)
    window.action_gate = ActionGate(cooldown=0)
    window.temporal_stabilizer.update("V")
    window.action_gate.update("V")
    window.last_prediction = {"label": "V", "score": 0.9}
    window.handle_live_prediction({"state": "no_hand", "landmarks": None})
    assert window.live_prediction.text() == "Sem mão"
    assert window.last_prediction is None
    assert window.temporal_stabilizer.stable is None
    assert window.action_gate.armed
    window.close()


def test_stop_keeps_worker_alive_until_finished(app):
    class Worker:
        running = True
        stopped = False
        deleted = False

        def stop(self):
            self.stopped = True

        def isRunning(self):
            return self.running

        def deleteLater(self):
            self.deleted = True

    window = MainWindow(EmptyStore())
    worker = Worker()
    window.camera_worker = worker
    window.stop_camera()
    assert worker.stopped and window.camera_worker is worker
    assert not worker.deleted
    worker.running = False
    window._camera_finished()
    assert window.camera_worker is None and worker.deleted
    window.close()


def test_training_ui_with_real_store_and_reports(app, qtbot, tmp_path):
    from gesturelab.store import Store
    from test_workflow import build_dataset
    store = Store(tmp_path)
    dataset = build_dataset(store)
    window = MainWindow(store)
    qtbot.addWidget(window)
    window.show_page(3)
    window.train_dataset.setCurrentIndex(window.train_dataset.findData(dataset))
    for name, checkbox in window.train_model_checks.items():
        checkbox.setChecked(name == "logistic")
    window.train_preset.setCurrentIndex(window.train_preset.findData("single"))
    window.start_training()
    qtbot.waitUntil(lambda: window.train_worker is None, timeout=30000)
    assert store.experiments()[0]["status"] == "completed", window.train_log.toPlainText()
    window.show_page(4)
    window.experiment_table.selectRow(0)
    window.show_selected_experiment()
    assert window.experiment_report.rowCount() == 2
    window.activate_selected_experiment()
    assert store.active()
    sample = store.samples().iloc[0]
    window.handle_live_prediction({"state": "ready", "landmarks": sample.landmarks, "handedness": sample.handedness})
    assert window.last_prediction and "score" in window.last_prediction
    window.close()


def test_review_then_delete_sample_with_confirmation(app, tmp_path, monkeypatch):
    import numpy as np
    from PySide6.QtWidgets import QMessageBox
    from gesturelab.store import Store
    store = Store(tmp_path)
    session = store.create_session()
    points = np.full((21, 3), 0.4)
    points[9, 1] = 0.7
    store.add_difficult(session, points, "Right", extractor_version="test-extractor")
    window = MainWindow(store)
    window.show_page(6)
    window.review_table.selectRow(0)
    assert window.review_preview._points
    window.review_label.setCurrentIndex(window.review_label.findData("Palma aberta"))
    window.apply_difficult_review()
    assert len(store.samples()) == 1
    assert store.samples().iloc[0].extractor_version == "test-extractor"
    assert window.review_table.rowCount() == 0
    window.show_page(2)
    window.dataset_table.selectRow(0)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Cancel)
    window.delete_selected()
    assert len(store.samples()) == 1
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Discard)
    window.delete_selected()
    assert store.samples().empty
    window.close()


def test_collection_rejects_a_stale_camera_packet(app):
    window = MainWindow(EmptyStore())
    window.collect_label.addItem("Palma aberta", "Palma aberta")
    window.collecting = True
    window.collect_session_id = "session_1"
    window.latest_frame = {
        "state": "ready",
        "landmarks": [[0.0, 0.0, 0.0]] * 21,
        "timestamp_ms": int(time.monotonic() * 1000) - 2000,
    }

    window.record_current_sample()

    assert "imagem recente" in window.collect_state.text()
    assert window.collect_count == 0
    window.close()
