"""Regressões dos botões com frames chegando durante o uso da coleta."""
import time

import numpy as np
import pytest
from PySide6.QtCore import Qt

from gesturelab.store import Store
from gesturelab.ui import MainWindow


@pytest.fixture
def collection(qtbot, tmp_path):
    class Camera:
        def isRunning(self):
            return False

        def stop(self):
            pass

        def deleteLater(self):
            pass

    window = MainWindow(Store(tmp_path))
    qtbot.addWidget(window)
    window.show_page(1)
    window.camera_worker = Camera()
    window.camera_target = "collect"
    window.collect_label.setCurrentIndex(window.collect_label.findData("Palma aberta"))
    yield window
    window.pause_collection()
    window.camera_worker = None
    window.close()


def frame(window, state="ready"):
    points = np.full((21, 3), 0.4)
    points[9, 1] = 0.7
    window.handle_camera_frame({"state": state, "landmarks": points if state == "ready" else None,
        "timestamp_ms": int(time.monotonic() * 1000), "handedness": "Right", "quality": {}, "fps": 30})


def test_countdown_message_survives_camera_frames(collection, qtbot):
    qtbot.mouseClick(collection.collect_start, Qt.MouseButton.LeftButton)
    assert "3" in collection.collect_state.text()
    frame(collection)
    assert "3" in collection.collect_state.text(), "O frame apagou a contagem regressiva"
    collection._countdown_tick()
    frame(collection)
    assert "2" in collection.collect_state.text()


def test_buttons_collect_pause_resume_finish_and_preserve_samples(collection, qtbot):
    qtbot.mouseClick(collection.collect_start, Qt.MouseButton.LeftButton)
    for _ in range(3):
        collection._countdown_tick()
    frame(collection)
    collection.record_current_sample()
    assert len(collection.store.samples()) == 1
    session = collection.collect_session_id
    qtbot.mouseClick(collection.collect_pause, Qt.MouseButton.LeftButton)
    frame(collection)
    collection.record_current_sample()
    assert "pausada" in collection.collect_state.text().lower()
    assert len(collection.store.samples()) == 1
    collection.collect_label.setCurrentIndex(collection.collect_label.findData("Punho fechado"))
    qtbot.mouseClick(collection.collect_start, Qt.MouseButton.LeftButton)
    for _ in range(3):
        collection._countdown_tick()
    frame(collection)
    collection.last_saved_timestamp_ms = None
    collection.record_current_sample()
    assert collection.collect_session_id == session
    assert len(collection.store.samples()) == 2
    qtbot.mouseClick(collection.collect_finish, Qt.MouseButton.LeftButton)
    frame(collection)
    assert "finalizada" in collection.collect_state.text().lower()
    assert not collection.collecting and collection.collect_session_id is None
    assert len(collection.store.samples()) == 2


def test_missing_gesture_has_persistent_instruction(collection):
    collection.collect_label.setCurrentIndex(0)
    collection.begin_countdown()
    frame(collection)
    assert "gesto" in collection.collect_state.text().lower()
    assert not collection.store.sessions()
