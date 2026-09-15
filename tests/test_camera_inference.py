import numpy as np
import pytest

from gesturelab.camera import CameraWorker, assess_quality
from gesturelab.inference import ActionGate, TemporalStabilizer


def test_quality_rejects_invalid_and_small():
    assert not assess_quality(np.zeros((21, 3)))[0]
    assert not assess_quality(np.full((21, 3), np.nan))[0]
    assert not assess_quality(np.zeros((20, 3)))[0]
    points = np.full((21, 3), 0.5)
    points[9, 1] = 0.7
    assert assess_quality(points)[0]
    points[8, 0] = 1.1
    assert not assess_quality(points)[0]


def test_action_requires_neutral_and_interval():
    gate = ActionGate(cooldown=2)
    assert gate.update("V", now=0) == "V"
    assert gate.update("V", now=5) is None
    gate.update(None, now=5)
    assert gate.update("V", now=6) == "V"
    gate.update(None, now=6.1)
    assert gate.update("V", now=7) is None
    assert gate.update("V", now=8) == "V"


def test_stabilization_does_not_keep_hand_when_absent():
    smoother = TemporalStabilizer(window=3, consensus=1)
    assert smoother.update("A", now=0) is None
    assert smoother.update("B", now=0.1) is None
    assert smoother.update("A", now=0.2) is None
    assert smoother.update("A", now=0.3) is None
    assert smoother.update("A", now=0.4) == "A"
    assert smoother.response_delay_ms == pytest.approx(200)
    assert smoother.update(None, now=0.5) is None
    assert smoother.update("A", now=0.6) is None


def test_missing_model_worker_exits_cleanly(qtbot, tmp_path):
    worker = CameraWorker(tmp_path / "absent.task")
    messages = []
    worker.error.connect(messages.append)
    with qtbot.waitSignal(worker.finished, timeout=10000):
        worker.start()
    assert messages and "Detector ausente" in messages[0]
    assert not worker.isRunning()
