"""Captura local e extração de pontos; nenhum frame é gravado em disco."""

from __future__ import annotations

import hashlib
import importlib.metadata
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def default_model_path() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "hand_landmarker.task"


def assess_quality(landmarks, min_size: float = 0.06) -> tuple[bool, dict]:
    """Checa geometria, não inventa confiança individual dos 21 pontos."""
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return False, {"reason": "Pontos inválidos ou incompletos"}
    span = float(np.linalg.norm(points[9] - points[0]))
    in_frame = bool(((points[:, :2] >= 0) & (points[:, :2] <= 1)).all())
    quality = {"palm_size": span, "in_frame": in_frame, "min_size": min_size}
    if not in_frame:
        return False, {**quality, "reason": "Mantenha a mão inteira dentro da imagem"}
    if span < min_size:
        return False, {**quality, "reason": "Aproxime a mão da câmera"}
    return True, {**quality, "reason": "Pontos válidos"}


def draw_landmarks(frame: np.ndarray, hands: list[np.ndarray]) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    for hand in hands:
        if hand.shape != (21, 3) or not np.isfinite(hand).all():
            continue
        xy = np.clip(hand[:, :2] * (w, h), -10000, 10000).astype(int)
        for start, end in CONNECTIONS:
            cv2.line(canvas, tuple(xy[start]), tuple(xy[end]), (162, 211, 47), 2, cv2.LINE_AA)
        for x, y in xy:
            cv2.circle(canvas, (int(x), int(y)), 4, (246, 237, 218), -1, cv2.LINE_AA)
    return canvas


class CameraWorker(QThread):
    frame_ready = Signal(object)
    error = Signal(str)

    def __init__(self, model_path: Path, camera_index=0, width=640, height=480,
                 min_confidence=0.5, min_size=0.06):
        super().__init__()
        self.model_path = Path(model_path)
        self.camera_index = camera_index
        self.width, self.height = width, height
        self.min_confidence, self.min_size = min_confidence, min_size
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()
        self.requestInterruption()

    def run(self):
        cap, detector = None, None
        try:
            if not self.model_path.is_file():
                raise RuntimeError("Detector ausente. Execute Instalar.ps1 para baixar o Hand Landmarker.")
            # Import tardio: navegação e treinamento funcionam sem iniciar MediaPipe.
            import mediapipe as mp

            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.camera_index, backend)
            if not cap.isOpened():
                raise RuntimeError("Câmera indisponível. Confira o índice, a permissão do Windows e outros aplicativos usando a câmera.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=self.min_confidence,
                min_hand_presence_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence,
            )
            detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
            digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()[:16]
            extractor = f"mediapipe-{importlib.metadata.version('mediapipe')}-hand-v1-{digest}"
            previous, fps, last_timestamp = time.perf_counter(), 0.0, -1
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("A câmera abriu, mas não entregou uma imagem. Feche outros aplicativos e tente reabrir.")
                started = time.perf_counter()
                timestamp = max(last_timestamp + 1, int(time.monotonic() * 1000))
                last_timestamp = timestamp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                latency = (time.perf_counter() - started) * 1000
                hands = [np.array([[p.x, p.y, p.z] for p in hand], dtype=float) for hand in result.hand_landmarks]
                state, points, handedness, quality = "no_hand", None, "Right", {}
                if len(hands) > 1:
                    state = "multiple_hands"
                elif len(hands) == 1:
                    points = hands[0]
                    valid, quality = assess_quality(points, self.min_size)
                    state = "ready" if valid else "invalid"
                    if result.handedness and result.handedness[0]:
                        side = result.handedness[0][0]
                        handedness = side.category_name
                        quality["handedness_score"] = float(side.score)
                    if not valid:
                        points = None
                display = cv2.flip(draw_landmarks(frame, hands), 1)
                display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                h, w, _ = display.shape
                image = QImage(display.data, w, h, display.strides[0], QImage.Format.Format_RGB888).copy()
                now = time.perf_counter()
                current_fps = 1.0 / max(now - previous, 1e-9)
                fps = current_fps if not fps else 0.9 * fps + 0.1 * current_fps
                previous = now
                if not self._stop_event.is_set():
                    self.frame_ready.emit({"image": image, "landmarks": points, "handedness": handedness,
                        "quality": quality, "state": state, "fps": fps, "latency_ms": latency,
                        "extractor_version": extractor, "timestamp_ms": timestamp, "resolution": [w, h]})
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            try:
                if detector is not None:
                    detector.close()
            finally:
                if cap is not None:
                    cap.release()


# Conveniência para consumidores do contrato inicial.
from gesturelab.inference import ActionGate, TemporalStabilizer  # noqa: E402,F401
