"""Diagnóstico reproduzível sem armazenar imagens; câmera exige --camera."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import platform
from pathlib import Path
import statistics
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def probe(camera=None, frames=60):
    report = {
        "platform": platform.platform(), "python": platform.python_version(),
        "dependencies": {name: version(name) for name in (
            "mediapipe", "opencv-contrib-python", "numpy", "pandas", "scikit-learn", "PySide6", "pyarrow")},
        "model": "hand_landmarker/float16/1", "images_saved": False,
        "method": "CPU; 5 aquecimentos; imagem preta 640x480, sem mãos; perf_counter; não mede reconhecimento real",
    }
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "assets" / "hand_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2)
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        black = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((480, 640, 3), np.uint8))
        timings = []
        for index in range(frames + 5):
            start = time.perf_counter()
            result = detector.detect_for_video(black, index * 100)
            if index >= 5:
                timings.append((time.perf_counter() - start) * 1000)
        report["detector_smoke"] = {"status": "passed", "frames": frames,
            "hands_on_black_frame": len(result.hand_landmarks), "median_ms": statistics.median(timings),
            "p95_ms": float(np.percentile(timings, 95))}
    if camera is not None:
        attempts = []
        for attempt in range(2):
            cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
            try:
                item = {"attempt": attempt + 1, "opened": cap.isOpened()}
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    count, started = 0, time.perf_counter()
                    for _ in range(frames):
                        ok, frame = cap.read()
                        if not ok:
                            break
                        count += 1
                    duration = time.perf_counter() - started
                    item.update({"frames_read": count, "duration_s": duration,
                        "capture_fps": count / duration, "resolution": list(frame.shape[:2][::-1]) if count else None})
                attempts.append(item)
            finally:
                cap.release()
        report["camera"] = {"index": camera, "attempts": attempts, "note": "Captura pura, sem classificação; nenhum frame salvo"}
    else:
        report["camera"] = {"status": "not_tested", "reason": "Use --camera 0 para testar explicitamente"}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames deve ser positivo")
    result = probe(args.camera, args.frames)
    formatted = json.dumps(result, ensure_ascii=False, indent=2)
    print(formatted)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
