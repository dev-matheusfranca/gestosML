"""Testa captura+extrator em QThread e duas aberturas; nunca salva imagens."""
import json
from pathlib import Path
import statistics
import sys
import time

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gesturelab.camera import CameraWorker, default_model_path  # noqa: E402


def run_once(index=0, count=45):
    worker = CameraWorker(default_model_path(), camera_index=index)
    loop = QEventLoop()
    report = {"frames": 0, "states": {}, "errors": [], "images_saved": False}
    latencies, fps = [], []
    started = time.perf_counter()

    def receive(packet):
        report["frames"] += 1
        state = packet["state"]
        report["states"][state] = report["states"].get(state, 0) + 1
        latencies.append(packet["latency_ms"])
        fps.append(packet["fps"])
        if report["frames"] >= count:
            worker.stop()

    worker.frame_ready.connect(receive)
    worker.error.connect(report["errors"].append)
    worker.finished.connect(loop.quit)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(worker.stop)
    timer.start(20000)
    worker.start()
    loop.exec()
    worker.wait(1000)
    report["duration_s"] = time.perf_counter() - started
    report["stopped"] = not worker.isRunning()
    if latencies:
        report["median_extractor_ms"] = statistics.median(latencies)
        report["median_pipeline_fps"] = statistics.median(fps)
    return report


if __name__ == "__main__":
    app = QCoreApplication(sys.argv)
    result = {"method": "640x480 CPU, 45 frames por abertura; captura+extração+desenho; sem classificador; nenhuma imagem persistida",
              "attempts": [run_once(), run_once()], "unavailable_camera": run_once(index=99)}
    target = Path(__file__).resolve().parent.parent / "docs" / "camera-worker-validation.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
