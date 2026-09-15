"""Smoke da janela real com captura física; não coleta amostras nem salva vídeo."""
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from gesturelab.store import Store
from gesturelab.ui import MainWindow


def main():
    app = QApplication([])
    temporary = tempfile.TemporaryDirectory(prefix="gesturelab-window-smoke-")
    store = Store(temporary.name)
    window = MainWindow(store)
    report = {"mode": "Qt offscreen com webcam física; sem imagens salvas", "frames": [], "errors": []}
    attempts = 0

    def receive(packet):
        report["frames"].append(packet["state"])
        if len(report["frames"]) == 10:
            window.stop_camera()
            QTimer.singleShot(250, reopen)
        elif len(report["frames"]) == 20:
            window.close()  # fecha com câmera ativa; closeEvent espera liberar

    def reopen():
        nonlocal attempts
        if window.camera_worker is not None:
            QTimer.singleShot(100, reopen)
            return
        attempts += 1
        window.show_page(1 if attempts == 1 else 5)
        window.start_camera("collect" if attempts == 1 else "live")
        window.camera_worker.frame_ready.connect(receive)
        window.camera_worker.error.connect(report["errors"].append)

    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(window.close)
    timer.start(20000)
    window.show()
    QTimer.singleShot(0, reopen)
    app.exec()
    report.update({"openings": attempts, "worker_released": window.camera_worker is None,
                   "samples_saved": len(store.samples()), "window_closed": not window.isVisible()})
    target = Path(__file__).resolve().parent.parent / "docs" / "ui-hardware-validation.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    window.deleteLater()
    app.processEvents()
    temporary.cleanup()
    return 0 if attempts == 2 and len(report["frames"]) >= 20 and report["worker_released"] and not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
