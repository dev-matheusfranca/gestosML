"""Gera uma captura local da tela inicial para revisão visual da interface.

Uso: .venv\\Scripts\\python.exe scripts\\capture_ui.py
"""

from __future__ import annotations

import os
import tempfile
import gc
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from gesturelab.store import Store
from gesturelab.ui import MainWindow
from gesturelab.ui_widgets import APP_STYLESHEET


def main() -> int:
    app = QApplication([])
    font_path = Path(r"C:\Windows\Fonts\segoeui.ttf")
    if font_path.is_file():
        QFontDatabase.addApplicationFont(str(font_path))
        app.setFont(QFont("Segoe UI"))
    app.setStyleSheet(APP_STYLESHEET)
    with tempfile.TemporaryDirectory(prefix="gesturelab-ui-") as directory:
        store = Store(directory)
        window = MainWindow(store)
        window.resize(1180, 700)
        window.show()
        app.processEvents()
        destination = Path("assets") / "gesturelab-interface-inicial.png"
        destination.parent.mkdir(exist_ok=True)
        if not window.grab().save(str(destination)):
            raise RuntimeError("Não foi possível gravar a captura Qt.")
        window.close()
        window.deleteLater()
        app.processEvents()
        del window
        del store
        gc.collect()
    print(destination.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
