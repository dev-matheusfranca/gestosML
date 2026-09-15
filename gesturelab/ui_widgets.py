"""Componentes visuais reutilizáveis da interface desktop do GestureLab."""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


COLORS = {
    "ink": "#102A43",
    "muted": "#627D98",
    "canvas": "#F6F3EC",
    "panel": "#FFFDF8",
    "line": "#D9E2EC",
    "teal": "#137C8B",
    "teal_soft": "#D9F1EE",
    "coral": "#D95D39",
    "coral_soft": "#FBE4DD",
    "gold": "#D59A20",
    "navy": "#102A43",
}


APP_STYLESHEET = f"""
QMainWindow {{ background: {COLORS["canvas"]}; color: {COLORS["ink"]}; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; }}
QWidget#appRoot, QWidget#pageContent {{ background: {COLORS["canvas"]}; color: {COLORS["ink"]}; }}
QLabel {{ background: transparent; }}
QScrollArea {{ border: 0; background: {COLORS["canvas"]}; }}
QScrollBar:vertical {{ width: 10px; background: transparent; }}
QScrollBar::handle:vertical {{ min-height: 28px; border-radius: 5px; background: #BCCCDC; }}
QFrame#sidebar {{ background: {COLORS["navy"]}; }}
QLabel#brand {{ color: #FFFFFF; font-size: 21px; font-weight: 700; }}
QLabel#subtitle {{ color: #B9D6E5; }}
QPushButton#navButton {{ text-align: left; background: transparent; color: #D9E2EC; border: 0; border-radius: 8px; padding: 10px 12px; font-size: 14px; }}
QPushButton#navButton:hover {{ background: #243E58; }}
QPushButton#navButton:checked {{ background: {COLORS["teal"]}; color: white; font-weight: 700; }}
QFrame#card {{ background: {COLORS["panel"]}; border: 1px solid {COLORS["line"]}; border-radius: 12px; }}
QLabel#metricValue {{ font-size: 25px; font-weight: 700; color: {COLORS["ink"]}; }}
QLabel#metricLabel, QLabel#hint {{ color: {COLORS["muted"]}; }}
QLabel#pageTitle {{ font-size: 28px; font-weight: 700; color: {COLORS["ink"]}; }}
QLabel#pageIntro {{ color: {COLORS["muted"]}; font-size: 14px; }}
QPushButton {{ background: {COLORS["teal"]}; color: white; border: 0; border-radius: 7px; padding: 9px 13px; font-weight: 600; }}
QPushButton:hover {{ background: #0F6975; }}
QPushButton:disabled {{ background: #BCCCDC; color: #627D98; }}
QPushButton#secondary {{ background: transparent; border: 1px solid #9FB3C8; color: {COLORS["ink"]}; }}
QPushButton#danger {{ background: {COLORS["coral"]}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {{ background: #FFFFFF; border: 1px solid #9FB3C8; border-radius: 6px; padding: 7px; min-height: 20px; selection-background-color: {COLORS["teal"]}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{ border: 2px solid {COLORS["teal"]}; }}
QTableWidget {{ background: #FFFFFF; border: 1px solid {COLORS["line"]}; border-radius: 8px; gridline-color: {COLORS["line"]}; }}
QHeaderView::section {{ background: #EAF2F8; border: 0; border-bottom: 1px solid {COLORS["line"]}; padding: 8px; color: {COLORS["ink"]}; font-weight: 700; }}
QProgressBar {{ border: 1px solid #9FB3C8; border-radius: 6px; background: white; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {COLORS["teal"]}; border-radius: 5px; }}
"""


class Card(QFrame):
    """Painel simples que preserva leitura clara sem decoração excessiva."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 15, 16, 15)
        self.layout.setSpacing(8)


class PageHeader(QWidget):
    def __init__(self, title: str, intro: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 10)
        layout.setSpacing(4)
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        heading.setWordWrap(True)
        intro_label = QLabel(intro)
        intro_label.setObjectName("pageIntro")
        intro_label.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(intro_label)


class MetricCard(Card):
    def __init__(self, label: str, value: str = "—", detail: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = QLabel(label)
        self.label.setObjectName("metricLabel")
        self.value = QLabel(value)
        self.value.setObjectName("metricValue")
        self.detail = QLabel(detail)
        self.detail.setObjectName("hint")
        self.detail.setWordWrap(True)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.value)
        self.layout.addWidget(self.detail)

    def set_value(self, value: object, detail: str | None = None) -> None:
        self.value.setText(str(value))
        if detail is not None:
            self.detail.setText(detail)


class EmptyState(Card):
    action_requested = Signal()

    def __init__(self, title: str, body: str, action: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon = QLabel("◌")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(f"font-size: 30px; color: {COLORS['teal']};")
        heading = QLabel(title)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setStyleSheet("font-size: 17px; font-weight: 700;")
        text = QLabel(body)
        text.setObjectName("hint")
        text.setWordWrap(True)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(icon)
        self.layout.addWidget(heading)
        self.layout.addWidget(text)
        if action:
            button = QPushButton(action)
            button.clicked.connect(self.action_requested)
            self.layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignCenter)


class NoticeBox(QFrame):
    def __init__(self, text: str, tone: str = "info", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        colors = {
            "info": (COLORS["teal_soft"], COLORS["teal"]),
            "warning": ("#FFF3CD", "#8A5B00"),
            "danger": (COLORS["coral_soft"], "#9C2C14"),
            "success": ("#E2F4EA", "#1E6A43"),
        }
        bg, fg = colors.get(tone, colors["info"])
        self.setStyleSheet(f"background: {bg}; border: 1px solid {fg}; border-radius: 7px;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(11, 9, 11, 9)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setStyleSheet(f"background: transparent; border: 0; color: {fg};")
        layout.addWidget(self.label)

    def set_text(self, text: str) -> None:
        self.label.setText(text)


class VideoPreview(QLabel):
    """Prévia de vídeo que mantém uma mensagem útil enquanto a câmera está fechada."""

    def __init__(self, empty_text: str = "A câmera ainda não está aberta.", parent: QWidget | None = None) -> None:
        super().__init__(empty_text, parent)
        self.empty_text = empty_text
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(410, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setWordWrap(True)
        self.setStyleSheet(
            "background: #0E2438; color: #D9E2EC; border: 1px solid #486581; border-radius: 10px; padding: 20px;"
        )

    def set_image(self, image: QImage | None) -> None:
        if image is None or image.isNull():
            self.clear()
            self.setText(self.empty_text)
            return
        from PySide6.QtGui import QPixmap

        self.setText("")
        pixmap = QPixmap.fromImage(image)
        self.setPixmap(
            pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        if self.pixmap() and not self.pixmap().isNull():
            self.setPixmap(
                self.pixmap().scaled(
                    self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            )


class LandmarkCanvas(QWidget):
    """Desenha os 21 pontos de uma amostra sem exigir imagem ou vídeo gravado."""

    CONNECTIONS: tuple[tuple[int, int], ...] = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (5, 9),
        (9, 13),
        (13, 17),
    )

    def __init__(self, landmarks: Iterable[float] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[tuple[float, float]] = []
        self.setMinimumSize(190, 170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.set_landmarks(landmarks)

    def set_landmarks(self, landmarks: Iterable[float] | None) -> None:
        values = list(landmarks or [])
        self._points = [(float(values[i]), float(values[i + 1])) for i in range(0, len(values) - 2, 3)]
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#F7FAFC"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self._points:
            painter.setPen(QColor(COLORS["muted"]))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Sem pontos para desenhar")
            return
        xs, ys = [point[0] for point in self._points], [point[1] for point in self._points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span = max(max_x - min_x, max_y - min_y, 0.01)
        margin = 22
        side = min(self.width() - 2 * margin, self.height() - 2 * margin)

        def point_at(point: tuple[float, float]) -> tuple[float, float]:
            return (
                margin + ((point[0] - min_x) / span) * side,
                margin + ((point[1] - min_y) / span) * side,
            )

        painter.setPen(QPen(QColor(COLORS["teal"]), 2.2))
        for first, second in self.CONNECTIONS:
            if first < len(self._points) and second < len(self._points):
                x1, y1 = point_at(self._points[first])
                x2, y2 = point_at(self._points[second])
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        painter.setBrush(QColor(COLORS["coral"]))
        painter.setPen(Qt.PenStyle.NoPen)
        for index, point in enumerate(self._points):
            x, y = point_at(point)
            radius = 5 if index == 0 else 3.8
            painter.drawEllipse(int(x - radius), int(y - radius), int(radius * 2), int(radius * 2))
