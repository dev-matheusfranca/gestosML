"""Aplicação desktop do GestureLab.

A interface mantém captura e treinamento fora da thread de eventos. Ela pode
ser aberta sem câmera, dados ou modelos para servir como ponto inicial seguro.
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .ui_widgets import (
    APP_STYLESHEET,
    Card,
    EmptyState,
    LandmarkCanvas,
    MetricCard,
    NoticeBox,
    PageHeader,
    VideoPreview,
)


def _module(name: str) -> Any:
    return importlib.import_module(f"{__package__}.{name}")


def _value(item: Any, key: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


class TrainingWorker(QThread):
    progress = Signal(str)
    completed = Signal(list)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        store: Any,
        dataset_id: str,
        strategy: str,
        models: list[str],
        seed: int,
        protocol: str,
        preset: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.store, self.dataset_id, self.strategy = store, dataset_id, strategy
        self.models, self.seed, self.protocol = models, seed, protocol
        self.preset = preset
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            ids = _module("ml").train(
                self.store,
                self.dataset_id,
                strategy=self.strategy,
                models=self.models,
                seed=self.seed,
                protocol=self.protocol,
                preset=self.preset,
                cancel=self.cancel_event,
                progress=lambda message: self.progress.emit(str(message)),
            )
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.completed.emit(list(ids))
        except Exception as exc:  # UI must turn domain failures into readable feedback.
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))


class ExperimentVisual(QWidget):
    """Matriz de confusão renderizada localmente a partir das métricas persistidas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.message = QLabel("Selecione um experimento concluído para visualizar a matriz de confusão.")
        self.message.setObjectName("hint")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.canvas: QWidget | None = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self.figure = Figure(figsize=(5.0, 3.2), facecolor="#FFFDF8")
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setMinimumHeight(240)
            layout.addWidget(self.canvas)
        except Exception:
            self.figure = None

    def show_metrics(self, metrics: dict[str, Any], section: str = "validation") -> None:
        report = metrics.get(section) if section in metrics else metrics
        report = report or {}
        classes = list(report.get("classes", []))
        matrix = report.get("confusion_matrix", [])
        if not classes or not matrix:
            self.message.setText("A matriz ficará disponível quando houver métricas de validação ou teste final.")
            if self.canvas:
                self.canvas.setVisible(False)
            return
        self.message.setText("Linhas: rótulo verdadeiro. Colunas: previsão. A diagonal mostra os acertos.")
        if not self.canvas or self.figure is None:
            return
        self.canvas.setVisible(True)
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        image = axis.imshow(matrix, cmap="YlGnBu")
        axis.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
        axis.set_yticks(range(len(classes)), classes)
        axis.set_xlabel("Previsão")
        axis.set_ylabel("Rótulo verdadeiro")
        axis.set_title("Matriz de confusão")
        for row, values in enumerate(matrix):
            for column, value in enumerate(values):
                axis.text(column, row, str(value), ha="center", va="center", color="#102A43")
        self.figure.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
        self.figure.tight_layout()
        self.canvas.draw_idle()


class MainWindow(QMainWindow):
    """Janela principal integrada somente aos contratos do núcleo."""

    NAV = ("Início", "Coleta", "Dataset", "Treinamento", "Experimentos", "Ao vivo", "Revisão", "Aprender")

    def __init__(self, store: Any) -> None:
        super().__init__()
        self.store = store
        self.camera_worker: Any | None = None
        self.camera_target: str | None = None
        self.latest_frame: dict[str, Any] | None = None
        self.collecting = False
        self.collect_session_id: str | None = None
        self.collect_count = 0
        self.last_saved_timestamp_ms: int | None = None
        self.countdown_remaining = 0
        self.train_worker: TrainingWorker | None = None
        self.live_bundle: Any | None = None
        self.action_gate: Any | None = None
        self.temporal_stabilizer: Any | None = None
        self.live_session_id: str | None = None
        self.last_prediction: dict[str, Any] | None = None
        self._nav_buttons: list[QPushButton] = []
        self._close_requested = False
        self.setWindowTitle("GestureLab · laboratório de gestos")
        self.setMinimumSize(980, 620)
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1180, available.width()), min(700, available.height()))
        self._build_shell()
        self._build_pages()
        self._make_shortcuts()
        self.refresh_all()

    # ----- structural helpers -------------------------------------------------
    def _build_shell(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(204)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(14, 20, 14, 16)
        brand = QLabel("GestureLab")
        brand.setObjectName("brand")
        subtitle = QLabel("ML com webcam, local")
        subtitle.setObjectName("subtitle")
        side_layout.addWidget(brand)
        side_layout.addWidget(subtitle)
        side_layout.addSpacing(22)
        for index, title in enumerate(self.NAV):
            button = QPushButton(title)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, i=index: self.show_page(i))
            side_layout.addWidget(button)
            self._nav_buttons.append(button)
        side_layout.addStretch(1)
        privacy = QLabel("Processamento local\nNenhum vídeo é salvo por padrão.")
        privacy.setObjectName("subtitle")
        privacy.setWordWrap(True)
        side_layout.addWidget(privacy)
        self.stack = QStackedWidget()
        root_layout.addWidget(sidebar)
        root_layout.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        # As páginas ainda não foram criadas nesta etapa da inicialização.
        self._nav_buttons[0].setChecked(True)

    def _page(self) -> tuple[QWidget, QVBoxLayout]:
        viewport = QWidget()
        viewport.setObjectName("pageContent")
        layout = QVBoxLayout(viewport)
        layout.setContentsMargins(28, 24, 28, 30)
        layout.setSpacing(16)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(viewport)
        self.stack.addWidget(scroll)
        return viewport, layout

    def _section(self, title: str, body: str | None = None) -> Card:
        card = Card()
        label = QLabel(title)
        label.setStyleSheet("font-size: 16px; font-weight: 700;")
        card.layout.addWidget(label)
        if body:
            note = QLabel(body)
            note.setObjectName("hint")
            note.setWordWrap(True)
            card.layout.addWidget(note)
        return card

    def show_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for position, button in enumerate(self._nav_buttons):
            button.setChecked(position == index)
        self.refresh_page(index)

    def _make_shortcuts(self) -> None:
        refresh_action = QAction(self)
        refresh_action.setShortcut("Ctrl+R")
        refresh_action.triggered.connect(self.refresh_all)
        self.addAction(refresh_action)
        collect_action = QAction(self)
        collect_action.setShortcut("Ctrl+Shift+C")
        collect_action.triggered.connect(lambda: self.show_page(1))
        self.addAction(collect_action)

    # ----- safe store queries --------------------------------------------------
    def _call(self, name: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
        try:
            return getattr(self.store, name)(*args, **kwargs)
        except Exception as exc:
            self.statusBar().showMessage(f"Não foi possível atualizar {name}: {exc}", 7000)
            return default

    def _categories(self) -> list[str]:
        return list(self._call("categories", default=[]) or [])

    def _samples(self) -> Any:
        return self._call("samples", default=None)

    @staticmethod
    def _rows(frame: Any) -> list[dict[str, Any]]:
        if frame is None:
            return []
        try:
            return frame.to_dict("records")
        except Exception:
            return list(frame) if isinstance(frame, list) else []

    # ----- Start page ----------------------------------------------------------
    def _build_pages(self) -> None:
        self._build_home()
        self._build_collect()
        self._build_dataset()
        self._build_training()
        self._build_experiments()
        self._build_live()
        self._build_review()
        self._build_learn()

    def _build_home(self) -> None:
        _, layout = self._page()
        layout.addWidget(
            PageHeader(
                "Laboratório de gestos",
                "Colete exemplos seus, treine classificadores locais e veja o que eles aprenderam.",
            )
        )
        grid = QGridLayout()
        grid.setSpacing(12)
        self.home_camera = MetricCard("Câmera", "Fechada", "Abra somente em Coleta ou Ao vivo.")
        self.home_categories = MetricCard("Categorias", "0", "Cadastre os gestos que quer comparar.")
        self.home_samples = MetricCard("Amostras", "0", "Uma amostra é um conjunto de 21 pontos da mão.")
        self.home_sessions = MetricCard("Sessões", "0", "Varie sessão, distância e iluminação.")
        self.home_experiments = MetricCard("Experimentos", "0", "Resultados aparecem após o treinamento.")
        self.home_model = MetricCard("Modelo ativo", "Nenhum", "Escolha um experimento concluído.")
        for index, card in enumerate(
            (
                self.home_camera,
                self.home_categories,
                self.home_samples,
                self.home_sessions,
                self.home_experiments,
                self.home_model,
            )
        ):
            grid.addWidget(card, index // 3, index % 3)
        layout.addLayout(grid)
        self.home_next = NoticeBox(
            "Próxima etapa: cadastre categorias e abra a câmera em Coleta para registrar exemplos variados."
        )
        layout.addWidget(self.home_next)
        distribution = self._section(
            "Distribuição atual",
            "O protocolo exige pelo menos quatro sessões independentes com todas as categorias para separar treino, validação e teste.",
        )
        self.home_distribution = QLabel()
        self.home_distribution.setWordWrap(True)
        self.home_distribution.setObjectName("hint")
        distribution.layout.addWidget(self.home_distribution)
        layout.addWidget(distribution)
        layout.addStretch(1)

    def _build_collect(self) -> None:
        _, layout = self._page()
        layout.addWidget(
            PageHeader(
                "Coleta",
                "Abra a câmera quando estiver pronto. A prévia espelhada não altera os dados usados pelo modelo.",
            )
        )
        layout.addWidget(
            NoticeBox("São salvos apenas pontos e metadados por padrão. Fotos ou vídeos não são gravados nesta versão.")
        )
        row = QHBoxLayout()
        row.setSpacing(16)
        left = QVBoxLayout()
        right = QVBoxLayout()
        self.collect_preview = VideoPreview("Câmera fechada. Clique em “Abrir câmera” para começar.")
        left.addWidget(self.collect_preview, 1)
        self.collect_detection = QLabel("Câmera fechada")
        self.collect_detection.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.collect_detection)
        self.collect_state = QLabel("Abra a câmera e escolha um gesto para começar a coleta.")
        self.collect_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.collect_state.setWordWrap(True)
        self.collect_state.setStyleSheet("font-size: 17px; font-weight: 700; padding: 10px;")
        left.addWidget(self.collect_state)
        row.addLayout(left, 3)
        settings = self._section("Sessão de coleta", "1. Escolha o gesto. 2. Inicie a contagem. 3. Confira o contador de amostras. Abrir a câmera só exibe a prévia.")
        form = QFormLayout()
        self.collect_participant = QLineEdit()
        self.collect_participant.setPlaceholderText("Pseudônimo opcional, ex.: pessoa-01")
        self.collect_label = QComboBox()
        self.collect_purpose = QComboBox()
        self.collect_purpose.addItem("Desenvolvimento", "development")
        self.collect_purpose.addItem("Desafio (novo contexto)", "challenge")
        self.collect_rate = QSpinBox()
        self.collect_rate.setRange(1, 10)
        self.collect_rate.setValue(3)
        self.collect_rate.setSuffix(" Hz")
        self.collect_camera_index = QSpinBox()
        self.collect_camera_index.setRange(0, 9)
        self.collect_min_confidence = QDoubleSpinBox()
        self.collect_min_confidence.setRange(0.1, 0.99)
        self.collect_min_confidence.setSingleStep(0.05)
        self.collect_min_confidence.setValue(0.5)
        self.collect_min_size = QDoubleSpinBox()
        self.collect_min_size.setRange(0.01, 0.50)
        self.collect_min_size.setDecimals(2)
        self.collect_min_size.setSingleStep(0.01)
        self.collect_min_size.setValue(0.06)
        form.addRow("Participante", self.collect_participant)
        form.addRow("Finalidade", self.collect_purpose)
        form.addRow("Gesto", self.collect_label)
        form.addRow("Frequência", self.collect_rate)
        form.addRow("Câmera", self.collect_camera_index)
        form.addRow("Confiança mín.", self.collect_min_confidence)
        form.addRow("Tamanho mín.", self.collect_min_size)
        settings.layout.addLayout(form)
        control = QHBoxLayout()
        self.collect_open = QPushButton("Abrir câmera")
        self.collect_open.clicked.connect(lambda: self.start_camera("collect"))
        self.collect_close = QPushButton("Fechar")
        self.collect_close.setObjectName("secondary")
        self.collect_close.clicked.connect(self.stop_camera)
        control.addWidget(self.collect_open)
        control.addWidget(self.collect_close)
        settings.layout.addLayout(control)
        self.collect_start = QPushButton("Iniciar contagem (3 s)")
        self.collect_start.clicked.connect(self.begin_countdown)
        self.collect_pause = QPushButton("Pausar coleta")
        self.collect_pause.setObjectName("secondary")
        self.collect_pause.clicked.connect(self.pause_collection)
        self.collect_finish = QPushButton("Finalizar sessão")
        self.collect_finish.setObjectName("danger")
        self.collect_finish.clicked.connect(self.finish_collection)
        settings.layout.addWidget(self.collect_start)
        settings.layout.addWidget(self.collect_pause)
        settings.layout.addWidget(self.collect_finish)
        self.collect_counter = QLabel("0 amostras nesta sessão")
        self.collect_counter.setStyleSheet("font-size: 17px; font-weight: 700;")
        settings.layout.addWidget(self.collect_counter)
        right.addWidget(settings)
        right.addStretch(1)
        row.addLayout(right, 2)
        layout.addLayout(row)
        quality = self._section(
            "Critérios de qualidade",
            "Uma amostra só é registrada quando há exatamente uma mão, pontos válidos e tamanho mínimo configurado pelo capturador.",
        )
        self.collect_quality = QLabel("Sem leitura de qualidade.")
        self.collect_quality.setObjectName("hint")
        self.collect_quality.setWordWrap(True)
        quality.layout.addWidget(self.collect_quality)
        layout.addWidget(quality)
        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self._countdown_tick)
        self.sample_timer = QTimer(self)
        self.sample_timer.timeout.connect(self.record_current_sample)
        self.collect_label.currentIndexChanged.connect(self._collection_gesture_changed)
        self._sync_collection_controls()

    def _build_dataset(self) -> None:
        _, layout = self._page()
        layout.addWidget(
            PageHeader(
                "Dataset",
                "Revise o que será usado no treinamento. O histórico de experimentos aponta para versões imutáveis.",
            )
        )
        add_card = self._section(
            "Categorias", "Comece com gestos visualmente distintos e inclua exemplos variados de cada um."
        )
        category_row = QHBoxLayout()
        self.new_category = QLineEdit()
        self.new_category.setPlaceholderText("Nome do novo gesto")
        add_button = QPushButton("Adicionar categoria")
        add_button.clicked.connect(self.add_category)
        category_row.addWidget(self.new_category, 1)
        category_row.addWidget(add_button)
        add_card.layout.addLayout(category_row)
        self.category_list = QLabel()
        self.category_list.setObjectName("hint")
        self.category_list.setWordWrap(True)
        add_card.layout.addWidget(self.category_list)
        layout.addWidget(add_card)
        review = self._section(
            "Amostras registradas", "Selecione uma amostra para alterar seu rótulo ou removê-la após confirmar."
        )
        self.dataset_session_filter = QComboBox()
        self.dataset_session_filter.currentIndexChanged.connect(self.refresh_dataset)
        review.layout.addWidget(self.dataset_session_filter)
        self.dataset_distribution = QLabel()
        self.dataset_distribution.setWordWrap(True)
        review.layout.addWidget(self.dataset_distribution)
        self.dataset_table = QTableWidget(0, 6)
        self.dataset_table.setHorizontalHeaderLabels(["ID", "Sessão", "Gesto", "Participante", "Lado", "Qualidade"])
        self.dataset_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.dataset_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.dataset_table.itemSelectionChanged.connect(self._dataset_selection_changed)
        review.layout.addWidget(self.dataset_table)
        sample_actions = QHBoxLayout()
        self.relabel_combo = QComboBox()
        self.relabel_button = QPushButton("Aplicar rótulo")
        self.relabel_button.setObjectName("secondary")
        self.relabel_button.clicked.connect(self.relabel_selected)
        self.delete_button = QPushButton("Excluir selecionada")
        self.delete_button.setObjectName("danger")
        self.delete_button.clicked.connect(self.delete_selected)
        sample_actions.addWidget(QLabel("Novo rótulo:"))
        sample_actions.addWidget(self.relabel_combo)
        sample_actions.addWidget(self.relabel_button)
        sample_actions.addStretch(1)
        sample_actions.addWidget(self.delete_button)
        review.layout.addLayout(sample_actions)
        layout.addWidget(review)
        tools = self._section(
            "Integridade e versões", "Duplicatas exatas e valores inválidos são sinalizados antes de treinar."
        )
        actions = QHBoxLayout()
        self.validate_dataset = QPushButton("Validar dataset")
        self.validate_dataset.setObjectName("secondary")
        self.validate_dataset.clicked.connect(self.validate_data)
        self.snapshot_dataset = QPushButton("Criar versão imutável")
        self.snapshot_dataset.clicked.connect(self.snapshot_data)
        self.export_dataset = QPushButton("Exportar dados numéricos")
        self.export_dataset.setObjectName("secondary")
        self.export_dataset.clicked.connect(self.export_data)
        actions.addWidget(self.validate_dataset)
        actions.addWidget(self.snapshot_dataset)
        actions.addWidget(self.export_dataset)
        actions.addStretch(1)
        tools.layout.addLayout(actions)
        self.dataset_feedback = QLabel("Ainda não validado.")
        self.dataset_feedback.setObjectName("hint")
        self.dataset_feedback.setWordWrap(True)
        tools.layout.addWidget(self.dataset_feedback)
        layout.addWidget(tools)
        self.dataset_preview = LandmarkCanvas()
        self.dataset_preview.setVisible(False)
        layout.addWidget(self.dataset_preview)

    def _build_training(self) -> None:
        _, layout = self._page()
        layout.addWidget(
            PageHeader(
                "Treinamento",
                "Compare modelos com grupos de sessão separados. Frames da mesma sessão não devem vazar para a avaliação.",
            )
        )
        layout.addWidget(
            NoticeBox(
                "O MediaPipe localiza pontos da mão; os classificadores abaixo aprendem somente com seus exemplos coletados.",
                "info",
            )
        )
        form_card = self._section(
            "Configuração", "O teste final é reservado. Use desenvolvimento e validação para decidir melhorias."
        )
        form = QFormLayout()
        self.train_dataset = QComboBox()
        self.train_strategy = QComboBox()
        for caption, value in [("Coordenadas normalizadas", "normalized"), ("Coordenadas brutas", "raw"),
                               ("Distâncias", "distances"), ("Ângulos", "angles")]:
            self.train_strategy.addItem(caption, value)
        model_picker = QWidget()
        model_row = QHBoxLayout(model_picker)
        model_row.setContentsMargins(0, 0, 0, 0)
        self.train_model_checks: dict[str, QCheckBox] = {}
        for model_name, caption in (
            ("dummy", "Baseline"),
            ("logistic", "Logística"),
            ("forest", "Floresta"),
            ("mlp", "MLP"),
        ):
            check = QCheckBox(caption)
            check.setChecked(model_name in ("dummy", "logistic", "forest"))
            self.train_model_checks[model_name] = check
            model_row.addWidget(check)
        model_row.addStretch(1)
        self.train_seed = QSpinBox()
        self.train_seed.setRange(0, 999999)
        self.train_seed.setValue(42)
        self.train_protocol = QComboBox()
        self.train_protocol.addItem("Sessões independentes", "session")
        self.train_protocol.addItem("Novos participantes", "participant")
        self.train_preset = QComboBox()
        self.train_preset.addItem("Rápido (grade pequena)", "quick")
        self.train_preset.addItem("Uma configuração por modelo", "single")
        form.addRow("Versão do dataset", self.train_dataset)
        form.addRow("Atributos", self.train_strategy)
        form.addRow("Modelos", model_picker)
        form.addRow("Hiperparâmetros", self.train_preset)
        form.addRow("Semente", self.train_seed)
        form.addRow("Protocolo", self.train_protocol)
        form_card.layout.addLayout(form)
        buttons = QHBoxLayout()
        self.train_start = QPushButton("Treinar modelo")
        self.train_start.clicked.connect(self.start_training)
        self.train_cancel = QPushButton("Cancelar com segurança")
        self.train_cancel.setObjectName("danger")
        self.train_cancel.clicked.connect(self.cancel_training)
        self.train_cancel.setEnabled(False)
        buttons.addWidget(self.train_start)
        buttons.addWidget(self.train_cancel)
        buttons.addStretch(1)
        form_card.layout.addLayout(buttons)
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 1)
        self.train_progress.setValue(0)
        form_card.layout.addWidget(self.train_progress)
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setPlaceholderText("O progresso e os avisos aparecerão aqui.")
        self.train_log.setMinimumHeight(125)
        form_card.layout.addWidget(self.train_log)
        layout.addWidget(form_card)
        layout.addWidget(
            NoticeBox(
                "Se os dados não permitirem uma divisão válida por sessão ou participante, o treinamento deve explicar o motivo em vez de avaliar frames aleatoriamente.",
                "warning",
            )
        )
        layout.addStretch(1)

    def _build_experiments(self) -> None:
        _, layout = self._page()
        layout.addWidget(
            PageHeader(
                "Experimentos",
                "Leia métricas junto do protocolo, dataset e limitações. Acurácia isolada não mostra todos os erros.",
            )
        )
        self.experiment_empty = EmptyState(
            "Nenhum experimento ainda",
            "Crie uma versão do dataset e treine pelo menos um modelo para ver comparações e relatórios.",
            "Ir para treinamento",
        )
        self.experiment_empty.action_requested.connect(lambda: self.show_page(3))
        layout.addWidget(self.experiment_empty)
        self.experiment_table = QTableWidget(0, 8)
        self.experiment_table.setHorizontalHeaderLabels(
            ["ID", "Modelo", "Estado", "Dataset", "Acurácia", "Macro-F1", "Tempo", "Protocolo"]
        )
        self.experiment_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.experiment_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.experiment_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.experiment_table.itemSelectionChanged.connect(self.show_selected_experiment)
        layout.addWidget(self.experiment_table)
        detail = self._section(
            "Detalhes e avaliação", "Selecione um experimento concluído e ative-o para usar na demonstração ao vivo."
        )
        action_row = QHBoxLayout()
        self.activate_experiment = QPushButton("Usar como modelo ativo")
        self.activate_experiment.clicked.connect(self.activate_selected_experiment)
        self.final_evaluate = QPushButton("Avaliar teste final")
        self.final_evaluate.setObjectName("secondary")
        self.final_evaluate.clicked.connect(self.evaluate_selected_final)
        self.compare_experiments = QPushButton("Comparar selecionados")
        self.compare_experiments.setObjectName("secondary")
        self.compare_experiments.clicked.connect(self.compare_selected_experiments)
        action_row.addWidget(self.activate_experiment)
        action_row.addWidget(self.final_evaluate)
        action_row.addWidget(self.compare_experiments)
        action_row.addStretch(1)
        detail.layout.addLayout(action_row)
        self.experiment_details = QLabel("Nenhum experimento selecionado.")
        self.experiment_details.setObjectName("hint")
        self.experiment_details.setWordWrap(True)
        detail.layout.addWidget(self.experiment_details)
        self.experiment_report = QTableWidget(0, 4)
        self.experiment_report.setHorizontalHeaderLabels(["Classe", "Precisão", "Recall", "F1 / suporte"])
        self.experiment_report.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.experiment_report.setMaximumHeight(190)
        detail.layout.addWidget(self.experiment_report)
        self.experiment_visual = ExperimentVisual()
        detail.layout.addWidget(self.experiment_visual)
        layout.addWidget(detail)
        layout.addStretch(1)

    def _build_live(self) -> None:
        _, layout = self._page()
        layout.addWidget(
            PageHeader(
                "Demonstração ao vivo",
                "O modelo ativo classifica os pontos da sua mão. Ações acontecem só dentro deste painel.",
            )
        )
        layout.addWidget(
            NoticeBox(
                "Não há controle do sistema operacional. Um estado neutro rearma a ação e o intervalo evita repetições.",
                "info",
            )
        )
        row = QHBoxLayout()
        self.live_preview = VideoPreview("Câmera fechada. Escolha um modelo ativo e abra a câmera.")
        row.addWidget(self.live_preview, 3)
        panel = self._section("Estado do reconhecimento")
        self.live_model = QLabel("Modelo ativo: nenhum")
        self.live_prediction = QLabel("Aguardando modelo e câmera")
        self.live_prediction.setStyleSheet("font-size: 21px; font-weight: 700;")
        self.live_score = QLabel("Pontuação: —")
        self.live_performance = QLabel("FPS: — · latência: —")
        self.live_action = QLabel("Demonstração: neutra")
        self.live_action.setStyleSheet("font-size: 16px; font-weight: 700;")
        for widget in (self.live_model, self.live_prediction, self.live_score, self.live_performance, self.live_action):
            widget.setWordWrap(True)
            panel.layout.addWidget(widget)
        stabilization = QFormLayout()
        self.live_window = QSpinBox()
        self.live_window.setRange(1, 15)
        self.live_window.setValue(5)
        self.live_consensus = QDoubleSpinBox()
        self.live_consensus.setRange(0.5, 1.0)
        self.live_consensus.setSingleStep(0.05)
        self.live_consensus.setValue(0.7)
        stabilization.addRow("Janela temporal", self.live_window)
        stabilization.addRow("Consenso mínimo", self.live_consensus)
        panel.layout.addLayout(stabilization)
        controls = QHBoxLayout()
        self.live_open = QPushButton("Abrir câmera")
        self.live_open.clicked.connect(lambda: self.start_camera("live"))
        self.live_close = QPushButton("Fechar")
        self.live_close.setObjectName("secondary")
        self.live_close.clicked.connect(self.stop_camera)
        self.live_mark_difficult = QPushButton("Marcar como incerta/incorreta")
        self.live_mark_difficult.setObjectName("secondary")
        self.live_mark_difficult.clicked.connect(self.mark_live_difficult)
        controls.addWidget(self.live_open)
        controls.addWidget(self.live_close)
        panel.layout.addLayout(controls)
        panel.layout.addWidget(self.live_mark_difficult)
        panel.layout.addStretch(1)
        row.addWidget(panel, 2)
        layout.addLayout(row)
        demo = self._section(
            "Área de demonstração",
            "Cada gesto move o marcador para uma posição. Retire a mão ou use um estado incerto antes de acionar novamente.",
        )
        self.slide_demo = QLabel("●  ○  ○  ○  ○")
        self.slide_demo.setStyleSheet("font-size: 28px; color: #137C8B;")
        self.slide_demo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        demo.layout.addWidget(self.slide_demo)
        layout.addWidget(demo)
        layout.addStretch(1)

    def _build_review(self) -> None:
        _, layout = self._page()
        layout.addWidget(
            PageHeader(
                "Revisão de exemplos difíceis",
                "Uma previsão só vira dado de treino após a pessoa revisar e informar o rótulo correto.",
            )
        )
        layout.addWidget(
            NoticeBox(
                "O GestureLab nunca usa a própria previsão como verdade para retreinar automaticamente.", "warning"
            )
        )
        self.review_empty = EmptyState(
            "Nenhum exemplo pendente",
            "Durante a demonstração, marque uma leitura como incerta ou incorreta para revisá-la aqui.",
        )
        layout.addWidget(self.review_empty)
        self.review_table = QTableWidget(0, 5)
        self.review_table.setHorizontalHeaderLabels(["ID", "Sessão", "Previsão", "Pontuação", "Lado"])
        self.review_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.review_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.review_table)
        self.review_preview = LandmarkCanvas()
        self.review_preview.setMaximumHeight(220)
        self.review_table.itemSelectionChanged.connect(self._review_selection_changed)
        layout.addWidget(self.review_preview)
        review_actions = self._section(
            "Decisão de revisão", "O rótulo verdadeiro é obrigatório para incorporar o exemplo depois."
        )
        action_row = QHBoxLayout()
        self.review_label = QComboBox()
        self.review_apply = QPushButton("Confirmar rótulo verdadeiro")
        self.review_apply.clicked.connect(self.apply_difficult_review)
        self.review_discard = QPushButton("Descartar exemplo")
        self.review_discard.setObjectName("danger")
        self.review_discard.clicked.connect(self.discard_difficult)
        action_row.addWidget(QLabel("Rótulo verdadeiro:"))
        action_row.addWidget(self.review_label)
        action_row.addWidget(self.review_apply)
        action_row.addStretch(1)
        action_row.addWidget(self.review_discard)
        review_actions.layout.addLayout(action_row)
        layout.addWidget(review_actions)
        layout.addStretch(1)

    def _build_learn(self) -> None:
        _, layout = self._page()
        layout.addWidget(
            PageHeader(
                "Aprender",
                "Use as explicações quando precisar; elas não interrompem a coleta, o treino ou a demonstração.",
            )
        )
        concepts = [
            (
                "Amostra, rótulo e atributo",
                "Uma amostra contém os 21 pontos. O rótulo é o gesto que você informa. Atributos são as medidas que chegam ao classificador.",
            ),
            (
                "Normalização",
                "Subtrair o punho e dividir por uma escala da mão reduz o efeito de posição e tamanho. Não elimina toda diferença de rotação.",
            ),
            (
                "Treino, validação e teste",
                "Treino ajusta o modelo; validação ajuda a escolher; teste final verifica uma vez a generalização. Misturar sessões causa vazamento.",
            ),
            (
                "Matriz de confusão",
                "Cada linha é um gesto verdadeiro e cada coluna uma previsão. Ela revela quais gestos parecidos são confundidos.",
            ),
            (
                "Modelo pré-treinado e classificador",
                "O MediaPipe encontra pontos da mão. O classificador do GestureLab aprende suas categorias com os seus dados.",
            ),
        ]
        for title, text in concepts:
            layout.addWidget(self._section(title, text))
        exercises = self._section(
            "Exercícios investigativos",
            "Registre uma hipótese, o procedimento e o que observou. Não conclua uma melhoria sem comparar o mesmo protocolo.",
        )
        exercise_text = QLabel(
            "1. Treine em uma sessão e teste em outra.\n2. Compare atributos brutos e normalizados.\n3. Reduza exemplos de uma classe e observe Macro-F1.\n4. Compare exemplos aleatórios contra exemplos revisados por erro.\n5. Varie iluminação, distância e inclinação."
        )
        exercise_text.setWordWrap(True)
        exercises.layout.addWidget(exercise_text)
        layout.addWidget(exercises)
        layout.addStretch(1)

    # ----- refresh -------------------------------------------------------------
    def refresh_all(self) -> None:
        for index in range(len(self.NAV)):
            self.refresh_page(index)

    def refresh_page(self, index: int) -> None:
        if index == 0:
            self.refresh_home()
        elif index == 1:
            self.refresh_collect_options()
        elif index == 2:
            self.refresh_dataset()
        elif index == 3:
            self.refresh_train_options()
        elif index == 4:
            self.refresh_experiments()
        elif index == 5:
            self.refresh_live_model()
        elif index == 6:
            self.refresh_difficult()

    def refresh_home(self) -> None:
        categories, samples = self._categories(), self._rows(self._samples())
        sessions = self._call("sessions", default=[]) or []
        experiments = self._call("experiments", default=[]) or []
        active = self._call("active", default=None)
        self.home_camera.set_value(
            "Aberta" if self.camera_worker else "Fechada",
            "Captura em execução." if self.camera_worker else "Abra somente quando for usar.",
        )
        self.home_categories.set_value(len(categories), ", ".join(categories) or "Cadastre o primeiro gesto.")
        self.home_samples.set_value(len(samples), "Pontos e metadados locais.")
        self.home_sessions.set_value(len(sessions), "Use várias sessões para avaliação.")
        self.home_experiments.set_value(len(experiments), "Concluídos, falhos ou cancelados.")
        active_experiment = next((item for item in experiments if item.get("id") == active), None)
        model_names = {"dummy": "Referência", "logistic": "Regressão logística", "forest": "Random Forest", "mlp": "Rede MLP"}
        self.home_model.set_value(model_names.get(active_experiment.get("model"), "Modelo ativo") if active_experiment else "Nenhum",
                                  "Disponível em Ao vivo." if active_experiment else "Ative um experimento concluído para demonstrar.")
        counts: dict[str, int] = {}
        for sample in samples:
            counts[str(sample.get("label", "sem rótulo"))] = counts.get(str(sample.get("label", "sem rótulo")), 0) + 1
        self.home_distribution.setText(
            " · ".join(f"{label}: {count}" for label, count in counts.items())
            or "Ainda não há amostras. Colete cada gesto em pelo menos quatro sessões independentes."
        )
        if not categories:
            message = "Próxima etapa: cadastre as categorias do seu experimento no Dataset."
        elif not samples:
            message = "Próxima etapa: abra a câmera em Coleta e registre exemplos em uma sessão identificável."
        elif not active:
            message = "Próxima etapa: crie uma versão imutável e treine um modelo com sessões separadas."
        else:
            message = "Próxima etapa: teste o modelo em uma nova sessão e revise previsões incertas ou incorretas."
        self.home_next.set_text(message)

    @staticmethod
    def _refill(
        combo: QComboBox, entries: list[tuple[str, str]], current: str | None = None, placeholder: str | None = None
    ) -> None:
        previous = current or combo.currentData() or combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        if placeholder:
            combo.addItem(placeholder, "")
        for text, value in entries:
            combo.addItem(text, value)
        position = combo.findData(previous)
        if position >= 0:
            combo.setCurrentIndex(position)
        combo.blockSignals(False)

    def refresh_collect_options(self) -> None:
        categories = self._categories()
        self._refill(self.collect_label, [(item, item) for item in categories], placeholder="Selecione um gesto")
        self._sync_collection_controls()

    def refresh_dataset(self) -> None:
        categories, rows = self._categories(), self._rows(self._samples())
        sessions = self._call("sessions", default=[]) or []
        self._refill(self.dataset_session_filter,
            [(f"{item['id'][:16]} · {item.get('participant') or 'sem pseudônimo'} · {item.get('sample_count', 0)} amostras", item['id']) for item in sessions],
            placeholder="Todas as sessões")
        selected_session = self.dataset_session_filter.currentData()
        if selected_session:
            rows = [row for row in rows if row.get("session_id") == selected_session]
        counts = {}
        for row in rows:
            counts[row.get("label", "—")] = counts.get(row.get("label", "—"), 0) + 1
        self.dataset_distribution.setText(" · ".join(f"{label}: {count}" for label, count in counts.items()) or "Nenhuma amostra nesta seleção.")
        self.category_list.setText("Categorias cadastradas: " + (", ".join(categories) if categories else "nenhuma"))
        self._refill(self.relabel_combo, [(item, item) for item in categories], placeholder="Escolha")
        self.dataset_table.setRowCount(len(rows))
        self._dataset_rows = rows
        for row_index, sample in enumerate(rows):
            values = [
                sample.get("id", ""),
                sample.get("session_id", ""),
                sample.get("label", ""),
                sample.get("participant", ""),
                sample.get("handedness", ""),
                sample.get("quality", ""),
            ]
            for column, value in enumerate(values):
                self.dataset_table.setItem(row_index, column, QTableWidgetItem(str(value)))
        self.dataset_table.resizeColumnsToContents()

    def refresh_train_options(self) -> None:
        snapshots = self._call("snapshots", default=[]) or []
        entries = [
            (f"{_value(item, 'id')} · {_value(item, 'created_at', '')}", str(_value(item, "id"))) for item in snapshots
        ]
        self._refill(self.train_dataset, entries, placeholder="Crie uma versão no Dataset")
        can_train = bool(entries) and self.train_worker is None
        self.train_start.setEnabled(can_train)

    def refresh_experiments(self) -> None:
        experiments = self._call("experiments", default=[]) or []
        self._experiment_rows = list(experiments)
        self.experiment_empty.setVisible(not bool(experiments))
        self.experiment_table.setVisible(bool(experiments))
        self.experiment_table.setRowCount(len(experiments))
        for row_index, exp in enumerate(experiments):
            metrics = _value(exp, "metrics", {}) or {}
            values = [
                _value(exp, "id"),
                _value(exp, "model"),
                _value(exp, "status"),
                _value(exp, "dataset_id"),
                metrics.get("accuracy", "—"),
                metrics.get("macro_f1", "—"),
                _value(exp, "duration_s", "—"),
                _value(exp, "protocol", "—"),
            ]
            for col, value in enumerate(values):
                self.experiment_table.setItem(row_index, col, QTableWidgetItem(str(value)))
        self.experiment_table.resizeColumnsToContents()

    def _selected_experiments(self) -> list[dict[str, Any]]:
        selected = (
            self.experiment_table.selectionModel().selectedRows() if self.experiment_table.selectionModel() else []
        )
        return [self._experiment_rows[index.row()] for index in selected]

    def show_selected_experiment(self) -> None:
        experiments = self._selected_experiments()
        if len(experiments) != 1:
            if len(experiments) > 1:
                self.experiment_details.setText(
                    "Vários experimentos selecionados. Use “Comparar selecionados” para checar compatibilidade."
                )
            return
        experiment = experiments[0]
        metrics = _value(experiment, "metrics", {}) or {}
        section = "final" if metrics.get("final") else "validation"
        report = metrics.get(section) or {}
        self.experiment_details.setText(
            " · ".join(
                [
                    f"Modelo: {_value(experiment, 'model')}",
                    f"Estado: {_value(experiment, 'status')}",
                    f"Atributos: {_value(experiment, 'strategy')}",
                    f"Dataset: {_value(experiment, 'dataset_id')}",
                    f"Protocolo: {_value(experiment, 'protocol')}",
                    f"Parâmetros: {_value(experiment, 'parameters', {})}",
                    f"Limiar: {_value(experiment, 'threshold', '—')}",
                    f"Cobertura: {report.get('coverage', '—')}",
                    f"Latência do classificador: {metrics.get('latency_ms', '—')} ms",
                    f"Macro-F1 entre aceitas: {report.get('accepted_macro_f1', '—')}",
                    f"Desafio: {metrics.get('challenge') or 'Ainda não coletado'}",
                    f"Validação por grupos no treino: {metrics.get('group_cv_train') or 'Grupos insuficientes'}",
                    f"Avisos de ajuste: {metrics.get('warnings') or 'Nenhum'}",
                ]
            )
        )
        per_class = report.get("per_class", {})
        classes = report.get("classes", [])
        self.experiment_report.setRowCount(len(classes))
        for row, label in enumerate(classes):
            values = per_class.get(label, {})
            cells = [
                label,
                f"{float(values.get('precision', 0)):.3f}",
                f"{float(values.get('recall', 0)):.3f}",
                f"{float(values.get('f1-score', 0)):.3f} / {int(values.get('support', 0))}",
            ]
            for column, cell in enumerate(cells):
                self.experiment_report.setItem(row, column, QTableWidgetItem(str(cell)))
        self.experiment_report.resizeColumnsToContents()
        self.experiment_visual.show_metrics(metrics, section)

    def compare_selected_experiments(self) -> None:
        experiments = self._selected_experiments()
        if len(experiments) < 2:
            self.statusBar().showMessage("Selecione pelo menos dois experimentos para comparar.", 5000)
            return
        try:
            comparison = _module("ml").compare(self.store, [_value(item, "id") for item in experiments])
            differences = comparison.get("differences", {})
            if comparison.get("compatible"):
                scores = []
                for item in experiments:
                    metrics = _value(item, "metrics", {}) or {}
                    scores.append(
                        f"{_value(item, 'model')}: acc {metrics.get('accuracy', '—')} · F1 {metrics.get('macro_f1', '—')}"
                    )
                variables = comparison.get("controlled_variables", {})
                self.experiment_details.setText("Comparação com avaliação congelada. " + " | ".join(scores)
                    + (f" · Variáveis diferentes: {variables}" if variables else ""))
            else:
                detail = "; ".join(f"{key}: {', '.join(values)}" for key, values in differences.items())
                self.experiment_details.setText("Comparação não equivalente. Diferenças metodológicas: " + detail)
        except Exception as exc:
            self.experiment_details.setText(f"Não foi possível comparar: {exc}")

    def refresh_live_model(self) -> None:
        active = self._call("active", default=None)
        self.live_model.setText(f"Modelo ativo: {active or 'nenhum'}")

    def refresh_difficult(self) -> None:
        self.review_preview.set_landmarks(None)
        entries = [item for item in (self._call("difficult", default=[]) or []) if _value(item, "status") == "pending"]
        self._difficult_rows = list(entries)
        self.review_empty.setVisible(not bool(entries))
        self.review_table.setVisible(bool(entries))
        self.review_table.setRowCount(len(entries))
        for row_index, item in enumerate(entries):
            values = [
                _value(item, "id"),
                _value(item, "session_id"),
                _value(item, "predicted_label", "—"),
                _value(item, "score", "—"),
                _value(item, "handedness", "—"),
            ]
            for col, value in enumerate(values):
                self.review_table.setItem(row_index, col, QTableWidgetItem(str(value)))
        self.review_table.resizeColumnsToContents()
        self._refill(self.review_label, [(item, item) for item in self._categories()], placeholder="Informe o rótulo")

    # ----- dataset actions -----------------------------------------------------
    def add_category(self) -> None:
        name = self.new_category.text().strip()
        if not name:
            self.statusBar().showMessage("Informe um nome para a categoria.", 4000)
            return
        result = self._call("add_category", name, default=False)
        if result is not False:
            self.new_category.clear()
            self.refresh_all()
            self.statusBar().showMessage(f"Categoria “{name}” adicionada.", 5000)

    def _selected_dataset(self) -> dict[str, Any] | None:
        selected = self.dataset_table.selectionModel().selectedRows() if self.dataset_table.selectionModel() else []
        return self._dataset_rows[selected[0].row()] if selected and selected[0].row() < len(self._dataset_rows) else None

    def _dataset_selection_changed(self) -> None:
        sample = self._selected_dataset()
        if sample:
            self.dataset_preview.setVisible(True)
            self.dataset_preview.set_landmarks(sample.get("landmarks", []))

    def relabel_selected(self) -> None:
        sample, label = self._selected_dataset(), self.relabel_combo.currentData()
        if not sample or not label:
            self.statusBar().showMessage("Selecione uma amostra e um rótulo.", 4000)
            return
        if self._call("relabel_sample", sample["id"], str(label), default=False) is False:
            return
        self.refresh_all()
        self.statusBar().showMessage("Rótulo atualizado.", 4000)

    def delete_selected(self) -> None:
        sample = self._selected_dataset()
        if not sample:
            self.statusBar().showMessage("Selecione uma amostra primeiro.", 4000)
            return
        answer = QMessageBox.question(
            self,
            "Excluir amostra",
            "Excluir esta amostra? Esta ação não altera experimentos já registrados.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Discard,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Discard:
            if self._call("delete_samples", [sample["id"]], default=False) is False:
                return
            self.refresh_all()
            self.statusBar().showMessage("Amostra excluída.", 4000)

    def validate_data(self) -> None:
        report = self._call("validate", default={}) or {}
        self.dataset_feedback.setText("Validação: " + ", ".join(f"{key}: {value}" for key, value in report.items()))

    def snapshot_data(self) -> None:
        snapshot_id = self._call("snapshot", default=None)
        if snapshot_id:
            self.dataset_feedback.setText(f"Versão imutável criada: {snapshot_id}")
            self.refresh_train_options()

    def export_data(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar dados numéricos", "gesturelab-dataset.parquet", "Parquet (*.parquet);;CSV (*.csv)"
        )
        if path:
            if self._call("export", Path(path), default=False) is not False:
                self.dataset_feedback.setText(f"Dados exportados para {path}.")

    # ----- camera and capture --------------------------------------------------
    def start_camera(self, target: str) -> None:
        if self.camera_worker:
            return
        try:
            camera = _module("camera")
            self.camera_worker = camera.CameraWorker(
                camera.default_model_path(),
                camera_index=self.collect_camera_index.value(),
                min_confidence=self.collect_min_confidence.value(),
                min_size=self.collect_min_size.value(),
            )
            self.camera_target = target
            self.live_session_id = None
            self._reset_live_state()
            self.camera_worker.frame_ready.connect(self.handle_camera_frame)
            self.camera_worker.error.connect(self.handle_camera_error)
            self.camera_worker.finished.connect(self._camera_finished)
            self.camera_worker.start()
            self.statusBar().showMessage("Abrindo câmera…", 3000)
            if target == "collect":
                self.collect_detection.setText("Abrindo câmera…")
                self._collection_gesture_changed()
        except Exception as exc:
            self.camera_worker = None
            self.camera_target = None
            self.handle_camera_error(f"Não foi possível preparar a câmera: {exc}")

    def stop_camera(self) -> None:
        self.pause_collection()
        worker = self.camera_worker
        if not worker:
            return
        try:
            worker.stop()
        except Exception:
            pass
        self.statusBar().showMessage("Encerrando captura e liberando a câmera…")
        if not worker.isRunning():
            self._camera_finished()

    def _camera_finished(self) -> None:
        worker = self.camera_worker
        if worker is not None and worker.isRunning():
            return
        self.camera_worker = None
        if worker is not None:
            worker.deleteLater()
        self.camera_target = None
        self.latest_frame = None
        self.collect_preview.set_image(None)
        self.live_preview.set_image(None)
        self.collect_detection.setText("Câmera fechada")
        self._sync_collection_controls()
        self._reset_live_state()
        self.live_prediction.setText("Câmera fechada")
        self.refresh_home()

    def handle_camera_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 9000)
        self.collect_state.setText(message)
        self.live_prediction.setText("Câmera indisponível")

    def handle_camera_frame(self, packet: dict[str, Any]) -> None:
        self.latest_frame = packet
        image = packet.get("image")
        if self.camera_target == "collect":
            self.collect_preview.set_image(image)
            state = packet.get("state", "invalid")
            self.collect_detection.setText(
                {
                    "ready": "Mão pronta",
                    "no_hand": "Nenhuma mão",
                    "multiple_hands": "Mais de uma mão",
                    "invalid": "Leitura inválida",
                }.get(state, state)
            )
            if self.collecting:
                if state == "ready":
                    self.collect_state.setText(f"Coletando: {self.collect_label.currentText()} • veja o contador de amostras.")
                else:
                    reason = (packet.get("quality") or {}).get("reason") or self.collect_detection.text()
                    self.collect_state.setText(f"Coleta em andamento • sem salvar: {reason}.")
            quality = dict(packet.get("quality") or {})
            hand_score = quality.pop("handedness_score", None) if isinstance(quality, dict) else None
            quality_text = (
                ", ".join(f"{key}: {value}" for key, value in quality.items())
                if quality
                else "sem detalhes disponíveis"
            )
            side_text = (
                f" · lateralidade {packet.get('handedness', '—')} ({float(hand_score):.0%})"
                if hand_score is not None
                else f" · lateralidade {packet.get('handedness', '—')}"
            )
            self.collect_quality.setText(
                "Qualidade geométrica: " + quality_text + side_text + f" · {packet.get('fps', 0):.1f} FPS"
            )
        elif self.camera_target == "live":
            self.live_preview.set_image(image)
            self.handle_live_prediction(packet)

    def _sync_collection_controls(self) -> None:
        busy = self.collecting or self.countdown_timer.isActive()
        can_start = self.camera_worker is not None and self.camera_target == "collect" and bool(self.collect_label.currentData())
        self.collect_start.setEnabled(can_start and not busy)
        self.collect_start.setText("Retomar contagem (3 s)" if self.collect_session_id else "Iniciar contagem (3 s)")
        self.collect_start.setToolTip("Abra a câmera em Coleta e escolha o gesto antes de iniciar.")
        self.collect_pause.setEnabled(busy)
        self.collect_finish.setEnabled(self.collect_session_id is not None)

    def _collection_gesture_changed(self, *_args) -> None:
        if self.collecting or self.countdown_timer.isActive():
            self.pause_collection()
        if not self.collect_label.currentData():
            self.collect_state.setText("Escolha uma categoria no campo Gesto para habilitar a contagem.")
        elif self.camera_worker is None or self.camera_target != "collect":
            self.collect_state.setText("Abra a câmera nesta tela para iniciar a coleta.")
        elif self.collect_session_id:
            self.collect_state.setText("Coleta pausada. Gesto selecionado; clique em Retomar contagem (3 s).")
        else:
            self.collect_state.setText(f"Gesto: {self.collect_label.currentText()}. Clique em Iniciar contagem (3 s).")
        self._sync_collection_controls()

    def begin_countdown(self) -> None:
        if self.collecting or self.countdown_timer.isActive():
            return
        if not self.camera_worker or self.camera_target != "collect":
            self.collect_state.setText("Abra a câmera em Coleta antes de iniciar a contagem.")
            return
        if not self.collect_label.currentData():
            self.collect_state.setText("Escolha uma categoria no campo Gesto antes de iniciar.")
            return
        if not self.collect_session_id:
            self.collect_session_id = self._call(
                "create_session",
                self.collect_participant.text().strip(),
                str(self.collect_purpose.currentData()),
                default=None,
            )
            self.collect_count = 0
            self.last_saved_timestamp_ms = None
            self.collect_counter.setText("0 amostras nesta sessão")
        if not self.collect_session_id:
            self.collect_state.setText("Não foi possível criar a sessão. Confira o aviso na barra inferior.")
            return
        self.countdown_remaining = 3
        self.collect_participant.setEnabled(False)
        self.collect_purpose.setEnabled(False)
        self.collect_start.setEnabled(False)
        self.collect_state.setText("Prepare a mão: 3")
        self.countdown_timer.start()
        self._sync_collection_controls()

    def _countdown_tick(self) -> None:
        self.countdown_remaining -= 1
        if self.countdown_remaining <= 0:
            self.countdown_timer.stop()
            self.collecting = True
            self.collect_state.setText("Coletando: mantenha o gesto estável")
            self.sample_timer.start(max(100, int(1000 / self.collect_rate.value())))
        else:
            self.collect_state.setText(f"Prepare a mão: {self.countdown_remaining}")
        self._sync_collection_controls()

    def record_current_sample(self) -> None:
        frame = self.latest_frame or {}
        if (
            not self.collecting
            or frame.get("state") != "ready"
            or frame.get("landmarks") is None
            or not self.collect_session_id
        ):
            return
        timestamp_ms = int(frame.get("timestamp_ms", 0) or 0)
        now_ms = int(time.monotonic() * 1000)
        if timestamp_ms and now_ms - timestamp_ms > 1500:
            self.collect_state.setText("Aguardando uma imagem recente da câmera")
            return
        if timestamp_ms and self.last_saved_timestamp_ms is not None and timestamp_ms <= self.last_saved_timestamp_ms:
            return
        result = self._call(
            "add_sample",
            self.collect_session_id,
            str(self.collect_label.currentData()),
            frame["landmarks"],
            handedness=frame.get("handedness", "Right"),
            quality=frame.get("quality"),
            extractor_version=frame.get("extractor_version", "mediapipe-hand-v1"),
            default=None,
        )
        if result:
            self.last_saved_timestamp_ms = timestamp_ms or now_ms
            self.collect_count += 1
            self.collect_counter.setText(f"{self.collect_count} amostras nesta sessão")
        else:
            self.pause_collection()
            self.collect_state.setText("Falha ao salvar a amostra. Coleta pausada; confira o aviso na barra inferior.")

    def pause_collection(self) -> None:
        self.collecting = False
        self.countdown_timer.stop()
        self.sample_timer.stop()
        if self.collect_session_id:
            self.collect_state.setText("Coleta pausada. A câmera continua na prévia; nenhuma amostra está sendo salva.")
        self._sync_collection_controls()

    def finish_collection(self) -> None:
        if self.collect_session_id is None:
            return
        total = self.collect_count
        self.pause_collection()
        self.collect_session_id = None
        self.collect_participant.setEnabled(True)
        self.collect_purpose.setEnabled(True)
        self.last_saved_timestamp_ms = None
        self.collect_counter.setText(f"{total} amostras salvas na sessão finalizada")
        self.collect_state.setText(f"Sessão finalizada: {total} amostras salvas. Revise no Dataset ou inicie uma nova sessão.")
        self.refresh_all()

    # ----- training ------------------------------------------------------------
    def start_training(self) -> None:
        if self.train_worker is not None and self.train_worker.isRunning():
            return
        dataset_id = self.train_dataset.currentData()
        if not dataset_id:
            self.statusBar().showMessage("Crie e selecione uma versão imutável do dataset.", 5000)
            return
        models = [name for name, check in self.train_model_checks.items() if check.isChecked()]
        if not models:
            self.statusBar().showMessage("Selecione ao menos um modelo para treinar.", 5000)
            return
        self.train_worker = TrainingWorker(
            self.store,
            str(dataset_id),
            self.train_strategy.currentData(),
            models,
            self.train_seed.value(),
            self.train_protocol.currentData(),
            str(self.train_preset.currentData()),
            self,
        )
        self.train_worker.progress.connect(self._train_message)
        self.train_worker.completed.connect(self._training_done)
        self.train_worker.failed.connect(self._training_failed)
        self.train_worker.cancelled.connect(self._training_cancelled)
        self.train_worker.finished.connect(self._training_thread_finished)
        self.train_start.setEnabled(False)
        self.train_cancel.setEnabled(True)
        self.train_progress.setRange(0, 0)
        self.train_log.clear()
        self.train_log.append("Treinamento iniciado em segundo plano…")
        self.train_worker.start()

    def _train_message(self, message: str) -> None:
        self.train_log.append(message)

    def cancel_training(self) -> None:
        if self.train_worker:
            self.train_worker.cancel()
            self.train_cancel.setEnabled(False)
            self.train_log.append("Cancelamento solicitado; aguardando ponto seguro.")

    def _training_done(self, ids: list[str]) -> None:
        if ids:
            self.train_log.append("Modelos concluídos: " + ", ".join(ids))
        else:
            self.train_log.append(
                "Nenhum modelo foi concluído. Veja os experimentos marcados como falhos ou cancelados."
            )
        recent = (self._call("experiments", default=[]) or [])[: max(1, len(self.train_model_checks))]
        statuses = [f"{_value(item, 'model')}: {_value(item, 'status')}" for item in recent]
        if statuses:
            self.train_log.append("Estados registrados: " + " · ".join(statuses))
        self._end_training()
        self.refresh_all()

    def _training_failed(self, message: str) -> None:
        self.train_log.append("Falhou: " + message)
        self._end_training()
        self.refresh_all()

    def _training_cancelled(self) -> None:
        self.train_log.append("Cancelado. Artefatos incompletos não são modelos válidos.")
        self._end_training()
        self.refresh_all()

    def _end_training(self) -> None:
        self.train_progress.setRange(0, 1)
        self.train_progress.setValue(1)
        self.train_cancel.setEnabled(False)
        self.refresh_train_options()

    def _training_thread_finished(self) -> None:
        worker = self.train_worker
        self.train_worker = None
        if worker is not None:
            worker.deleteLater()
        self.refresh_train_options()

    # ----- experiments ---------------------------------------------------------
    def _selected_experiment(self) -> dict[str, Any] | None:
        selected = self._selected_experiments()
        return selected[0] if len(selected) == 1 else None

    def activate_selected_experiment(self) -> None:
        exp = self._selected_experiment()
        if not exp:
            self.statusBar().showMessage("Selecione um experimento primeiro.", 4000)
            return
        if _value(exp, "status") != "completed":
            self.statusBar().showMessage("Somente experimentos concluídos podem ser ativados.", 5000)
            return
        if self._call("activate", _value(exp, "id"), default=False) is False:
            return
        self.refresh_all()
        self.statusBar().showMessage("Modelo ativo atualizado.", 4000)

    def evaluate_selected_final(self) -> None:
        exp = self._selected_experiment()
        if not exp:
            return
        try:
            report = _module("ml").evaluate_final(self.store, _value(exp, "id"))
            self.refresh_experiments()
            self.show_selected_experiment()
            self.experiment_details.setText(
                "Teste final reservado avaliado. "
                + ", ".join(
                    f"{key}: {value}"
                    for key, value in (report or {}).items()
                    if key not in {"per_class", "confusion_matrix"}
                )
            )
        except Exception as exc:
            self.experiment_details.setText(f"Não foi possível avaliar o teste final: {exc}")

    # ----- live and difficult samples -----------------------------------------
    def _reset_live_state(self) -> None:
        self.last_prediction = None
        if self.temporal_stabilizer is not None:
            self.temporal_stabilizer.reset()
        if self.action_gate is not None:
            self.action_gate.update(None)
        self.live_score.setText("Pontuação: —")
        self.live_action.setText("Demonstração: neutra (rearmada)")

    def handle_live_prediction(self, packet: dict[str, Any]) -> None:
        state = packet.get("state")
        self.live_performance.setText(
            f"FPS: {packet.get('fps', 0):.1f} · latência: {packet.get('latency_ms', 0):.1f} ms"
        )
        if state != "ready" or packet.get("landmarks") is None:
            self._reset_live_state()
            labels = {"no_hand": "Sem mão", "multiple_hands": "Mais de uma mão", "invalid": "Leitura inválida"}
            self.live_prediction.setText(labels.get(state, "Aguardando leitura"))
            return
        active = self._call("active", default=None)
        if not active:
            self._reset_live_state()
            self.live_prediction.setText("Modelo indisponível")
            return
        try:
            if self.live_bundle is None or self.live_bundle.get("id") != active:
                bundle = _module("ml").load_model(self.store, active)
                if isinstance(bundle, dict):
                    bundle["id"] = active
                self.live_bundle = bundle
                camera = _module("camera")
                self.action_gate = camera.ActionGate()
                self.temporal_stabilizer = camera.TemporalStabilizer(
                    window=self.live_window.value(), consensus=self.live_consensus.value()
                )
            result = _module("ml").predict(
                self.live_bundle, packet["landmarks"], handedness=packet.get("handedness", "Right")
            )
            self.last_prediction = dict(result)
            if self.temporal_stabilizer and (self.temporal_stabilizer.window != self.live_window.value()
                or self.temporal_stabilizer.consensus != self.live_consensus.value()):
                self.temporal_stabilizer = _module("camera").TemporalStabilizer(
                    window=self.live_window.value(), consensus=self.live_consensus.value())
            label, score, accepted = (
                result.get("label", "—"),
                float(result.get("score", 0)),
                bool(result.get("accepted")),
            )
            stable = (
                self.temporal_stabilizer.update(label if accepted else None)
                if self.temporal_stabilizer
                else (label if accepted else None)
            )
            self.live_prediction.setText(stable or ("Estabilizando…" if accepted else "Previsão incerta"))
            self.live_score.setText(f"Previsão individual: {label} · pontuação: {score:.1%}" + (" · aceita" if accepted else " · abaixo do limiar"))
            if self.temporal_stabilizer and stable:
                self.live_performance.setText(
                    f"FPS: {packet.get('fps', 0):.1f} · extração: {packet.get('latency_ms', 0):.1f} ms · "
                    f"classificação: {float(result.get('latency_ms', 0)):.1f} ms · suavização: {self.temporal_stabilizer.response_delay_ms:.0f} ms"
                )
            else:
                self.live_performance.setText(
                    f"FPS: {packet.get('fps', 0):.1f} · extração: {packet.get('latency_ms', 0):.1f} ms · "
                    f"classificação: {float(result.get('latency_ms', 0)):.1f} ms"
                )
            action = self.action_gate.update(stable) if self.action_gate else None
            if action:
                self.live_action.setText(f"Demonstração: ação “{action}”")
                dots = ["○"] * 5
                dots[(sum(map(ord, action)) % 5)] = "●"
                self.slide_demo.setText("  ".join(dots))
            elif not accepted:
                self.live_action.setText("Demonstração: neutra (rearmada)")
        except Exception as exc:
            self._reset_live_state()
            self.live_prediction.setText(f"Modelo incompatível: {exc}")

    def mark_live_difficult(self) -> None:
        """Guarda um exemplo de desafio para revisão humana, sem dar-lhe um rótulo."""
        frame, result = self.latest_frame or {}, self.last_prediction or {}
        if self.camera_target != "live" or frame.get("state") != "ready" or frame.get("landmarks") is None:
            self.statusBar().showMessage("Espere uma leitura válida da mão antes de marcar o exemplo.", 5000)
            return
        if not self.live_session_id:
            self.live_session_id = self._call("create_session", "", "development", default=None)
        if not self.live_session_id:
            return
        sample_id = self._call(
            "add_difficult",
            self.live_session_id,
            frame["landmarks"],
            frame.get("handedness", "Right"),
            predicted_label=str(result.get("label", "")),
            score=float(result.get("score", 0.0)),
            extractor_version=frame.get("extractor_version", "mediapipe-hand-v1"),
            default=None,
        )
        if sample_id:
            self.statusBar().showMessage("Exemplo salvo para revisão. Informe o rótulo verdadeiro em Revisão.", 6000)
            self.refresh_difficult()

    def apply_difficult_review(self) -> None:
        selected = self.review_table.selectionModel().selectedRows() if self.review_table.selectionModel() else []
        label = self.review_label.currentData()
        if not selected or not label:
            self.statusBar().showMessage("Selecione um exemplo e informe seu rótulo verdadeiro.", 5000)
            return
        item = self._difficult_rows[selected[0].row()]
        if self._call("review_difficult", _value(item, "id"), str(label), default=False) is False:
            return
        self.refresh_all()
        self.statusBar().showMessage("Exemplo revisado. Ele poderá entrar em uma nova versão do dataset.", 5000)

    def _review_selection_changed(self) -> None:
        selected = self.review_table.selectionModel().selectedRows()
        if selected and selected[0].row() < len(self._difficult_rows):
            self.review_preview.set_landmarks(self._difficult_rows[selected[0].row()].get("landmarks"))
        else:
            self.review_preview.set_landmarks(None)

    def discard_difficult(self) -> None:
        selected = self.review_table.selectionModel().selectedRows() if self.review_table.selectionModel() else []
        if not selected:
            return
        item = self._difficult_rows[selected[0].row()]
        answer = QMessageBox.question(
            self,
            "Descartar exemplo",
            "Descartar este exemplo difícil?",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Discard,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Discard:
            if self._call("discard_difficult", _value(item, "id"), default=False) is False:
                return
            self.refresh_all()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.pause_collection()
        worker = self.camera_worker
        if worker and worker.isRunning():
            try:
                worker.stop()
            except Exception:
                pass
        if self.train_worker and self.train_worker.isRunning():
            self.train_worker.cancel()
        camera_busy = bool(worker and worker.isRunning())
        training_busy = bool(self.train_worker and self.train_worker.isRunning())
        if camera_busy or training_busy:
            self._close_requested = True
            event.ignore()
            QTimer.singleShot(100, self._finish_close_when_workers_stop)
            return
        event.accept()

    def _finish_close_when_workers_stop(self) -> None:
        """A janela fica viva até os workers liberarem recursos, sem bloqueio da UI."""
        camera_busy = bool(self.camera_worker and self.camera_worker.isRunning())
        training_busy = bool(self.train_worker and self.train_worker.isRunning())
        if camera_busy or training_busy:
            QTimer.singleShot(100, self._finish_close_when_workers_stop)
            return
        self.stop_camera()
        self.close()


def run(data_dir: Path | str | None = None) -> int:
    """Abre a interface sem inicializar a webcam automaticamente."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    store_module = _module("store")
    window = MainWindow(store_module.Store(Path(data_dir) if data_dir else Path("data")))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
