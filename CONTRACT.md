# Contrato interno de integração

Dados locais em `data/` por padrão, configuráveis com `--data-dir`.

## Núcleo (store.py, features.py, ml.py, cli.py)

`Store(root: Path | str)` cria metadados SQLite locais e diretórios.
- `categories() -> list[str]`, `add_category(name)`
- `create_session(participant: str = '', purpose: str = 'development') -> str` (purpose development/challenge)
- `sessions() -> list[dict]`
- `add_sample(session_id, label, landmarks, handedness='Right', quality=None, extractor_version='mediapipe-hand-v1') -> str`; landmarks np.array (21,3). Rejeita inválidos. label existente.
- `samples(session_id=None) -> pd.DataFrame`: id, session_id, participant, label, timestamp, handedness, quality, extractor_version, purpose, landmarks (lista plana de 63 floats).
- `delete_samples(ids)`, `relabel_sample(id, label)`
- `validate() -> dict`, `snapshot() -> str`, `snapshots() -> list[dict]`, `load_snapshot(id) -> pd.DataFrame`, `export(path)`
- `add_difficult(session_id, landmarks, handedness, predicted_label='', score=0.0) -> str`, `difficult() -> list[dict]`, `review_difficult(id, label)`, `discard_difficult(id)`
- `experiments() -> list[dict]`, `experiment(id) -> dict`, `activate(id)`, `active() -> str | None`

`train(store, dataset_id, strategy='normalized', models=None, seed=42, protocol='session', preset='quick', cancel=None, progress=None) -> list[str]` em ml.py. models dummy/logistic/forest/mlp; strategy raw/normalized/distances/angles; cancel threading.Event; progress callback(str). status completed/failed/cancelled; usar split persistido grupos e teste reservado. Experimentos com id, model, status, dataset_id, strategy, protocol, seed, metrics (accuracy, macro_f1), duration_s, threshold, parameters. `load_model(store, experiment_id) -> bundle` dict com pipeline, threshold, classes; `predict(bundle, landmarks, handedness='Right') -> dict` label, score, accepted, latency_ms. `evaluate_final(store, experiment_id)` teste final único por linhagem/reserva; `compare(store, ids) -> dict`. CLI entry `main()`.

## Captura (camera.py) — responsabilidade principal

`CameraWorker(QThread)` construtor `(model_path: Path, camera_index=0, width=640, height=480, min_confidence=0.5, min_size=0.06)`.
Signals: `frame_ready(object)` dict: image (QImage RGB com desenho e espelhada só na prévia), landmarks (np.ndarray(21,3) ou None), handedness ('Left'/'Right'), quality (dict), state ('ready'/'no_hand'/'multiple_hands'/'invalid'), fps, latency_ms, extractor_version. `error(str)`. `stop()` pede parada sem bloquear; finished herdado. Não abre câmera automaticamente.
`TemporalStabilizer(window=5, consensus=0.7)` `update(label: str | None, now=None) -> str | None`; reset(). `ActionGate(cooldown=1.5)` `update(label: str | None, now=None) -> str | None`; neutro rearma.
`default_model_path() -> Path` aponta assets/hand_landmarker.task.

## Interface (ui.py) — agente UI

`MainWindow(store)` e `run(data_dir=None)`; importar módulos acima. Worker treino próprio chama train. Nenhum trabalho de captura/treino na thread principal. Inicialização sem webcam. Interface pt-BR com navegação Início, Coleta, Dataset, Treinamento, Experimentos, Ao vivo, Revisão, Aprender. Pode separar ui_widgets.py. Integração usa estritamente contratos acima e sinaliza mudanças.
