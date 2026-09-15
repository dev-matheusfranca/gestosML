"""Persistência local do GestureLab.

SQLite é a fonte de metadados; shards Parquet guardam as amostras numéricas.
Snapshots são cópias Parquet com hash, logo edição e remoção no
dataset atual não reescrevem o histórico experimental.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import numpy as np

from .features import as_landmarks


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Store:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.root / "snapshots"
        self.artifacts_dir = self.root / "artifacts"
        self.snapshots_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)
        self.db_path = self.root / "gesturelab.sqlite3"
        self._setup()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path)
        try:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys=ON")
            with con:
                yield con
        finally:
            con.close()

    def _setup(self) -> None:
        with self._connect() as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS categories (name TEXT PRIMARY KEY, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, participant TEXT NOT NULL, purpose TEXT NOT NULL CHECK(purpose IN ('development','challenge')), created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS samples (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), label TEXT NOT NULL REFERENCES categories(name), timestamp TEXT NOT NULL, handedness TEXT NOT NULL, quality_json TEXT, extractor_version TEXT NOT NULL, landmarks_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS snapshots (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL, rows INTEGER NOT NULL, lineage_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS difficult (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), landmarks_json TEXT NOT NULL, handedness TEXT NOT NULL, predicted_label TEXT NOT NULL, score REAL NOT NULL, created_at TEXT NOT NULL, extractor_version TEXT NOT NULL DEFAULT 'difficult-review-v1', status TEXT NOT NULL CHECK(status IN ('pending','reviewed','discarded')), reviewed_label TEXT, reviewed_at TEXT, incorporated_sample_id TEXT);
            CREATE TABLE IF NOT EXISTS splits (lineage_id TEXT NOT NULL, protocol TEXT NOT NULL, group_id TEXT NOT NULL, partition TEXT NOT NULL CHECK(partition IN ('train','validation','test')), created_at TEXT NOT NULL, PRIMARY KEY(lineage_id, protocol, group_id));
            CREATE TABLE IF NOT EXISTS split_samples (lineage_id TEXT NOT NULL, protocol TEXT NOT NULL, sample_id TEXT NOT NULL, partition TEXT NOT NULL CHECK(partition IN ('train','validation','test')), fingerprint TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(lineage_id, protocol, sample_id));
            CREATE TABLE IF NOT EXISTS final_evaluations (lineage_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, report_json TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, model TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('completed','failed','cancelled','running')), dataset_id TEXT NOT NULL REFERENCES snapshots(id), lineage_id TEXT NOT NULL, strategy TEXT NOT NULL, protocol TEXT NOT NULL, seed INTEGER NOT NULL, parameters_json TEXT NOT NULL, metrics_json TEXT NOT NULL, threshold REAL, duration_s REAL, artifact_path TEXT, artifact_sha256 TEXT, environment_json TEXT NOT NULL, created_at TEXT NOT NULL, completed_at TEXT, error TEXT);
            """)
            if con.execute("SELECT 1 FROM settings WHERE key='lineage_id'").fetchone() is None:
                con.execute("INSERT INTO settings(key,value) VALUES('lineage_id',?)", (_id("lineage"),))
            # Categorias iniciais só são semeadas quando ausentes; abrir o
            # projeto nunca substitui escolhas da pessoa usuária.
            for category in ("Palma aberta", "Punho fechado", "Sinal de positivo", "Sinal de V", "Indicador apontando"):
                con.execute("INSERT OR IGNORE INTO categories(name,created_at) VALUES(?,?)", (category, _now()))
            difficult_columns = {row[1] for row in con.execute("PRAGMA table_info(difficult)")}
            if "incorporated_sample_id" not in difficult_columns:
                con.execute("ALTER TABLE difficult ADD COLUMN incorporated_sample_id TEXT")
            if "extractor_version" not in difficult_columns:
                con.execute("ALTER TABLE difficult ADD COLUMN extractor_version TEXT NOT NULL DEFAULT 'difficult-review-v1'")

    def _lineage(self) -> str:
        with self._connect() as con:
            return str(con.execute("SELECT value FROM settings WHERE key='lineage_id'").fetchone()[0])

    def categories(self) -> list[str]:
        with self._connect() as con:
            return [str(row[0]) for row in con.execute("SELECT name FROM categories ORDER BY name")]

    def add_category(self, name: str) -> None:
        name = str(name).strip()
        if not name or len(name) > 80:
            raise ValueError("O nome da categoria deve ter entre 1 e 80 caracteres.")
        with self._connect() as con:
            con.execute("INSERT OR IGNORE INTO categories(name,created_at) VALUES(?,?)", (name, _now()))

    def create_session(self, participant: str = "", purpose: str = "development") -> str:
        if purpose not in {"development", "challenge"}:
            raise ValueError("purpose deve ser development ou challenge.")
        ident = _id("session")
        with self._connect() as con:
            con.execute("INSERT INTO sessions(id,participant,purpose,created_at) VALUES(?,?,?,?)", (ident, str(participant).strip(), purpose, _now()))
        return ident

    def sessions(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("""SELECT s.id,s.participant,s.purpose,s.created_at,COUNT(a.id) AS sample_count
                FROM sessions s LEFT JOIN samples a ON a.session_id=s.id GROUP BY s.id ORDER BY s.created_at DESC""").fetchall()
        return [dict(row) for row in rows]

    def _write_current_parquet(self) -> None:
        frame = self.samples()
        target = self.root / "samples.parquet"
        self._atomic_parquet(frame, target)

    @staticmethod
    def _atomic_parquet(frame: pd.DataFrame, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Arrow não consegue inferir uma struct sem campos de ``{}``. Qualidade
        # é um metadado flexível; no Parquet ela é JSON, enquanto samples()
        # mantém o dict previsto pela API.
        persisted = frame.copy()
        if "quality" in persisted:
            persisted["quality"] = persisted["quality"].map(lambda value: json.dumps(value or {}, sort_keys=True))
        temp = target.parent / f".writing-{uuid.uuid4().hex}.parquet"
        try:
            persisted.to_parquet(temp, index=False)
            os.replace(temp, target)
        finally:
            try:
                temp.unlink(missing_ok=True)
            except PermissionError:
                # Um erro de Arrow pode soltar o handle após esta chamada no
                # Windows. O arquivo é temporário e nunca substitui o destino.
                pass

    def _write_sample_shard(self, ident: str, points: np.ndarray) -> None:
        """Um shard numérico por coleta evita reescrever O(n) exemplos a cada frame."""
        target = self.root / "sample_shards" / f"{ident}.parquet"
        numeric = pd.DataFrame([{ "id": ident, **{f"landmark_{index}": float(value) for index, value in enumerate(points.reshape(-1))} }])
        self._atomic_parquet(numeric, target)

    def add_sample(self, session_id: str, label: str, landmarks: object, handedness: str = "Right", quality: dict | None = None, extractor_version: str = "mediapipe-hand-v1") -> str:
        points = as_landmarks(landmarks)
        # Mesmo uma estratégia raw não deve aceitar a mão degenerada: ela não
        # representa uma geometria extraída utilizável.
        from .features import canonical_landmarks
        canonical_landmarks(points, handedness)
        if handedness not in {"Left", "Right"}:
            raise ValueError("handedness deve ser Left ou Right.")
        ident = _id("sample")
        with self._connect() as con:
            if con.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is None:
                raise ValueError("Sessão inexistente.")
            if con.execute("SELECT 1 FROM categories WHERE name=?", (label,)).fetchone() is None:
                raise ValueError("A categoria precisa existir antes da coleta.")
            con.execute("INSERT INTO samples(id,session_id,label,timestamp,handedness,quality_json,extractor_version,landmarks_json) VALUES(?,?,?,?,?,?,?,?)", (ident, session_id, label, _now(), handedness, json.dumps(quality or {}, sort_keys=True), extractor_version, json.dumps(points.reshape(-1).tolist()),))
        self._write_sample_shard(ident, points)
        return ident

    def samples(self, session_id: str | None = None) -> pd.DataFrame:
        query = """SELECT a.id,a.session_id,s.participant,a.label,a.timestamp,a.handedness,a.quality_json,a.extractor_version,s.purpose,a.landmarks_json
                   FROM samples a JOIN sessions s ON s.id=a.session_id"""
        params: tuple = ()
        if session_id is not None:
            query += " WHERE a.session_id=?"
            params = (session_id,)
        query += " ORDER BY a.timestamp,a.id"
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["quality"] = json.loads(item.pop("quality_json") or "{}")
            item["landmarks"] = json.loads(item.pop("landmarks_json"))
            result.append(item)
        return pd.DataFrame(result, columns=["id", "session_id", "participant", "label", "timestamp", "handedness", "quality", "extractor_version", "purpose", "landmarks"])

    def delete_samples(self, ids: Iterable[str]) -> int:
        identifiers = [str(value) for value in ids]
        if not identifiers:
            return 0
        with self._connect() as con:
            existing = [ident for ident in identifiers if con.execute("SELECT 1 FROM samples WHERE id=?", (ident,)).fetchone()]
            cursor = con.executemany("DELETE FROM samples WHERE id=?", [(value,) for value in identifiers])
            count = cursor.rowcount
        for ident in existing:
            if Path(ident).name != ident:
                continue
            (self.root / "sample_shards" / f"{ident}.parquet").unlink(missing_ok=True)
        return max(0, count)

    def relabel_sample(self, ident: str, label: str) -> None:
        with self._connect() as con:
            if con.execute("SELECT 1 FROM categories WHERE name=?", (label,)).fetchone() is None:
                raise ValueError("A nova categoria não existe.")
            if con.execute("UPDATE samples SET label=? WHERE id=?", (label, ident)).rowcount != 1:
                raise ValueError("Amostra inexistente.")
        # Shards contêm somente os 63 números; rótulo é metadado no SQLite.

    def validate(self) -> dict:
        frame = self.samples()
        invalid: list[str] = []
        for row in frame.itertuples(index=False):
            try:
                from .features import canonical_landmarks
                canonical_landmarks(row.landmarks, row.handedness)
                if row.handedness not in {"Left", "Right"}:
                    raise ValueError("Lateralidade inválida")
            except (TypeError, ValueError):
                invalid.append(row.id)
        if frame.empty:
            duplicate_count = 0
        else:
            fingerprints = frame.assign(_landmarks_key=frame["landmarks"].map(lambda value: json.dumps(value, separators=(",", ":"))))
            duplicate_count = int(fingerprints.duplicated(subset=["handedness", "_landmarks_key"], keep=False).sum())
        return {"valid": not invalid, "samples": len(frame), "invalid_ids": invalid, "exact_duplicate_rows": duplicate_count, "categories": self.categories()}

    def snapshot(self) -> str:
        report = self.validate()
        if not report["valid"]:
            raise ValueError("Há amostras inválidas; corrija-as antes do snapshot.")
        frame = self.samples()
        if frame.empty:
            raise ValueError("Não é possível versionar um dataset vazio.")
        ident = _id("dataset")
        target = self.snapshots_dir / f"{ident}.parquet"
        self._atomic_parquet(frame, target)
        digest = _sha256(target)
        with self._connect() as con:
            con.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?)", (ident, _now(), target.name, digest, len(frame), self._lineage()))
        return ident

    def snapshots(self) -> list[dict]:
        with self._connect() as con:
            return [dict(row) for row in con.execute("SELECT id,created_at,sha256,rows,lineage_id FROM snapshots ORDER BY created_at DESC")]

    def snapshot_info(self, ident: str) -> dict:
        with self._connect() as con:
            row = con.execute("SELECT * FROM snapshots WHERE id=?", (ident,)).fetchone()
        if row is None:
            raise ValueError("Versão de dataset inexistente.")
        return dict(row)

    def load_snapshot(self, ident: str) -> pd.DataFrame:
        info = self.snapshot_info(ident)
        path = self.snapshots_dir / info["path"]
        if not path.is_file() or _sha256(path) != info["sha256"]:
            raise ValueError("O arquivo do snapshot está ausente ou teve sua integridade alterada.")
        return pd.read_parquet(path)

    def export(self, path: Path | str) -> Path:
        target = Path(path)
        frame = self.samples()
        if target.suffix.lower() == ".csv":
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(target, index=False)
        else:
            self._atomic_parquet(frame, target)
        return target

    def add_difficult(self, session_id: str, landmarks: object, handedness: str, predicted_label: str = "", score: float = 0.0, extractor_version: str = "difficult-review-v1") -> str:
        points = as_landmarks(landmarks)
        from .features import canonical_landmarks
        canonical_landmarks(points, handedness)
        if handedness not in {"Left", "Right"} or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Lateralidade ou pontuação inválida.")
        ident = _id("difficult")
        with self._connect() as con:
            if con.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is None:
                raise ValueError("Sessão inexistente.")
            con.execute("INSERT INTO difficult(id,session_id,landmarks_json,handedness,predicted_label,score,created_at,extractor_version,status,reviewed_label,reviewed_at,incorporated_sample_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (ident, session_id, json.dumps(points.reshape(-1).tolist()), handedness, str(predicted_label), float(score), _now(), extractor_version, "pending", None, None, None))
        return ident

    def difficult(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM difficult ORDER BY created_at DESC").fetchall()
        answer = []
        for row in rows:
            item = dict(row)
            item["landmarks"] = json.loads(item.pop("landmarks_json"))
            answer.append(item)
        return answer

    def review_difficult(self, ident: str, label: str) -> None:
        sample_id = _id("sample")
        with self._connect() as con:
            if con.execute("SELECT 1 FROM categories WHERE name=?", (label,)).fetchone() is None:
                raise ValueError("A categoria revisada não existe.")
            difficult = con.execute("SELECT session_id,landmarks_json,handedness,created_at,extractor_version FROM difficult WHERE id=? AND status='pending'", (ident,)).fetchone()
            if difficult is None:
                raise ValueError("Exemplo difícil inexistente ou já revisado.")
            points = as_landmarks(json.loads(difficult["landmarks_json"]))
            from .features import canonical_landmarks
            canonical_landmarks(points, difficult["handedness"])
            con.execute("INSERT INTO samples(id,session_id,label,timestamp,handedness,quality_json,extractor_version,landmarks_json) VALUES(?,?,?,?,?,?,?,?)", (sample_id, difficult["session_id"], label, difficult["created_at"], difficult["handedness"], json.dumps({"source": "difficult_review"}, sort_keys=True), difficult["extractor_version"], json.dumps(points.reshape(-1).tolist())))
            con.execute("UPDATE difficult SET status='reviewed',reviewed_label=?,reviewed_at=?,incorporated_sample_id=? WHERE id=?", (label, _now(), sample_id, ident))
        self._write_sample_shard(sample_id, points)

    def discard_difficult(self, ident: str) -> None:
        with self._connect() as con:
            if con.execute("UPDATE difficult SET status='discarded',reviewed_at=? WHERE id=? AND status='pending'", (_now(), ident)).rowcount != 1:
                raise ValueError("Exemplo difícil inexistente ou já tratado.")

    def reviewed_difficult_to_samples(self, session_id: str) -> int:
        """Compatibilidade legada: revisão já incorpora uma única vez, atomicamente."""
        return 0

    def split_rows(self, lineage_id: str, protocol: str) -> dict[str, str]:
        with self._connect() as con:
            rows = con.execute("SELECT group_id,partition FROM splits WHERE lineage_id=? AND protocol=?", (lineage_id, protocol)).fetchall()
        return {str(row["group_id"]): str(row["partition"]) for row in rows}

    def split_protocol(self, lineage_id: str) -> str | None:
        with self._connect() as con:
            row = con.execute("SELECT value FROM settings WHERE key=?", (f"split_protocol:{lineage_id}",)).fetchone()
        return None if row is None else str(row[0])

    def lock_split_protocol(self, lineage_id: str, protocol: str) -> None:
        key = f"split_protocol:{lineage_id}"
        with self._connect() as con:
            current = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            if current is not None and current[0] != protocol:
                raise ValueError("Esta linhagem já reservou o teste final com outro protocolo; crie uma nova Store/linhagem para trocar o protocolo.")
            con.execute("INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)", (key, protocol))

    def save_split_rows(self, lineage_id: str, protocol: str, assignments: dict[str, str]) -> None:
        with self._connect() as con:
            con.executemany("INSERT OR IGNORE INTO splits(lineage_id,protocol,group_id,partition,created_at) VALUES(?,?,?,?,?)", [(lineage_id, protocol, group, part, _now()) for group, part in assignments.items()])

    def split_sample_rows(self, lineage_id: str, protocol: str) -> dict[str, tuple[str, str]]:
        with self._connect() as con:
            rows = con.execute("SELECT sample_id,partition,fingerprint FROM split_samples WHERE lineage_id=? AND protocol=?", (lineage_id, protocol)).fetchall()
        return {str(row["sample_id"]): (str(row["partition"]), str(row["fingerprint"])) for row in rows}

    def save_split_samples(self, lineage_id: str, protocol: str, rows: Iterable[tuple[str, str, str]]) -> None:
        with self._connect() as con:
            con.executemany("INSERT OR IGNORE INTO split_samples(lineage_id,protocol,sample_id,partition,fingerprint,created_at) VALUES(?,?,?,?,?,?)", [(lineage_id, protocol, sample_id, partition, fingerprint, _now()) for sample_id, partition, fingerprint in rows])

    def claim_final_evaluation(self, lineage_id: str, experiment_id: str) -> dict | None:
        """Reserva transacionalmente a única consulta do teste final da linhagem."""
        with self._connect() as con:
            row = con.execute("SELECT experiment_id,report_json FROM final_evaluations WHERE lineage_id=?", (lineage_id,)).fetchone()
            if row is not None:
                if row["experiment_id"] != experiment_id:
                    raise ValueError("O teste final desta linhagem já foi consultado por outro experimento; crie uma nova coleta/linhagem para uma nova decisão.")
                report = json.loads(row["report_json"])
                if report:
                    return report
                raise ValueError("Uma avaliação final deste experimento já está em andamento.")
            con.execute("INSERT INTO final_evaluations(lineage_id,experiment_id,report_json,created_at) VALUES(?,?,?,?)", (lineage_id, experiment_id, "{}", _now()))
        return None

    def complete_final_evaluation(self, lineage_id: str, experiment_id: str, report: dict) -> None:
        with self._connect() as con:
            if con.execute("UPDATE final_evaluations SET report_json=? WHERE lineage_id=? AND experiment_id=?", (json.dumps(report, sort_keys=True), lineage_id, experiment_id)).rowcount != 1:
                raise RuntimeError("Reserva do teste final não pertence a este experimento.")

    def release_final_evaluation(self, lineage_id: str, experiment_id: str) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM final_evaluations WHERE lineage_id=? AND experiment_id=? AND report_json='{}'", (lineage_id, experiment_id))

    def create_experiment(self, **values: object) -> str:
        ident = _id("experiment")
        with self._connect() as con:
            con.execute("""INSERT INTO experiments(id,model,status,dataset_id,lineage_id,strategy,protocol,seed,parameters_json,metrics_json,threshold,duration_s,artifact_path,artifact_sha256,environment_json,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (ident, values["model"], "running", values["dataset_id"], values["lineage_id"], values["strategy"], values["protocol"], values["seed"], json.dumps(values.get("parameters", {}), sort_keys=True), json.dumps({}), None, None, None, None, json.dumps(values.get("environment", {}), sort_keys=True), _now()))
        return ident

    def finish_experiment(self, ident: str, status: str, **values: object) -> None:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("Estado final inválido.")
        with self._connect() as con:
            con.execute("UPDATE experiments SET status=?,parameters_json=COALESCE(?,parameters_json),metrics_json=?,threshold=?,duration_s=?,artifact_path=?,artifact_sha256=?,error=?,completed_at=? WHERE id=?", (status, json.dumps(values["parameters"], sort_keys=True) if "parameters" in values else None, json.dumps(values.get("metrics", {}), sort_keys=True), values.get("threshold"), values.get("duration_s"), values.get("artifact_path"), values.get("artifact_sha256"), values.get("error"), _now(), ident))

    def experiments(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
        return [self._experiment_dict(row) for row in rows]

    @staticmethod
    def _experiment_dict(row: sqlite3.Row) -> dict:
        item = dict(row)
        for field in ("parameters_json", "metrics_json", "environment_json"):
            item[field.removesuffix("_json")] = json.loads(item.pop(field))
        return item

    def experiment(self, ident: str) -> dict:
        with self._connect() as con:
            row = con.execute("SELECT * FROM experiments WHERE id=?", (ident,)).fetchone()
        if row is None:
            raise ValueError("Experimento inexistente.")
        return self._experiment_dict(row)

    def activate(self, ident: str) -> None:
        experiment = self.experiment(ident)
        if experiment["status"] != "completed" or not experiment["artifact_path"]:
            raise ValueError("Somente experimentos concluídos com artefato podem ser ativados.")
        with self._connect() as con:
            con.execute("INSERT INTO settings(key,value) VALUES('active_experiment',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (ident,))

    def active(self) -> str | None:
        with self._connect() as con:
            row = con.execute("SELECT value FROM settings WHERE key='active_experiment'").fetchone()
        return None if row is None else str(row[0])
