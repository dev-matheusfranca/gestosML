"""Transformações determinísticas dos 21 pontos de uma mão.

As funções deste módulo nunca alteram os pontos persistidos. A versão atual
canonicamente espelha a coordenada X de mãos esquerdas *nos atributos*, e não
na imagem de prévia. Isto reduz uma fonte simples de variação, mas não torna o
classificador invariante a rotação ou a oclusões.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

LANDMARK_COUNT = 21
COORDINATES = 3
FEATURE_VERSION = "hand-features-v1"


def as_landmarks(value: object) -> np.ndarray:
    """Converte uma amostra para ``(21, 3)`` e rejeita entradas imprecisas."""
    array = np.asarray(value, dtype=np.float64)
    if array.size != LANDMARK_COUNT * COORDINATES:
        raise ValueError("Uma amostra precisa conter exatamente 21 pontos (x, y, z).")
    array = array.reshape(LANDMARK_COUNT, COORDINATES)
    if not np.isfinite(array).all():
        raise ValueError("Os pontos da mão precisam ser finitos.")
    return array


def canonical_landmarks(value: object, handedness: str = "Right") -> np.ndarray:
    """Centraliza no punho, normaliza a escala e padroniza lateralidade.

    A escala é a distância punho--MCP do dedo médio (0--9), uma medida estável
    para a maioria das poses. Escala nula é um dado inválido, não um valor a ser
    silenciosamente corrigido.
    """
    points = as_landmarks(value).copy()
    if str(handedness).lower() == "left":
        points[:, 0] *= -1.0
    points -= points[0]
    scale = float(np.linalg.norm(points[9]))
    if not np.isfinite(scale) or scale <= 1e-9:
        raise ValueError("Escala da mão nula ou inválida.")
    return points / scale


_DISTANCE_PAIRS = ((0, 4), (0, 8), (0, 12), (0, 16), (0, 20), (4, 8), (8, 12), (12, 16), (16, 20))
_ANGLE_TRIPLES = ((0, 1, 2), (1, 2, 3), (2, 3, 4), (0, 5, 6), (5, 6, 7), (6, 7, 8),
                  (0, 9, 10), (9, 10, 11), (10, 11, 12), (0, 13, 14), (13, 14, 15),
                  (14, 15, 16), (0, 17, 18), (17, 18, 19), (18, 19, 20))


def feature_vector(value: object, strategy: str = "normalized", handedness: str = "Right") -> np.ndarray:
    """Retorna atributos versionados para ``raw``, ``normalized``, ``distances`` ou ``angles``."""
    strategy = str(strategy).lower()
    if strategy not in {"raw", "normalized", "distances", "angles"}:
        raise ValueError(f"Estratégia de atributos desconhecida: {strategy}")
    raw = as_landmarks(value)
    if strategy == "raw":
        return raw.reshape(-1).astype(np.float64, copy=False)
    points = canonical_landmarks(raw, handedness)
    if strategy == "normalized":
        return points.reshape(-1)
    if strategy == "distances":
        return np.asarray([np.linalg.norm(points[a] - points[b]) for a, b in _DISTANCE_PAIRS], dtype=np.float64)
    angles: list[float] = []
    for a, b, c in _ANGLE_TRIPLES:
        first, second = points[a] - points[b], points[c] - points[b]
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator <= 1e-12:
            angles.append(0.0)
        else:
            angles.append(float(np.arccos(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))))
    return np.asarray(angles, dtype=np.float64)


class LandmarkFeatures(BaseEstimator, TransformerMixin):
    """Transformador sklearn sem estado; recebe linhas planas já canonicalizadas quando aplicável."""

    def __init__(self, strategy: str = "normalized") -> None:
        self.strategy = strategy

    def fit(self, X: object, y: object = None) -> "LandmarkFeatures":
        return self

    def transform(self, X: object) -> np.ndarray:
        array = np.asarray(X, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != 63:
            raise ValueError("O pipeline espera linhas com 63 coordenadas.")
        # Entradas de treino/inferência já passam por canonical_landmarks quando
        # normalização geométrica é necessária. Não há lateralidade nesta etapa.
        if self.strategy == "raw" or self.strategy == "normalized":
            return array
        if self.strategy == "distances":
            return np.asarray([[np.linalg.norm(row.reshape(21, 3)[a] - row.reshape(21, 3)[b])
                                for a, b in _DISTANCE_PAIRS] for row in array], dtype=np.float64)
        if self.strategy == "angles":
            rows = []
            for row in array:
                points = row.reshape(21, 3)
                row_angles = []
                for a, b, c in _ANGLE_TRIPLES:
                    first, second = points[a] - points[b], points[c] - points[b]
                    den = float(np.linalg.norm(first) * np.linalg.norm(second))
                    row_angles.append(0.0 if den <= 1e-12 else float(np.arccos(np.clip(np.dot(first, second) / den, -1.0, 1.0))))
                rows.append(row_angles)
            return np.asarray(rows, dtype=np.float64)
        raise ValueError(f"Estratégia de atributos desconhecida: {self.strategy}")


def training_matrix(landmarks: Iterable[object], handedness: Iterable[str], strategy: str) -> np.ndarray:
    """Prepara dados exatamente como ``predict``; valida cada amostra."""
    rows = []
    for points, side in zip(landmarks, handedness, strict=True):
        # Raw preserva pontos crus; as demais estratégias usam o espaço canônico.
        values = as_landmarks(points) if strategy == "raw" else canonical_landmarks(points, side)
        rows.append(values.reshape(-1))
    if not rows:
        raise ValueError("O dataset não possui amostras.")
    return np.asarray(rows, dtype=np.float64)
