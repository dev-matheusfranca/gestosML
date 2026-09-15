"""Estado temporal da demonstração, separado das métricas do classificador."""

from collections import Counter, deque
import math
import time


class TemporalStabilizer:
    def __init__(self, window=5, consensus=0.7):
        if window < 1 or not 0 < consensus <= 1:
            raise ValueError("Janela e consenso inválidos")
        self.window = int(window)
        self.consensus = consensus
        self.history = deque(maxlen=self.window)
        self.changed_at = None
        self.stable = None
        self.response_delay_ms = 0.0
        self._candidate = None
        self._candidate_since = None

    def reset(self):
        self.history.clear()
        self.stable = None
        self._candidate = None
        self._candidate_since = None
        self.response_delay_ms = 0.0

    def update(self, label, now=None):
        now = time.monotonic() if now is None else now
        if label is None:
            self.reset()
            return None
        if label != self._candidate:
            self._candidate, self._candidate_since = label, now
        self.history.append(label)
        most, count = Counter(self.history).most_common(1)[0]
        value = most if len(self.history) == self.window and count >= math.ceil(self.window * self.consensus) else None
        if value != self.stable:
            self.changed_at = now
            self.response_delay_ms = max(0, (now - self._candidate_since) * 1000) if value == self._candidate else 0.0
        self.stable = value
        return value


class ActionGate:
    """Um gesto aciona uma vez; é necessário neutro e cooldown para rearmar."""
    def __init__(self, cooldown=1.5):
        if cooldown < 0:
            raise ValueError("Intervalo não pode ser negativo")
        self.cooldown = cooldown
        self.last_action = -float("inf")
        self.armed = True

    def update(self, label, now=None):
        now = time.monotonic() if now is None else now
        if label is None:
            self.armed = True
            return None
        if self.armed and now - self.last_action >= self.cooldown:
            self.armed = False
            self.last_action = now
            return label
        return None
