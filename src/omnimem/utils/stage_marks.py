"""CPU-clock stage markers ``(name, time.monotonic(), extra)`` for inference timelines."""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class StageMarks:
    """Append-only, thread-safe list of named timestamps."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[Tuple[str, float, Dict[str, Any]]] = []
        self.t0 = time.monotonic()

    def mark(self, name: str, **extra: Any) -> float:
        t = time.monotonic()
        with self._lock:
            self._events.append((name, t, extra))
        return t

    def events(self) -> List[Tuple[str, float, Dict[str, Any]]]:
        with self._lock:
            return list(self._events)

    def first(self, name: str) -> Optional[float]:
        for n, t, _ in self.events():
            if n == name:
                return t
        return None

    def all(self, name: str) -> List[Tuple[float, Dict[str, Any]]]:
        return [(t, e) for n, t, e in self.events() if n == name]

    def relative(self, t0: Optional[float] = None) -> List[Dict[str, Any]]:
        base = self.t0 if t0 is None else t0
        return [{"name": n, "t": t - base, **e} for n, t, e in self.events()]
