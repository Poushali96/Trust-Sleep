
from __future__ import annotations
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, Optional
import numpy as np


@dataclass
class SessionRecord:
    previous_contribution: Optional[np.ndarray] = None
    previous_timestamp: Optional[float] = None
    active_event_id: Optional[str] = None


class SessionStateStore:
    """Thread-safe patient/recording/session-specific temporal state."""
    def __init__(self):
        self._lock = RLock()
        self._records: Dict[str, SessionRecord] = {}

    def get(self, session_id: str) -> SessionRecord:
        with self._lock:
            return self._records.setdefault(session_id, SessionRecord())

    def update_contribution(
        self, session_id: str, contribution: np.ndarray, timestamp: float
    ) -> None:
        with self._lock:
            record = self._records.setdefault(session_id, SessionRecord())
            record.previous_contribution = np.asarray(contribution).copy()
            record.previous_timestamp = float(timestamp)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._records.pop(session_id, None)
