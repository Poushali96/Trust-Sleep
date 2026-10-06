
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Deque, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class StreamSession:
    max_points: int
    timestamps: Deque[float] = field(default_factory=deque)
    rows: Deque[List[float]] = field(default_factory=deque)

    def append(
        self,
        timestamps: Sequence[float],
        signal_rows: Sequence[Sequence[float]],
    ) -> None:
        for timestamp, row in zip(timestamps, signal_rows):
            if len(row) != 4:
                raise ValueError("Every signal row must have four channels.")
            self.timestamps.append(float(timestamp))
            self.rows.append([float(value) for value in row])
            while len(self.rows) > self.max_points:
                self.rows.popleft()
                self.timestamps.popleft()

    def snapshot(self, minimum_points: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.rows) < minimum_points:
            raise ValueError(
                f"Need at least {minimum_points} samples; received {len(self.rows)}."
            )
        return (
            np.asarray(self.timestamps, dtype=np.float64),
            np.asarray(self.rows, dtype=np.float32),
        )


class StreamBufferStore:
    def __init__(self, max_points: int = 600):
        self.max_points = int(max_points)
        self._lock = RLock()
        self._sessions: Dict[str, StreamSession] = {}

    def append(
        self,
        session_id: str,
        timestamps: Sequence[float],
        signal_rows: Sequence[Sequence[float]],
    ) -> StreamSession:
        if len(timestamps) != len(signal_rows):
            raise ValueError("timestamps and signal_rows must have equal length.")
        with self._lock:
            session = self._sessions.setdefault(
                session_id,
                StreamSession(max_points=self.max_points),
            )
            session.append(timestamps, signal_rows)
            return session

    def snapshot(
        self,
        session_id: str,
        minimum_points: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown session: {session_id}")
            return self._sessions[session_id].snapshot(minimum_points)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
