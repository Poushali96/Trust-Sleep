
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import time
import uuid


@dataclass
class Alert:
    event_id: str
    session_id: str
    start_time: float
    last_seen: float
    end_time: Optional[float]
    subtype: str
    maximum_probability: float
    status: str


class AlertManager:
    """Merges overlapping rolling-window alerts and enforces a refractory interval."""
    def __init__(self, merge_gap_seconds: float = 15.0, refractory_seconds: float = 20.0):
        self.merge_gap_seconds = float(merge_gap_seconds)
        self.refractory_seconds = float(refractory_seconds)
        self.active: Dict[str, Alert] = {}
        self.last_closed: Dict[str, float] = {}

    def update(
        self,
        session_id: str,
        event_active: bool,
        subtype: str,
        probability: float,
        timestamp: Optional[float] = None,
    ) -> Optional[Alert]:
        timestamp = float(timestamp if timestamp is not None else time.time())
        current = self.active.get(session_id)

        if event_active:
            last_closed = self.last_closed.get(session_id, float("-inf"))
            if current is None and timestamp - last_closed < self.refractory_seconds:
                return None
            if current is None:
                current = Alert(
                    event_id=str(uuid.uuid4()),
                    session_id=session_id,
                    start_time=timestamp,
                    last_seen=timestamp,
                    end_time=None,
                    subtype=subtype,
                    maximum_probability=float(probability),
                    status="active",
                )
                self.active[session_id] = current
            else:
                current.last_seen = timestamp
                current.maximum_probability = max(
                    current.maximum_probability, float(probability)
                )
                current.subtype = subtype
            return current

        if current is not None and timestamp - current.last_seen > self.merge_gap_seconds:
            current.end_time = timestamp
            current.status = "closed"
            self.last_closed[session_id] = timestamp
            self.active.pop(session_id, None)
            return current
        return current
