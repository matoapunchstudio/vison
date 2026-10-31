"""Event plane: pure-Python activities and conditions over track observations.

No video, model, I/O, or wall-clock dependency lives here — every class is
driven by TrackObservations carrying source-clock timestamps, so behavior is
fully pinned by golden fixtures (tests/fixtures/*.json).

Activities: SequenceDetector (ordered zone visits), DwellDetector.
Conditions: Gate (declarative field predicates), Debouncer (cooldown).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from vision_platform.geometry import Polygon, center_point, foot_point, point_in_polygon
from vision_platform.types import MachineEvent, TrackObservation


class ZoneSet:
    """Named normalized polygons; resolves an observation to its containing zones."""

    def __init__(self, zones: dict[str, Polygon], anchor: str = "foot"):
        if anchor not in ("foot", "center"):
            raise ValueError(f"anchor must be 'foot' or 'center', got {anchor!r}")
        self.zones = zones
        self.anchor = anchor

    def zones_of(self, obs: TrackObservation) -> frozenset[str]:
        w, h = obs.frame_size
        px, py = foot_point(obs.bbox) if self.anchor == "foot" else center_point(obs.bbox)
        pt = (px / w, py / h)
        return frozenset(name for name, poly in self.zones.items() if point_in_polygon(pt, poly))


def _event(obs: TrackObservation, event_type: str, data: dict[str, Any]) -> MachineEvent:
    return MachineEvent(
        event_id=str(uuid.uuid4()),
        type=event_type,
        ts=obs.ts,
        camera_id=obs.camera_id,
        track_id=obs.track_id,
        identity=obs.identity,
        identity_score=obs.identity_score,
        data=data,
    )


@dataclass
class _Progress:
    step: int
    started_ts: float
    last_seen_ts: float


class SequenceDetector:
    """Emit once when a track visits zones in `order` within `within_s`.

    A track is eligible only if its first-ever observation lies in order[0]
    (e.g. it was born in the door zone); tracks first seen elsewhere can never
    complete the sequence. This prevents mid-room track births (occlusion
    splits) from being counted as entries.
    """

    def __init__(self, order: list[str], within_s: float, event_type: str):
        if len(order) < 2:
            raise ValueError("sequence order needs at least two zones")
        self.order = list(order)
        self.within_s = float(within_s)
        self.event_type = event_type
        self._progress: dict[int, _Progress] = {}
        self._ineligible: dict[int, float] = {}  # track_id -> last seen ts

    def update(self, obs: TrackObservation, zones: frozenset[str]) -> MachineEvent | None:
        self._prune(obs.ts)
        tid = obs.track_id
        if tid in self._ineligible:
            self._ineligible[tid] = obs.ts
            return None
        prog = self._progress.get(tid)
        if prog is None:
            if self.order[0] not in zones:
                self._ineligible[tid] = obs.ts
                return None
            prog = _Progress(step=1, started_ts=obs.ts, last_seen_ts=obs.ts)
            self._progress[tid] = prog
        prog.last_seen_ts = obs.ts
        if obs.ts - prog.started_ts > self.within_s:
            del self._progress[tid]
            return None
        # Advance through any expected zones present now (handles overlapping zones).
        while prog.step < len(self.order) and self.order[prog.step] in zones:
            prog.step += 1
        if prog.step >= len(self.order):
            del self._progress[tid]
            self._ineligible[tid] = obs.ts  # one completion per track
            return _event(obs, self.event_type, {
                "from_zone": self.order[0],
                "to_zone": self.order[-1],
                "via": self.order[1:-1],
            })
        return None

    def _prune(self, now_ts: float) -> None:
        horizon = self.within_s * 4
        self._progress = {k: v for k, v in self._progress.items() if now_ts - v.last_seen_ts <= horizon}
        self._ineligible = {k: v for k, v in self._ineligible.items() if now_ts - v <= horizon}


@dataclass
class _DwellState:
    since_ts: float
    emitted: bool = False


class DwellDetector:
    """Emit once when a track stays continuously in `zone` for `min_dwell_s`."""

    def __init__(self, zone: str, min_dwell_s: float, event_type: str):
        self.zone = zone
        self.min_dwell_s = float(min_dwell_s)
        self.event_type = event_type
        self._state: dict[int, _DwellState] = {}

    def update(self, obs: TrackObservation, zones: frozenset[str]) -> MachineEvent | None:
        tid = obs.track_id
        if self.zone not in zones:
            self._state.pop(tid, None)
            return None
        state = self._state.setdefault(tid, _DwellState(since_ts=obs.ts))
        if not state.emitted and obs.ts - state.since_ts >= self.min_dwell_s:
            state.emitted = True
            return _event(obs, self.event_type,
                          {"zone": self.zone, "dwell_s": obs.ts - state.since_ts})
        return None


class Debouncer:
    """Suppress repeat events per (identity or track, type) within a cooldown window."""

    def __init__(self, cooldown_s: float):
        self.cooldown_s = float(cooldown_s)
        self._last: dict[tuple[str, str], float] = {}

    def allow(self, event: MachineEvent) -> bool:
        key = (event.identity or f"track:{event.track_id}", event.type)
        last = self._last.get(key)
        if last is not None and event.ts - last < self.cooldown_s:
            return False
        self._last[key] = event.ts
        return True


_OPS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "gt": lambda a, b: a is not None and a > b,
    "gte": lambda a, b: a is not None and a >= b,
    "lt": lambda a, b: a is not None and a < b,
    "lte": lambda a, b: a is not None and a <= b,
    "in": lambda a, b: a in b,
    "is_null": lambda a, b: a is None,
    "not_null": lambda a, b: a is not None,
}


def _field_value(obj: Any, name: str) -> Any:
    if isinstance(obj, MachineEvent) and name in obj.data:
        return obj.data[name]
    return getattr(obj, name, None)


class Gate:
    """Declarative condition list, `all` semantics (POC).

    Each condition is {"field": str, "op": one of eq/ne/gt/gte/lt/lte/in/is_null/not_null,
    "value": ...}. Fields resolve on the observation/event attributes first,
    then on MachineEvent.data.
    """

    def __init__(self, conditions: list[dict[str, Any]]):
        for cond in conditions:
            if cond.get("op") not in _OPS:
                raise ValueError(f"unknown gate op: {cond.get('op')!r}")
            if "field" not in cond:
                raise ValueError(f"gate condition missing 'field': {cond!r}")
        self.conditions = conditions

    def passes(self, obj: TrackObservation | MachineEvent) -> bool:
        return all(_OPS[c["op"]](_field_value(obj, c["field"]), c.get("value"))
                   for c in self.conditions)
