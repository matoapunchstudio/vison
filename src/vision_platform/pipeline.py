"""Pipeline documents: node registry, validation, and (Task 7) the runner.

A pipeline is a YAML document declaring a DAG of registered nodes. Validation
happens before any frame is processed: unknown node types, duplicate names,
cycles, port-kind mismatches along edges, unknown zone references, missing
emit node, and node/backend incompatibility with the declared runtime
profile are all PipelineSpecErrors naming the offender.

Nodes may omit `inputs`; the default edge is the previous node in the list,
which keeps simple linear pipelines terse while the document remains a DAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vision_platform.geometry import Polygon


class PipelineSpecError(ValueError):
    """Raised when a pipeline document fails validation. Message names the offender."""


ALL_BACKENDS = ("cpu", "tensorrt", "deepstream")
PROFILE_BACKEND = {"dev-cpu": "cpu", "jetson-ort": "tensorrt", "jetson-deepstream": "deepstream"}


@dataclass(frozen=True)
class NodeManifest:
    type: str
    version: str
    consumes: str  # port kind: None | Frame | Detections | Observations | Events
    produces: str
    stateful: bool
    backends: tuple[str, ...] = ("cpu",)


def _m(t, consumes, produces, stateful, backends=ALL_BACKENDS):
    return NodeManifest(type=t, version="1", consumes=consumes, produces=produces,
                        stateful=stateful, backends=backends)


REGISTRY: dict[str, NodeManifest] = {
    m.type: m for m in [
        _m("source.file", "None", "Frame", False),
        _m("detect.persons", "Frame", "Detections", False, ("cpu", "tensorrt")),
        _m("track.bytetrack", "Detections", "Observations", True, ("cpu", "tensorrt")),
        _m("face.encode", "Observations", "Observations", True, ("cpu", "tensorrt")),
        _m("face.match", "Observations", "Observations", False, ("cpu", "tensorrt")),
        _m("zone.sequence", "Observations", "Events", True),
        _m("zone.dwell", "Observations", "Events", True),
        _m("condition.gate", "Events", "Events", False),
        _m("debounce", "Events", "Events", True),
        _m("emit.webhook", "Events", "None", False),
        _m("emit.file", "Events", "None", False),
    ]
}


@dataclass(frozen=True)
class NodeSpec:
    name: str
    type: str
    config: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    camera_id: str
    source: dict[str, Any]
    runtime: dict[str, Any]
    nodes: list[NodeSpec]
    zones: dict[str, Polygon]
    version: int = 1

    def node(self, name: str) -> NodeSpec:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def nodes_of_type(self, *types: str) -> list[NodeSpec]:
        return [n for n in self.nodes if n.type in types]


def load_pipeline(path: Path) -> PipelineSpec:
    path = Path(path)
    if not path.exists():
        raise PipelineSpecError(f"pipeline file not found: {path}")
    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict):
        raise PipelineSpecError(f"pipeline document must be a mapping: {path}")
    return parse_pipeline(doc)


def parse_pipeline(doc: dict[str, Any]) -> PipelineSpec:
    for key in ("name", "camera_id", "nodes"):
        if key not in doc:
            raise PipelineSpecError(f"pipeline missing required field: {key}")
    zones: dict[str, Polygon] = {
        name: [(float(x), float(y)) for x, y in poly]
        for name, poly in (doc.get("zones") or {}).items()
    }
    nodes: list[NodeSpec] = []
    seen: set[str] = set()
    for raw in doc["nodes"]:
        name, ntype = raw.get("name"), raw.get("type")
        if not name or not ntype:
            raise PipelineSpecError(f"node missing name/type: {raw!r}")
        if name in seen:
            raise PipelineSpecError(f"duplicate node name: {name}")
        seen.add(name)
        if ntype not in REGISTRY:
            raise PipelineSpecError(f"unknown node type: {ntype} (node {name})")
        nodes.append(NodeSpec(name=name, type=ntype,
                              config=dict(raw.get("config") or {}),
                              inputs=list(raw.get("inputs") or [])))
    # Default edges: previous node in list order.
    nodes = [NodeSpec(n.name, n.type, n.config,
                      n.inputs or ([nodes[i - 1].name] if i > 0 else []))
             for i, n in enumerate(nodes)]
    by_name = {n.name: n for n in nodes}
    for n in nodes:
        for inp in n.inputs:
            if inp not in by_name:
                raise PipelineSpecError(f"node {n.name} has unknown input: {inp}")
    _check_acyclic(nodes)
    _check_ports(nodes, by_name)
    _check_zone_refs(nodes, zones)
    _check_node_configs(nodes)
    emit = [n for n in nodes if n.type.startswith("emit.")]
    if len(emit) != 1:
        raise PipelineSpecError(
            f"pipeline must end in exactly one emit.* node, found {len(emit)}")
    runtime = dict(doc.get("runtime") or {})
    _check_backends(nodes, runtime)
    return PipelineSpec(name=str(doc["name"]), camera_id=str(doc["camera_id"]),
                        source=dict(doc.get("source") or {}), runtime=runtime,
                        nodes=nodes, zones=zones, version=int(doc.get("version", 1)))


def _check_acyclic(nodes: list[NodeSpec]) -> None:
    by_name = {n.name: n for n in nodes}
    visiting, done = set(), set()

    def visit(name: str, trail: list[str]) -> None:
        if name in done:
            return
        if name in visiting:
            raise PipelineSpecError(f"cycle detected in pipeline graph: {' -> '.join(trail + [name])}")
        visiting.add(name)
        for inp in by_name[name].inputs:
            visit(inp, trail + [name])
        visiting.discard(name)
        done.add(name)

    for n in nodes:
        visit(n.name, [])


def _check_ports(nodes: list[NodeSpec], by_name: dict[str, NodeSpec]) -> None:
    for n in nodes:
        want = REGISTRY[n.type].consumes
        if want == "None":
            continue
        if not n.inputs:
            raise PipelineSpecError(
                f"port mismatch: node {n.name} ({n.type}) consumes {want} but has no inputs")
        for inp in n.inputs:
            got = REGISTRY[by_name[inp].type].produces
            if got != want:
                raise PipelineSpecError(
                    f"port mismatch: node {n.name} ({n.type}) consumes {want} "
                    f"but input {inp} produces {got}")


_REQUIRED_CONFIG = {
    "zone.sequence": ("order", "within_s", "event_type"),
    "zone.dwell": ("zone", "min_dwell_s", "event_type"),
    "debounce": ("cooldown_s",),
}


def _check_node_configs(nodes: list[NodeSpec]) -> None:
    types = {n.type for n in nodes}
    if "face.match" in types and "face.encode" not in types:
        raise PipelineSpecError("face.match requires a face.encode node upstream")
    for n in nodes:
        for key in _REQUIRED_CONFIG.get(n.type, ()):
            if key not in n.config:
                raise PipelineSpecError(
                    f"node {n.name} ({n.type}) missing required config: {key}")
        if n.type == "zone.sequence" and len(n.config.get("order") or []) < 2:
            raise PipelineSpecError(
                f"node {n.name} (zone.sequence) config order needs at least two zones")
        if n.type == "condition.gate":
            conds = n.config.get("conditions")
            if not isinstance(conds, list) or not conds:
                raise PipelineSpecError(
                    f"node {n.name} (condition.gate) missing required config: conditions")


def _check_zone_refs(nodes: list[NodeSpec], zones: dict[str, Polygon]) -> None:
    for n in nodes:
        refs: list[str] = []
        if n.type == "zone.sequence":
            refs = list(n.config.get("order") or [])
        elif n.type == "zone.dwell" and n.config.get("zone"):
            refs = [n.config["zone"]]
        for r in refs:
            if r not in zones:
                raise PipelineSpecError(f"node {n.name} references unknown zone: {r}")


# ---------------------------------------------------------------------------
# Runner (Task 7). One generic execution path for every pipeline document:
# frame plane nodes (detect/track/face) produce enriched observations;
# activity nodes (sequence/dwell) turn them into events; gate and debounce
# filter; the emitter persists + delivers. Use-case knowledge exists only in
# node config and zone geometry, never here.
# ---------------------------------------------------------------------------

# Node types this runner actually constructs/executes, listed explicitly so
# the conformance test fails when a manifest is registered without wiring.
RUNNER_NODE_TYPES = frozenset({
    "source.file", "detect.persons", "track.bytetrack", "face.encode",
    "face.match", "zone.sequence", "zone.dwell", "condition.gate",
    "debounce", "emit.webhook", "emit.file",
})


@dataclass
class RunStats:
    frames: int = 0
    events: int = 0


@dataclass
class _TrackFace:
    best_embedding: Any = None
    best_quality: float = -1.0
    max_width: float = 0.0
    last_seen_ts: float = 0.0
    seen_face: bool = False
    identity: str | None = None
    identity_score: float | None = None
    entry_event_id: str | None = None
    entry_identity_was_none: bool = False
    resolved_emitted: bool = False


class PipelineRunner:
    """Executes a validated PipelineSpec over an iterable of (ts, frame_bgr)."""

    def __init__(self, spec: PipelineSpec, detector, encoder, gallery, emitter,
                 machine_id: str = "dev-pc", recorder=None):
        from vision_platform.events import (Debouncer, DwellDetector, Gate,
                                            SequenceDetector, ZoneSet)
        from vision_platform.track import ByteTrackAdapter

        self.spec = spec
        self.detector = detector
        self.encoder = encoder
        self.gallery = gallery
        self.emitter = emitter
        self.machine_id = machine_id
        self.recorder = recorder
        self.pipeline_label = f"{spec.name}@{spec.version}"

        fps = float(spec.source.get("analytics_fps", 5))
        self._tracker = ByteTrackAdapter(camera_id=spec.camera_id, frame_rate=int(fps))
        # Each activity carries its own ZoneSet so per-node anchors are honored.
        self._activities: list[tuple[Any, Any]] = []
        for n in spec.nodes_of_type("zone.sequence", "zone.dwell"):
            zones = ZoneSet(spec.zones, anchor=str(n.config.get("anchor", "foot")))
            if n.type == "zone.sequence":
                act = SequenceDetector(
                    order=list(n.config["order"]), within_s=float(n.config["within_s"]),
                    event_type=str(n.config["event_type"]))
            else:
                act = DwellDetector(
                    zone=str(n.config["zone"]), min_dwell_s=float(n.config["min_dwell_s"]),
                    event_type=str(n.config["event_type"]))
            self._activities.append((act, zones))
        self._gates = [Gate(list(n.config.get("conditions") or []))
                       for n in spec.nodes_of_type("condition.gate")]
        self._debouncers = [Debouncer(float(n.config["cooldown_s"]))
                            for n in spec.nodes_of_type("debounce")]
        self._face_enabled = encoder is not None and bool(spec.nodes_of_type("face.encode"))
        self._match_enabled = bool(spec.nodes_of_type("face.match"))
        match = spec.nodes_of_type("face.match")
        self._match_threshold = float(match[0].config.get("match_threshold", 0.5)) if match else 0.5
        enc = spec.nodes_of_type("face.encode")
        self._min_face_px = float(enc[0].config.get("min_face_px", 40)) if enc else 40.0
        self._face_state: dict[int, _TrackFace] = {}
        self._identity_owner: dict[str, int] = {}

    # -- face association ----------------------------------------------------
    def _enrich(self, observations, frame):
        from dataclasses import replace

        if not self._face_enabled or not observations:
            return observations, set()
        now = max(o.ts for o in observations)
        self._face_state = {k: v for k, v in self._face_state.items()
                            if now - v.last_seen_ts <= 3600.0}
        if all((s := self._face_state.get(o.track_id)) is not None and s.identity
               for o in observations):
            # Tiered face policy: identified tracks cost no further encoding.
            from dataclasses import replace as _rep
            return [_rep(o, identity=self._face_state[o.track_id].identity,
                         identity_score=self._face_state[o.track_id].identity_score)
                    for o in observations], set()
        hits = self.encoder.embed(frame)
        # Exclusive assignment: each face belongs to at most one track — the
        # smallest (most specific) bbox containing its center; strongest
        # faces first. Without this, one face identifies every overlapping
        # person in a crowd.
        assignment: dict[int, Any] = {}
        free = {o.track_id: o for o in observations}
        for hit in sorted(hits, key=lambda h: -h.quality):
            cx = (hit.bbox[0] + hit.bbox[2]) / 2
            cy = (hit.bbox[1] + hit.bbox[3]) / 2
            cands = [o for o in free.values()
                     if o.bbox[0] <= cx <= o.bbox[2] and o.bbox[1] <= cy <= o.bbox[3]]
            if not cands:
                continue
            owner = min(cands, key=lambda o: (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]))
            assignment[owner.track_id] = hit
            del free[owner.track_id]
        just_resolved: set[int] = set()
        enriched = []
        for obs in observations:
            state = self._face_state.setdefault(obs.track_id, _TrackFace())
            state.last_seen_ts = obs.ts
            best = assignment.get(obs.track_id)
            if best is not None:
                width = best.bbox[2] - best.bbox[0]
                state.seen_face = True
                state.max_width = max(state.max_width, width)
                if best.quality > state.best_quality:
                    state.best_quality = best.quality
                    state.best_embedding = best.embedding
                # Try identification on every gated face, not only on
                # score improvements: detector score is a weak proxy for
                # matchability, and a high-score non-match must not block a
                # later slightly-lower-score true match. Size/quality gate:
                # sub-gate faces inform the unresolved reason only.
                if (self._match_enabled and self.gallery is not None
                        and state.identity is None
                        and width >= self._min_face_px and best.quality >= 0.6):
                    hit = self.gallery.identify_with_model(
                        best.embedding, self.encoder.model_id, self._match_threshold)
                    if hit is not None:
                        state.identity, state.identity_score = hit
                        just_resolved.add(obs.track_id)
            enriched.append(replace(obs, identity=state.identity,
                                    identity_score=state.identity_score))
        return enriched, just_resolved

    def _unresolved_reason(self, track_id: int) -> str:
        state = self._face_state.get(track_id)
        if state is None or not state.seen_face:
            return "no_face_detected"
        if state.max_width < self._min_face_px:
            return "face_too_small"
        if state.best_quality < 0.6:
            return "poor_quality"
        return "below_threshold"

    # -- main loop -------------------------------------------------------------
    def run(self, source) -> RunStats:
        import time
        import uuid
        from dataclasses import replace

        from vision_platform.types import MachineEvent

        stats = RunStats()
        for ts, frame in source:
            stats.frames += 1
            h, w = frame.shape[:2]
            t0 = time.perf_counter()
            detections = self.detector.detect(frame)
            self._record("detect.persons", t0)
            t0 = time.perf_counter()
            observations = self._tracker.update(detections, ts=ts, frame_size=(w, h))
            self._record("track.bytetrack", t0)
            t0 = time.perf_counter()
            observations, just_resolved = self._enrich(observations, frame)
            self._record("face.encode+match", t0)

            for obs in observations:
                for activity, zone_set in self._activities:
                    event = activity.update(obs, zone_set.zones_of(obs))
                    if event is None:
                        continue
                    if event.identity is None and self._face_enabled:
                        event = replace(event, data={
                            **event.data,
                            "unresolved_reason": self._unresolved_reason(obs.track_id)})
                    if any(not g.passes(event) for g in self._gates):
                        continue
                    if any(not d.allow(event) for d in self._debouncers):
                        continue
                    self.emitter.emit(event, self.machine_id, self.pipeline_label)
                    stats.events += 1
                    state = self._face_state.setdefault(obs.track_id, _TrackFace())
                    if state.entry_event_id is None:
                        state.entry_event_id = event.event_id
                        state.entry_identity_was_none = event.identity is None
                # Late identity resolution for a track whose entry already went
                # out unresolved: emit a linked event so downstream can merge.
                state = self._face_state.get(obs.track_id)
                if (state is not None and obs.track_id in just_resolved
                        and state.entry_event_id is not None
                        and state.entry_identity_was_none and not state.resolved_emitted):
                    data = {"entry_event_id": state.entry_event_id}
                    owner = self._identity_owner.get(state.identity or "")
                    if owner is not None and owner != obs.track_id:
                        data["duplicate_identity"] = True
                    else:
                        self._identity_owner[state.identity or ""] = obs.track_id
                    resolved = MachineEvent(
                        event_id=str(uuid.uuid4()), type="identity_resolved", ts=obs.ts,
                        camera_id=obs.camera_id, track_id=obs.track_id,
                        identity=state.identity, identity_score=state.identity_score,
                        data=data)
                    self.emitter.emit(resolved, self.machine_id, self.pipeline_label)
                    stats.events += 1
                    state.resolved_emitted = True
                elif state is not None and state.identity:
                    self._identity_owner.setdefault(state.identity, obs.track_id)
        return stats

    def _record(self, node: str, t0: float) -> None:
        if self.recorder is not None:
            import time
            self.recorder.record(node, (time.perf_counter() - t0) * 1000.0)


def _check_backends(nodes: list[NodeSpec], runtime: dict[str, Any]) -> None:
    profile = runtime.get("profile", "dev-cpu")
    backend = PROFILE_BACKEND.get(profile)
    if backend is None:
        raise PipelineSpecError(f"unknown runtime profile: {profile}")
    for n in nodes:
        if backend not in REGISTRY[n.type].backends:
            raise PipelineSpecError(
                f"node {n.name} ({n.type}) has no {backend} backend for profile {profile}")
    providers = runtime.get("providers")
    if providers:
        acceptable_first = {
            "cpu": {"CPUExecutionProvider"},
            "tensorrt": {"TensorrtExecutionProvider", "CUDAExecutionProvider"},
        }.get(backend)
        if acceptable_first is not None and providers[0] not in acceptable_first:
            raise PipelineSpecError(
                f"runtime provider {providers[0]!r} does not match profile {profile} "
                f"(expected first provider one of {sorted(acceptable_first)})")
