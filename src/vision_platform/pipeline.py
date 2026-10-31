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
    emit = [n for n in nodes if n.type.startswith("emit.")]
    if len(emit) != 1:
        raise PipelineSpecError(
            f"pipeline must end in exactly one emit.* node, found {len(emit)}")
    runtime = dict(doc.get("runtime") or {})
    _check_backends(nodes, runtime)
    return PipelineSpec(name=str(doc["name"]), camera_id=str(doc["camera_id"]),
                        source=dict(doc.get("source") or {}), runtime=runtime,
                        nodes=nodes, zones=zones)


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
        for inp in n.inputs:
            got = REGISTRY[by_name[inp].type].produces
            if got != want:
                raise PipelineSpecError(
                    f"port mismatch: node {n.name} ({n.type}) consumes {want} "
                    f"but input {inp} produces {got}")


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


def _check_backends(nodes: list[NodeSpec], runtime: dict[str, Any]) -> None:
    profile = runtime.get("profile", "dev-cpu")
    backend = PROFILE_BACKEND.get(profile)
    if backend is None:
        raise PipelineSpecError(f"unknown runtime profile: {profile}")
    for n in nodes:
        if backend not in REGISTRY[n.type].backends:
            raise PipelineSpecError(
                f"node {n.name} ({n.type}) has no {backend} backend for profile {profile}")
