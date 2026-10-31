# vision-platform — machine-side composable computer-vision core (POC)

A machine-side core that turns camera/video input into **composed event
payloads**: object detection, tracking, and face recognition run on the
machine; users compose pipelines from registered capabilities plus
activities and conditions; finished events are delivered to a downstream
system over an HTTP webhook (or a byte-identical file during development).

This project is deliberately **only the machine side**. Dashboards,
attendance/HR rules, reporting, and any central server are downstream
consumers of the payloads and are out of scope. The machine reports what it
saw; downstream decides what it means.

- Research and design rationale: [`docs/2026-11-01-cv-platform-and-face-encoding-research.md`](docs/2026-11-01-cv-platform-and-face-encoding-research.md)
- POC implementation plan: [`docs/superpowers/plans/2026-11-01-cpu-poc-plan.md`](docs/superpowers/plans/2026-11-01-cpu-poc-plan.md)
- Model licenses and hashes: [`models/README.md`](models/README.md)

## Architecture

Three planes, one generic runner. Use-case knowledge lives only in pipeline
documents (YAML), never in core code — the word "attendance" appears in
example config and documentation, not in `src/`.

```
FRAME PLANE (per analyzed frame)          EVENT PLANE (per track)              EMISSION
source.file -> detect.persons ->          zone.sequence / zone.dwell ->        SQLite outbox ->
track.bytetrack -> [face.encode ->        condition.gate -> debounce ->        webhook POST
face.match]  = TrackObservations          = MachineEvents                      or JSONL file
```

Key rules:

- **Frames never leave the frame plane.** Only structured observations
  (track id, bbox, class, optional identity) reach the event plane, which is
  pure Python over source-clock timestamps and is verified by golden JSON
  fixtures (`tests/fixtures/`).
- **Identity binds to a track.** Each track keeps its best-quality face
  embedding; once identified, encoding for that track stops. A track that
  entered unresolved and is identified later emits a linked
  `identity_resolved` event (with `data.entry_event_id`) so downstream can
  merge instead of double-counting. Unresolved entries carry
  `data.unresolved_reason` (`no_face_detected` / `face_too_small` /
  `poor_quality` / `below_threshold`).
- **Events are emitted once, delivered at-least-once.** Everything passes
  through the outbox first; `event_id` is stable across retries/restarts and
  is the downstream dedup key.
- **Runtime profiles swap the frame plane, not the product.** `dev-cpu`
  (this PC), `jetson-ort` (same Python nodes, TensorRT/CUDA providers), and
  a future `jetson-deepstream` frame-plane translation all feed the
  identical event plane. Node manifests declare supported backends; the
  validator rejects a pipeline on a profile that cannot run it.

## Quickstart (this PC)

Requires Python tooling via [uv](https://docs.astral.sh/uv/). The project
environment is `vision-platform/.venv` (Python 3.12, pinned) — never the
system interpreter.

```bash
cd vision-platform
uv venv --python 3.12     # first time only
uv sync                   # install pinned dependencies from uv.lock
# fetch model weights (verify SHA-256 against models/README.md):
cd models && curl -sL -O https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  && curl -sL -O https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx \
  && curl -sL -O https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx && cd ..
uv run pytest             # full suite (model tests included when weights exist)
```

## Usage

```bash
# 1) Enroll a person (each photo must contain exactly one face; a bad photo
#    leaves the gallery file untouched). Enroll with a frontal photo PLUS an
#    in-situ capture from the target camera angle when possible.
uv run vpoc enroll --gallery out/gallery.npz --identity alice \
  --photos alice_front.jpg alice_idspot.jpg \
  --face-detector-model models/face_detection_yunet_2023mar.onnx \
  --face-recognizer-model models/face_recognition_sface_2021dec.onnx

# 2) Run a pipeline on a video file, payloads to a JSONL file (dev default):
uv run vpoc run --pipeline pipelines/attendance_poc.yaml \
  --source some_clip.mp4 --gallery out/gallery.npz \
  --events-file out/events.jsonl --outbox out/outbox.db

#    ...or straight to a webhook (same payloads, store-and-forward):
uv run vpoc run --pipeline pipelines/attendance_poc.yaml \
  --source some_clip.mp4 --gallery out/gallery.npz \
  --webhook https://downstream.example/events --outbox out/outbox.db

# 3) After an outage, re-send anything still pending:
uv run vpoc flush --outbox out/outbox.db --webhook https://downstream.example/events

# 4) Measure per-node latency and estimate cameras per box:
uv run vpoc bench --pipeline pipelines/attendance_poc.yaml \
  --source some_clip.mp4 --frames 100 --face-cameras-fraction 0.4
```

Example payload (`vision.event/1`):

```json
{"schema": "vision.event/1", "event_id": "…uuid4…", "type": "person_entered",
 "ts": 123.4, "machine_id": "dev-pc", "camera_id": "labkom_2",
 "pipeline": "attendance_poc@1", "track_id": 42,
 "identity": "alice", "identity_score": 0.71,
 "data": {"from_zone": "door", "to_zone": "room", "via": []}}
```

Payload fields: `schema` (contract version), `event_id` (dedup key),
`type` (from pipeline config), `ts` (source-clock seconds), `machine_id`,
`camera_id`, `pipeline` (`name@version`), `track_id`, `identity` and
`identity_score` (both `null` when unknown — the event is still emitted),
and `data` (activity-specific: zones, dwell seconds, unresolved reason,
entry link for `identity_resolved`).

## Writing pipelines

A pipeline is one YAML document (see `pipelines/attendance_poc.yaml` and the
non-attendance `pipelines/dwell_alert.yaml` — the generality proof). It
declares `name`, `camera_id`, `source.analytics_fps`, a `runtime` profile,
named `zones` (normalized 0–1 polygons over the frame), and a `nodes` DAG.
Nodes may omit `inputs` (default: previous node). Validation rejects unknown
node types, duplicate names, cycles, port mismatches, unknown zone
references, a missing/extra `emit.*` node, and nodes without a backend for
the declared runtime profile — all before the first frame.

Registered nodes in the POC:

| Type | Consumes → Produces | Key config |
|---|---|---|
| `source.file` | — → Frame | (source path comes from the CLI) |
| `detect.persons` | Frame → Detections | `score_threshold`, `tile_grid: [x, y]` (native-resolution tiling for wide shots) |
| `track.bytetrack` | Detections → Observations | — |
| `face.encode` | Observations → Observations | `min_face_px` (quality/size gate) |
| `face.match` | Observations → Observations | `match_threshold` (cosine; **must be calibrated per site**) |
| `zone.sequence` | Observations → Events | `order: [zones…]`, `within_s`, `event_type`, `anchor: foot\|center` |
| `zone.dwell` | Observations → Events | `zone`, `min_dwell_s`, `event_type`, `anchor` |
| `condition.gate` | Events → Events | `conditions: [{field, op, value}]`, ops `eq/ne/gt/gte/lt/lte/in/is_null/not_null`, all must pass |
| `debounce` | Events → Events | `cooldown_s` (keyed by identity or track + type) |
| `emit.webhook` / `emit.file` | Events → — | delivery target comes from the CLI |

Sequence semantics worth knowing: a track is eligible for a sequence only
if its **first** observation lies in the first zone (born at the door), so
occlusion-split tracks born mid-room never count as entries. Zone anchors:
`foot` (bottom-center, for walking people) or `center` (for seated/occluded
tracks whose feet are hidden).

### Adding a capability

1. Implement the node in `src/vision_platform/` behind the relevant protocol
   (frame-plane capabilities produce/consume observations; activities follow
   the `update(obs, zones) -> MachineEvent | None` shape in `events.py`).
2. Add a `NodeManifest` to `REGISTRY` in `pipeline.py` with typed ports and
   the backends it honestly supports.
3. Wire construction in `PipelineRunner` (the registry-conformance test
   fails until every manifest is handled).
4. Add tests first (this codebase is TDD): unit tests for the node, a golden
   fixture if it touches the event plane, and a model-marked integration test
   if it loads weights. Record new weights in `models/README.md`.

The validator, runner contract, outbox, and payload schema must not change
when a capability is added. User pipelines are config only — arbitrary user
code never executes on the machine.

## Face recognition notes

- Baseline encoder: YuNet + SFace (MIT / Apache-2.0), chosen for clean
  commercial licensing. Embeddings are 128-d, L2-normalized; matching is
  cosine similarity against per-person mean templates.
- Embeddings are **model-bound**: every gallery stores its `model_id`, and
  cross-model matching raises. Changing encoders means re-enrollment.
- `match_threshold: 0.5` in the example is an uncalibrated placeholder.
  Calibrate per camera/site against a target false-accept rate before any
  real use.
- On the Labkom 2 sample frame (1920×1080, one wide classroom camera), only
  **2 of ~30 seated students** yield usable faces, and tiled YOLOX finds a
  meaningful subset of persons. Identification is therefore designed around
  standing moments — walk-in capture, a start-of-class look-up, and a
  front `id_spot` for unresolved students — with a downstream teacher
  counter comparing *counted* entries vs *attended* (identified) entries.
  That SOP and any counter UI are downstream, outside this repo. Full
  analysis: the research report in `docs/`.
- Attendance means *entered the room* (check-in only). `person_left` is
  available as a reversed sequence per pipeline, default off.

## Measured POC results (this PC, CPU, 2026-11-01)

- Test suite: see `uv run pytest` (unit + golden-fixture + real-model
  integration tests; model tests skip with a printed reason if weights are
  absent).
- Latency on the Labkom still clip (tiled 2×2 detection): detect ≈ 539 ms,
  face encode+match ≈ 79 ms, tracking ≈ 0.7 ms per analyzed frame. CPU
  tiled detection is the bottleneck by an order of magnitude; these numbers
  say nothing about Jetson+GPU performance, which must be re-measured on
  target hardware (`vpoc bench` prints its assumptions with every estimate).

## Known limits (not proven by this POC)

- No Jetson verification yet: `jetson-ort` packaging, TensorRT providers,
  and any DeepStream backend are unexercised (no NVIDIA GPU on the dev PC).
  `Dockerfile.jetson` is a skeleton with TODOs, not a verified build.
- RTSP/multi-camera operation is unexercised; sources are video files here.
- Thresholds are uncalibrated; no accuracy claim is made for any camera.
- Track ID switches can still split one person into an unresolved entry
  plus a later resolution; identity-keyed debounce and `identity_resolved`
  linking mitigate, and unknown-track duplicates remain a documented limit.
- Face alignment/encoding in a future DeepStream backend needs a hybrid
  probe or custom preprocess library (see research report).
- Biometric data of minors is highly regulated (GDPR special category,
  EU AI Act high-risk, school policies). Consent, retention limits,
  template protection, human review, and a non-biometric fallback require
  institutional/legal review before any production deployment. Do not
  commit sample frames containing students to this repository.

## Project layout

```
src/vision_platform/
  types.py      Detection / TrackObservation / FaceHit / MachineEvent
  geometry.py   foot/center anchors, point-in-polygon
  events.py     ZoneSet, SequenceDetector, DwellDetector, Gate, Debouncer
  pipeline.py   REGISTRY + manifests, YAML validation, PipelineRunner
  detect.py     YoloxOnnxDetector (tiled), FakeDetector
  track.py      ByteTrackAdapter (trackers package)
  face.py       YuNetSFaceEncoder, FakeEncoder
  gallery.py    enrollment templates, cosine identify, .npz persistence
  emit.py       payload envelope, SQLite Outbox, Webhook/File emitters
  sources.py    VideoFileSource (fps-throttled)
  bench.py      LatencyRecorder, cameras-per-box estimator
  cli.py        vpoc argument parsing (run/enroll/flush/bench)
  cli_impl.py   command implementations + runner construction
pipelines/      example pipeline documents
models/         weight provenance registry (weights themselves git-ignored)
tests/          unit, golden-fixture, and model integration tests
docs/           research report and implementation plans
```

## Development

```bash
uv run pytest            # everything
uv run pytest -m model   # only tests needing downloaded weights
docker build -f Dockerfile.cpu -t vision-platform:cpu .
docker run --rm vision-platform:cpu --help
```

Code style follows the codebase itself: small modules with one
responsibility, dataclasses for data, protocols at capability boundaries,
and a failing test before every behavior change.
