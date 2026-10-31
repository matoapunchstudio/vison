# CPU-Only POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove on this CPU-only Linux PC (no GPU, no webcam) that the machine-side core — video in, object detection, tracking, face recognition, composed-event payloads out — turns a video file into debounced A→B events with optional face identity, delivered through a store-and-forward HTTP webhook emitter, in a form that ports unchanged to Jetson at site (60 cameras / 3 Jetsons).

**Architecture:** Three planes from the research report. Frame plane (file source → person detector → tracker → optional face encode/match) emits structured track observations only; the event plane (zone membership → sequence state machine → debounce) is pure Python with no video or model dependency; its only output is versioned JSON payloads persisted to a local outbox and POSTed to a configurable HTTP webhook (a file emitter with the identical payload serves the POC on this PC). Pipelines are YAML documents validated against a node registry before any frame is processed. The event plane is built and proven first with synthetic tracks, so the POC's core claim does not depend on model downloads succeeding on this PC.

**Tech Stack:** Python 3.12 (uv-managed, project-local `.venv`), ONNX Runtime CPU, OpenCV (YuNet + SFace via `cv2.FaceDetectorYN` / `cv2.FaceRecognizerSF`), supervision ByteTrack, NumPy, PyYAML, pytest.

**Spec:** `docs/2026-11-01-cv-platform-and-face-encoding-research.md` (especially §2.2–2.6 and the 2026-11-01 dev-vs-site update).

## Scope Boundary (machine side only, decided 2026-11-01)

- **In scope:** camera/file ingestion, object detection, tracking, face detection/recognition against a local gallery, pipeline composition (zones, sequences, dwell, debounce), local pipeline config, health/latency metrics, and emitting payloads.
- **Out of scope:** dashboards/UI, downstream business systems (attendance rules, HR, reporting, access control decisions), central multi-site servers, user management. Downstream systems consume payloads and decide what an event *means*; the machine only reports what it *saw*. The example pipeline file may name its emitted event for an attendance use case, but no attendance rule (shifts, lateness, absence) exists in this codebase.
- **General core, attendance is only an example:** the product is a composable capability platform. Nothing in the runner, registry, or payload envelope may hard-code attendance, classrooms, students, or faces. New computer-vision capabilities are added over time as registered nodes; users compose pipelines from existing capabilities plus activities (temporal operators: sequence, dwell, count, absence) and conditions (zone/schedule/score/class/identity gates, boolean combination, debounce) declared in config — never arbitrary user code.
- **Payload contract:** HTTP webhook `POST` of one JSON object per composed event (never per-frame detections), envelope `{"schema": "vision.event/1", "event_id": "<uuid4>", "type": "<from pipeline config>", "ts": <source-clock seconds>, "machine_id": "...", "camera_id": "...", "pipeline": "<name@version>", "track_id": 0, "identity": null, "identity_score": null, "data": {"from_zone": "...", "to_zone": "...", "via": ["..."]}}`. Unknown face ⇒ `identity: null`, event still emitted. Fields a pipeline does not produce are omitted/null, so non-face pipelines emit valid payloads unchanged.
- **Delivery:** store-and-forward. Every event is written to a local SQLite outbox before any send attempt; a sender POSTs pending rows, marks them sent only on HTTP 2xx, and retries with backoff. Delivery is at-least-once; `event_id` is the downstream dedup key and is stable across retries and restarts.

## Global Constraints

- Project root is `vision-platform/`; environment is `vision-platform/.venv` created by `uv venv --python 3.12`; never system Python 3.14, never `--break-system-packages`.
- CPU only on this PC: ONNX Runtime provider list defaults to `["CPUExecutionProvider"]` and is config-driven (`runtime.providers`), so Jetson later sets TensorRT/CUDA with no code change.
- No webcam and no RTSP in the POC acceptance path: sources are video files and synthetic track feeds. RTSP stays a config value on the same source node, unexercised here.
- Shipped-model licensing stays commercial-clean: person detector YOLOX (Apache-2.0, OpenCV Zoo ONNX), face YuNet (MIT) + SFace (Apache-2.0). No InsightFace packs, no Ultralytics code or weights anywhere in the repo, including tests.
- Every stored embedding carries `model_id`; matching an embedding against a gallery built by a different `model_id` raises, never silently compares.
- Site target is 60 cameras / 3 Jetsons = 20 cameras per box, of which classroom (face) cameras are fewer than half site-wide (< 30 total, < 10/box only if assignment is balanced — plan for the worst box). The POC must produce a per-node latency table; it must not claim Jetson throughput it did not measure. Extrapolation math lives in one estimator with its assumptions printed in its output, including a synchronized entry-burst scenario (all classrooms start together).
- Analytics FPS is a per-pipeline config (`source.analytics_fps`, POC default 5); the runner drops frames to meet it instead of accumulating latency.
- All timestamps in observations and events are float seconds from the source clock, passed in — core logic never calls `time.time()`.

## Review Focus

- Track stands on a zone boundary flickering A↔B — expect at most one emitted event per identity/track per cooldown window (Task 3).
- Track ID switches mid-doorway (same person, new `track_id`) — expect no crash and at most one event per recognized identity per cooldown; unrecognized duplicate is a known POC limitation, documented in README (Task 3, Task 8).
- Person crosses A→B with no usable face (back to camera, too small) — expect the composed event emitted with `identity: null`, not a dropped crossing (Task 7).
- Webhook receiver down or returning 500 during a run — expect events retained in the outbox, sent after recovery with unchanged `event_id`s, none lost and none marked sent early (Task 7).
- Video file missing/corrupt, or model file missing — expect a clear startup error naming the path, before any event file is created (Task 6, Task 7).
- Gallery enrollment photo contains zero or multiple faces — expect enrollment to refuse that photo with a named reason, leaving the gallery unchanged (Task 5).

## File Structure

```
vision-platform/
  pyproject.toml                  # deps, pytest config, `vpoc` console script
  pipelines/attendance_poc.yaml   # the POC pipeline document (zones, nodes, thresholds)
  models/README.md                # model provenance registry (source, license, sha256) + fetch commands
  src/vision_platform/
    types.py        # Detection, TrackObservation, FaceHit, MachineEvent (dataclasses)
    emit.py         # payload envelope, SQLite Outbox, WebhookEmitter, FileEmitter
    geometry.py     # foot_point(), point_in_polygon()
    events.py       # ZoneTracker, SequenceDetector, Debouncer — pure, no I/O
    gallery.py      # Gallery: enroll/identify over normalized embeddings, model_id guard
    face.py         # YuNetSFaceEncoder behind FaceEncoder protocol (+ FakeEncoder for tests)
    detect.py       # YoloxOnnxDetector behind Detector protocol (+ FakeDetector)
    track.py        # ByteTrackAdapter wrapping supervision.ByteTrack
    pipeline.py     # load_pipeline(path) validation + PipelineRunner wiring nodes by name
    sources.py      # VideoFileSource (fps-throttled frame iterator)
    bench.py        # per-node latency recorder + cameras-per-box estimator
    cli.py          # vpoc run|enroll|bench
  tests/
    test_cli.py  test_geometry.py  test_events.py  test_gallery.py  test_track.py
    test_pipeline_spec.py  test_runner_e2e.py  test_bench.py
    test_face_integration.py   # marked `model`; skipped when weights absent
  Dockerfile.cpu                  # this PC / CI parity
  Dockerfile.jetson               # site parity skeleton (base image + providers differ only)
  README.md                       # run instructions against .venv, POC scope + known limits
```

---

### Task 1: Project environment and CLI skeleton

**Files:**
- Create: `pyproject.toml`, `src/vision_platform/__init__.py`, `src/vision_platform/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: console script `vpoc` with `main(argv: list[str] | None = None) -> int`; subcommands `run`, `enroll`, `bench` (bodies filled by Tasks 5–8; unknown subcommand exits 2).

- [ ] **Step 1: Create the environment.** Install uv if absent (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then in `vision-platform/`: `uv venv --python 3.12`. Expected: `.venv/bin/python --version` prints 3.12.x.
- [ ] **Step 2: Write the failing test.** `test_cli.py` asserts `main(["--help"])` exits 0 via `SystemExit` code 0 and `main(["nope"])` returns/exits 2.
- [ ] **Step 3: Run it.** `uv run pytest tests/test_cli.py -v` — expected FAIL (module missing). First `uv add numpy opencv-python-headless onnxruntime supervision pyyaml` and `uv add --dev pytest` so later tasks inherit pinned deps in `uv.lock`.
- [ ] **Step 4: Implement** `cli.main` with `argparse` subparsers only; each subcommand prints `not implemented yet: <name>` and returns 1 until its task lands.
- [ ] **Step 5: Run tests, expect PASS. Commit:** `git init` if needed, `git add -A`, `git commit -m "chore: scaffold vision-platform POC (uv, py3.12, vpoc CLI)"`.

### Task 2: Core types and geometry

**Files:**
- Create: `src/vision_platform/types.py`, `src/vision_platform/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Produces: `Detection(bbox: tuple[float,float,float,float], score: float, class_id: int)`, `TrackObservation(camera_id: str, ts: float, track_id: int, bbox, frame_size: tuple[int,int], class_id: int, identity: str | None = None, identity_score: float | None = None)`, `MachineEvent(event_id: str, type: str, ts: float, camera_id: str, track_id: int, identity: str | None, identity_score: float | None, data: dict)`.
- Produces: `foot_point(bbox) -> tuple[float, float]` (bottom-center), `point_in_polygon(pt, polygon: list[tuple[float,float]]) -> bool` (ray casting; polygon in normalized 0–1 coords, boundary counts as inside).

- [ ] **Step 1: Write failing tests** — foot point of `(10, 20, 30, 60)` is `(20, 60)`; square polygon `[(0,0),(0.5,0),(0.5,0.5),(0,0.5)]` contains `(0.25,0.25)`, excludes `(0.75,0.25)`; concave polygon case from a doorway-shaped zone.
- [ ] **Step 2: Run** `uv run pytest tests/test_geometry.py -v` — expected FAIL.
- [ ] **Step 3: Implement** the dataclasses (frozen) and the two functions. No third-party geometry dependency.
- [ ] **Step 4: Run tests, expect PASS. Commit:** `feat: core types and zone geometry`.

### Task 3: Event plane — zones, A→B sequence, debounce (synthetic proof)

**Files:**
- Create: `src/vision_platform/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: `TrackObservation` (pixel bbox + `frame_size`, so foot points can be normalized), `point_in_polygon`, `foot_point` (Task 2).
- Produces: `ZoneSet(zones: dict[str, list[tuple[float,float]]])` with `zones_of(obs) -> frozenset[str]`; `SequenceDetector(order: list[str], within_s: float, event_type: str)` with `update(obs, zones) -> MachineEvent | None` — `event_type` comes from pipeline config (the example pipeline sets `person_entered`); `data` carries `from_zone`, `to_zone`, `via`. The detector itself has no attendance semantics. (per-track state, expires after `within_s`, resets after emitting); `Debouncer(cooldown_s: float)` with `allow(event) -> bool` keyed by `(identity or f"track:{track_id}", type)`; `DwellDetector(zone: str, min_dwell_s: float, event_type: str)` with `update(obs, zones) -> MachineEvent | None` (second generic activity, proves activities are interchangeable); `Gate(conditions: list[dict])` with `passes(obs_or_event) -> bool` evaluating declarative conditions (`field`/`op`/`value`, combined with `all` semantics in the POC; `any`/`not` are later registry additions, not new runner code).

- [ ] **Step 1: Write failing tests** — scripted observation feeds, no video: (a) A→TH→B within 30 s emits exactly one event of the configured type with `data.from_zone/to_zone` set; (b) A→B after 31 s emits none; (c) B→TH→A with `order=["outside","doorway","inside"]` emits none; (d) boundary flicker A,B,A,B inside cooldown yields one debounced event; (e) ID switch: track 1 goes A→TH, track 2 (same identity `"alice"`) goes TH→B — no event from sequence alone, and a later full crossing by `"alice"` within cooldown of an earlier `"alice"` event is suppressed by `Debouncer`.
- [ ] **Step 2: Run** `uv run pytest tests/test_events.py -v` — expected FAIL.
- [ ] **Step 3: Implement** the three classes. State is `dict[track_id, _Progress]`; prune tracks unseen for `> within_s` on each `update`.
- [ ] **Step 4: Run tests, expect PASS. Commit:** `feat: zone sequence state machine with debounce`.

### Task 4: Pipeline document validation

**Files:**
- Create: `src/vision_platform/pipeline.py`, `pipelines/attendance_poc.yaml`
- Test: `tests/test_pipeline_spec.py`

**Interfaces:**
- Produces: `PipelineSpec(name: str, camera_id: str, source: dict, runtime: dict, nodes: list[NodeSpec], zones: dict)`, `NodeSpec(name: str, type: str, config: dict, inputs: list[str] = [])` where `inputs` names upstream nodes (explicit DAG edges, not list order), `load_pipeline(path: Path) -> PipelineSpec` raising `PipelineSpecError` with the offending node/field named. A node registry maps each type to a manifest `NodeManifest(type: str, version: str, consumes: str, produces: str, stateful: bool, backends: tuple[str, ...] = ("cpu",))` over port kinds `Frame | Detections | Tracks | Observations | Events`; validation checks unknown types, duplicate names, cycles, unconnected inputs, port-kind mismatches along every edge, and — when the pipeline declares `runtime.profile` (`dev-cpu` in the POC; `jetson-ort`/`jetson-deepstream` later) — that every node's `backends` supports that profile's backend. Synthetic event-plane tests must be expressible as golden fixtures (JSON observations in → JSON events out, loaded from `tests/fixtures/*.json`), so future frame-plane backends (TensorRT, DeepStream) can be verified for parity against the identical files. Python is the production language as well as the POC language; no port is planned, and runtime profiles share one codebase. Initial registry: `{"source.file", "detect.persons", "track.bytetrack", "face.encode", "face.match", "zone.sequence", "zone.dwell", "condition.gate", "debounce", "emit.webhook", "emit.file"}`. Every pipeline must end in exactly one `emit.*` node; both emitters write through the outbox (Task 7). Adding a future capability = one new manifest + one node class + registry entry; the validator and runner do not change — a registry-conformance test enumerates all manifests and asserts each has a constructible node class.
- `attendance_poc.yaml` pins the single-wide-camera classroom layout: `source.analytics_fps: 5`, zones `door`/`room`/`id_spot` as normalized polygons, entry sequence `door → room` with `within_s: 30` (track must be born in `door`; mid-`room` births never count as entries), debounce `cooldown_s: 300`, face `match_threshold: 0.5` (placeholder until Task 5 calibration note), `runtime.providers: [CPUExecutionProvider]`. Face encoding runs on native-resolution person crops, never on a downscaled full frame.

- [ ] **Step 1: Write failing tests** — the shipped YAML loads; unknown node type, missing zone referenced by a sequence, and duplicate node names each raise `PipelineSpecError` naming the offender.
- [ ] **Step 2: Run** `uv run pytest tests/test_pipeline_spec.py -v` — expected FAIL.
- [ ] **Step 3: Implement** `load_pipeline` with plain dict validation (no pydantic — keep the aarch64 wheel surface small).
- [ ] **Step 4: Run tests, expect PASS. Commit:** `feat: validated YAML pipeline documents`.

### Task 5: Face encoding and gallery (YuNet + SFace behind an interface)

**Files:**
- Create: `src/vision_platform/face.py`, `src/vision_platform/gallery.py`, `models/README.md`
- Test: `tests/test_gallery.py`, `tests/test_face_integration.py`

**Interfaces:**
- Produces: `FaceEncoder` protocol with `model_id: str` and `embed(frame_bgr: np.ndarray) -> list[FaceHit]`; `FaceHit(bbox, embedding: np.ndarray, quality: float)` where `embedding` is L2-normalized float32. Implementations: `YuNetSFaceEncoder(detector_path, recognizer_path, min_face_px: int = 40)` using `cv2.FaceDetectorYN` + `cv2.FaceRecognizerSF` (align via `alignCrop`, then `feature`); `FakeEncoder` returning caller-supplied hits for unit tests.
- Produces: `Gallery(model_id: str)` with `enroll(identity: str, embedding) -> None` (stores running mean template per identity, re-normalized), `identify(embedding, threshold: float) -> tuple[str, float] | None`, `save(path)` / `load(path)` as `.npz` including `model_id`; cross-`model_id` use raises `ValueError`.

- [ ] **Step 1: Write failing gallery tests** (no models needed) — enroll two near-identical vectors for `"alice"`, one orthogonal for `"bob"`; a probe near alice identifies as `("alice", >0.9)`; an orthogonal probe returns `None` at threshold 0.5; loading a gallery then identifying with a mismatched-`model_id` encoder path raises.
- [ ] **Step 2: Run** `uv run pytest tests/test_gallery.py -v` — expected FAIL.
- [ ] **Step 3: Implement** `gallery.py`, then `face.py`. In `YuNetSFaceEncoder.embed`, drop faces whose bbox width `< min_face_px`; `quality` = detector score.
- [ ] **Step 4: Write the integration test** (`@pytest.mark.model`, skip if `models/face_detection_yunet_2023mar.onnx` absent): embedding the same bundled test portrait twice gives cosine > 0.99; enrollment via CLI (Task 7 wiring may stub here) refuses a no-face image with a named error. Record both ONNX files in `models/README.md`: OpenCV Zoo source path, license (MIT / Apache-2.0), sha256 after download, exact `curl -L` fetch commands. Do not commit weight files; add `models/*.onnx` to `.gitignore`.
- [ ] **Step 5: Run** `uv run pytest tests/test_gallery.py -v` PASS; `uv run pytest -m model -v` PASS or SKIP with reason printed. **Commit:** `feat: SFace face encoding with gallery enrollment/matching`.

### Task 6: Person detection + tracking adapters

**Files:**
- Create: `src/vision_platform/detect.py`, `src/vision_platform/track.py`, `src/vision_platform/sources.py`
- Modify: `models/README.md` (YOLOX entry)
- Test: `tests/test_track.py`

**Interfaces:**
- Produces: `Detector` protocol `detect(frame_bgr) -> list[Detection]`; `YoloxOnnxDetector(model_path, providers: list[str], score_threshold: float = 0.4, person_class_id: int = 0)` — ONNX Runtime session, 640×640 letterbox in, NMS via `cv2.dnn.NMSBoxes`; missing `model_path` raises `FileNotFoundError` naming the path at construction. `FakeDetector(script: list[list[Detection]])` for tests.
- Produces: `ByteTrackAdapter(camera_id: str)` with `update(detections: list[Detection], ts: float, frame_size: tuple[int,int]) -> list[TrackObservation]` wrapping `supervision.ByteTrack`.
- Produces: `VideoFileSource(path: Path, analytics_fps: float)` iterable of `(ts: float, frame_bgr)`; raises `FileNotFoundError`/`ValueError("cannot open video: <path>")` at construction; skips frames to honor `analytics_fps` using the container FPS from `cv2.VideoCapture`.

- [ ] **Step 1: Write failing tests** — `FakeDetector` feeding one moving box for 10 frames yields a single stable `track_id` from `ByteTrackAdapter`; two crossing boxes yield two ids; `VideoFileSource` on a nonexistent path raises before yielding.
- [ ] **Step 2: Run** `uv run pytest tests/test_track.py -v` — expected FAIL.
- [ ] **Step 3: Implement** the three modules. Add the YOLOX ONNX entry (OpenCV Zoo `object_detection_yolox`, Apache-2.0, sha256, fetch command) to `models/README.md`.
- [ ] **Step 4: Run tests, expect PASS. Commit:** `feat: YOLOX person detection with ByteTrack adapter and file source`.

### Task 7: Outbox + emitters, end-to-end runner, and CLI (`vpoc run` / `vpoc enroll` / `vpoc flush`)

**Files:**
- Create: `src/vision_platform/emit.py`
- Modify: `src/vision_platform/pipeline.py` (add `PipelineRunner`), `src/vision_platform/cli.py`
- Test: `tests/test_runner_e2e.py`, `tests/test_emit.py`

**Interfaces:**
- Produces: `to_payload(event: MachineEvent, machine_id: str, pipeline: str) -> dict` implementing the Scope Boundary envelope exactly (`schema: "vision.event/1"`).
- Produces: `Outbox(db_path: Path)` (stdlib `sqlite3`, table `events(event_id TEXT PRIMARY KEY, payload TEXT, status TEXT, attempts INT, created REAL, sent REAL)`) with `put(event, machine_id, pipeline) -> None` (idempotent on `event_id`), `pending(limit: int) -> list[dict]`, `mark_sent(event_id)`, `mark_failed(event_id)`.
- Produces: `WebhookEmitter(outbox, url: str, timeout_s: float = 5.0, post: Callable | None = None)` with `emit(event)` = `outbox.put` then `flush()`, and `flush() -> int` POSTing pending payloads via stdlib `urllib` (injectable `post` for tests; no `requests`/`httpx` dependency), marking sent only on 2xx. `FileEmitter(outbox, path)` appends the same payload dict as JSONL — the POC default on this PC.
- Consumes: Tasks 3–6. Produces: `PipelineRunner(spec: PipelineSpec, detector, encoder, gallery, emitter)` with `run(source) -> RunStats(frames: int, events: int)`. Per track, the runner keeps the best-`quality` face embedding seen so far and attaches `gallery.identify(...)` result to observations before the event plane; a track with no face reaches the emitter with `identity=None`.
- Teacher-reconciliation support (SOP itself is downstream, out of scope): unresolved entries carry `data.unresolved_reason` (`no_face_detected | face_too_small | poor_quality | below_threshold` — derived from whether any `FaceHit` was seen, its size/quality, and the identify result); the runner remembers each track's emitted entry `event_id` and, if identity resolves later while the track is alive (slow-cadence retry of the best embedding, POC: re-evaluate on each new best-quality embedding), emits `identity_resolved` with the same `track_id` and `data.entry_event_id` pointing at the entry. If the resolved identity is already bound to another live track, add `data.duplicate_identity: true` instead of rebinding silently.
- CLI: `vpoc run --pipeline pipelines/attendance_poc.yaml --source video.mp4 --outbox out/outbox.db [--webhook URL | --events-file out/events.jsonl] [--gallery out/gallery.npz]`; `vpoc flush --outbox out/outbox.db --webhook URL` re-sends pending payloads (the site recovery path, also run periodically by `vpoc run`); `vpoc enroll --gallery out/gallery.npz --identity alice --photos a.jpg b.jpg` refuses any photo with zero or multiple faces (named error, gallery file untouched).

- [ ] **Step 1: Write failing emitter tests** (`test_emit.py`, fake `post` callable, no network): payload envelope matches the contract field-for-field; `flush` with `post` returning 500 leaves the row pending, a second `flush` returning 200 sends the *same* `event_id`; duplicate `put` of one `event_id` stores one row; a fresh `Outbox` on the same file still sees unsent rows (restart survival).
- [ ] **Step 2: Write the failing e2e test** with `FakeDetector` + `FakeEncoder` + synthetic source frames (blank arrays) + `FileEmitter`: a scripted person walking A→TH→B whose face identifies as `"alice"` produces exactly one JSONL payload with `type=person_entered, identity=alice`; the same walk with `FakeEncoder` returning no hits produces one payload with `identity: null` and `data.unresolved_reason=no_face_detected`; a third scripted walk whose encoder returns no hit at entry but an `"alice"` hit 10 s later produces the unresolved entry first, then an `identity_resolved` payload whose `data.entry_event_id` equals the entry payload's `event_id`.
- [ ] **Step 3: Run** `uv run pytest tests/test_emit.py tests/test_runner_e2e.py -v` — expected FAIL.
- [ ] **Step 4: Implement** `emit.py`, `PipelineRunner`, and the three CLI subcommands. Validate pipeline and check every referenced file exists before opening the outbox.
- [ ] **Step 5: Run the full suite** `uv run pytest -v` — expected PASS (model tests may SKIP). Then run the CLI on any sample clip on this PC if one is available; otherwise record in README that real-footage proof awaits a user-supplied clip. **Commit:** `feat: store-and-forward webhook emitter with end-to-end runner`.

### Task 8: Bench harness and cameras-per-box estimator

**Files:**
- Create: `src/vision_platform/bench.py`
- Modify: `src/vision_platform/pipeline.py` (optional `LatencyRecorder` hook), `src/vision_platform/cli.py` (`vpoc bench`)
- Test: `tests/test_bench.py`

**Interfaces:**
- Produces: `LatencyRecorder` with `record(node: str, ms: float)` and `summary() -> dict[str, dict]` (count, mean, p95 per node). `estimate_cameras_per_box(summary, analytics_fps: float, face_cameras_fraction: float, accelerator_speedup: float = 1.0) -> dict` returning `cameras`, `bottleneck_node`, and `assumptions: list[str]` (speedup is an explicit, default-1.0 input — the estimator never invents a Jetson number).
- CLI: `vpoc bench --pipeline … --source video.mp4 --frames 300` prints the latency table and the estimate at `speedup=1.0` labeled "this PC, CPU".

- [ ] **Step 1: Write failing tests** — hand-built summary (detect mean 100 ms, others ~0) at 5 fps estimates 2 cameras with bottleneck `detect.persons`; p95 math on a known sample list.
- [ ] **Step 2: Run** `uv run pytest tests/test_bench.py -v` — expected FAIL.
- [ ] **Step 3: Implement** recorder + estimator + CLI wiring (runner wraps each node call with `time.perf_counter()` only when a recorder is passed).
- [ ] **Step 4: Run tests, expect PASS. Commit:** `feat: per-node latency bench with cameras-per-box estimator`.

### Task 9: Jetson portability pack and README

**Files:**
- Create: `Dockerfile.cpu`, `Dockerfile.jetson`, `README.md`
- Modify: `docs/2026-11-01-cv-platform-and-face-encoding-research.md` (link this plan + POC results once run)

**Interfaces:**
- Produces: two Dockerfiles that differ only in base image and default `runtime.providers` note (Jetson file comments pin the JetPack/L4T base to fill in once the exact Jetson model is confirmed — do not guess a tag); README with fresh-shell run instructions against the project environment and an explicit "POC proves / does not prove" list.

- [ ] **Step 1: Write** `Dockerfile.cpu` (`python:3.12-slim`, installs uv + `uv sync --locked`, entrypoint `vpoc`). Build it only if a Docker daemon is reachable; on this PC the sandbox cannot reach the daemon, so a failed build for that reason is recorded in README as unverified, not silently skipped.
- [ ] **Step 2: Write** `Dockerfile.jetson` as the same layers over a `# TODO(site): pin L4T base for <Jetson model>` placeholder plus provider instructions (`TensorrtExecutionProvider`, then `CUDAExecutionProvider`).
- [ ] **Step 3: Write README:** setup (`uv venv --python 3.12 && uv sync`), model fetch (pointer to `models/README.md`), enroll + run + bench examples, POC scope (file source, CPU, SFace baseline), known limits (ID-switch duplicates for unknown faces, uncalibrated threshold 0.5, no RTSP/multi-camera yet, 20 cameras/box unproven until Jetson bench).
- [ ] **Step 4: Final verification:** `uv run pytest -v` from a fresh shell in `vision-platform/`; paste the result summary into the research doc update. **Commit:** `docs: POC portability pack and README`.

## POC Acceptance (all on this PC)

1. `uv run pytest -v` green; model-marked tests pass if weights were fetched, else skip with printed reason.
2. `vpoc run` on a video file emits debounced composed-event payloads (`vision.event/1`) through the outbox to the file emitter or a webhook; the synthetic e2e test proves A→TH→B, timeout, reverse-direction, flicker, and unknown-face behavior without any model, and `test_emit` proves failed sends stay pending and are resent with stable `event_id`s.
3. `vpoc enroll` + `vpoc run --gallery` attaches identities from SFace embeddings; cross-model gallery use raises.
4. `vpoc bench` prints per-node mean/p95 on this PC and a cameras-per-box estimate whose assumptions are printed with it.
5. Nothing in the repo imports InsightFace or Ultralytics; `models/README.md` lists license + sha256 for every weight file used.
6. Generality proof: a second, non-attendance pipeline document (e.g. `pipelines/dwell_alert.yaml` — person dwells in a zone beyond a threshold, no face nodes) validates and runs on synthetic tracks through the identical runner and emits valid `vision.event/1` payloads. Attendance appears only in example YAML config and docs, never in core code paths (`grep -ri attendance src/` returns nothing).