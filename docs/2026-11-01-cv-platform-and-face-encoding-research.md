# Computer-Vision Platform Core + Face Encoding — Research Report

Date: 2026-11-01
Status: Research complete. No product code written yet — architecture decisions at the end need your confirmation.

## Executive summary

1. **Face encoding is a solved pattern with a clear winner family.** Detect a face, align it to a
   canonical 112×112 crop from 5 landmarks, run a deep network trained with a margin-based
   softmax loss (ArcFace / AdaFace family), and output an L2-normalized 512-dimensional vector.
   Identity comparison is then just cosine similarity between vectors. Nobody serious trains a
   classifier per person anymore, and nobody serious should start from FaceNet/dlib in 2026.
2. **Most capable openly available technique today: AdaFace** (quality-adaptive margin), with a
   ResNet-100 or ViT backbone trained on WebFace12M, as distributed by the CVLface project (the
   AdaFace authors' successor repo). ArcFace ResNet-100 trained on Glint360K (InsightFace
   `antelopev2`) is a close second and the most battle-tested in production.
3. **The biggest trap is licensing, not accuracy.** InsightFace code is MIT, but its pretrained
   model packs (`buffalo_l`, `antelopev2`, …) are non-commercial research-only. Ultralytics YOLO
   is AGPL-3.0. If this platform will ever be sold, those two facts decide your model choices
   more than any benchmark does. Commercial-safe fallbacks exist (OpenCV YuNet + SFace, RF-DETR)
   at a modest accuracy cost, or you train/license models yourself.
4. **Your A→B attendance idea is not a model — it is event logic on top of tracking.** The right
   core separates three planes: a heavy per-frame plane (decode → detect → track → embed),
   a cheap event plane (per-track state machines over zones/lines/time), and a business plane
   (attendance, intrusion, loitering rules that consume events). Users compose pipelines mostly
   in the event plane; the frame plane is a fixed catalog of typed nodes.
5. **Recommended core v1:** Python, declarative YAML/JSON pipeline DAG with typed node ports,
   ONNX Runtime as the single inference engine (CUDA/TensorRT/OpenVINO execution providers
   underneath), ByteTrack/BoT-SORT for tracking, supervision-style zone primitives, embeddings
   in PostgreSQL + pgvector (FAISS/Qdrant only when galleries grow past ~100k faces).
   Do not start with NVIDIA DeepStream/GStreamer; add it later only if many-camera throughput
   on NVIDIA hardware becomes the bottleneck.

---

## Part 1 — Face encoding (face picture → vector)

### 1.1 How modern face encoding works

The standard pipeline, used by every competitive system:

```
image/frame
  → face detector            bbox + 5 landmarks (both eyes, nose, both mouth corners)
  → similarity alignment     warp to canonical 112×112 using a fixed landmark template
  → recognition network      backbone (ResNet-50/100, MobileFaceNet, EdgeNeXt, ViT)
  → embedding                512-d vector (128-d for some lightweight models), L2-normalized
  → matching                 cosine similarity = dot product of normalized vectors
```

Key facts that shape the core design:

- **Alignment is not optional.** Skipping the 5-point warp or using the wrong normalization
  (e.g. `(RGB-127.5)/127.5` vs `/128`) silently destroys accuracy while still producing
  plausible-looking vectors. Alignment + preprocessing belong inside the encoder node, never
  left to pipeline users.
- **The embedding is a metric space, not a classifier.** Enrolling a new person = storing one
  or more vectors. No retraining. Verification (1:1, "is this Alice?") and identification
  (1:N, "who is this?") are both nearest-neighbor lookups with a threshold; below threshold
  the correct answer is "unknown" (open-set recognition).
- **Thresholds are deployment-specific.** A cosine threshold around 0.4–0.6 (model-dependent)
  is typical, but the right value must be calibrated on your own cameras at your target false
  accept rate. Any core must expose threshold + calibration tooling, not hard-code a number.
- **Template quality beats model micro-differences.** Enrolling with several good photos and
  averaging their (quality-weighted) embeddings into one template per person improves real
  accuracy more than switching between top-3 models. The core should support multi-photo
  enrollment and template aggregation from day one.

### 1.2 Why the ArcFace/AdaFace family won

Loss-function lineage:

- **Triplet loss (FaceNet, 2015; dlib's 128-d model):** historically important, now obsolete —
  slow to train, sensitive to sample mining, clearly beaten on hard benchmarks.
- **Margin softmax (CosFace 2018, ArcFace 2019):** add an angular/cosine margin between classes
  on a hypersphere. ArcFace's additive angular margin became the production standard; most
  deployed systems in 2026 still use it.
- **MagFace (2021):** makes embedding magnitude encode face quality — useful signal, since
  vector norm can gate low-quality faces.
- **AdaFace (CVPR 2022, Kim et al.):** adapts the margin to estimated image quality — emphasize
  hard samples when quality is high, ignore unidentifiable low-quality samples instead of
  forcing them apart. This is exactly the CCTV/attendance regime (mixed-quality faces), and it
  shows: AdaFace reduced errors vs MagFace by ~21% (IJB-B) and ~23% (IJB-C) relatively, and
  leads on IJB-S / TinyFace low-quality sets.
- **Later work (ElasticFace, UniFace, TransFace, EdgeFace, DArFace…):** incremental gains,
  mostly on specific regimes (edge size, very low resolution). The 2024–2026 practice is
  "AdaFace/ArcFace loss + stronger backbone (ViT with KP-RPE in CVLface) + bigger cleaner data
  (WebFace12M / Glint360K)" rather than a new loss.

**Benchmark literacy:** LFW is saturated (~99.8% for everything) and proves almost nothing.
Trust instead: **IJB-B / IJB-C** TAR@FAR=1e-4 (mixed-quality verification), **IJB-S / TinyFace**
(surveillance/low-res), and **NIST FRTE** (the authoritative vendor evaluation; top commercial
vendors — IDEMIA, NEC, SenseTime, ROC — reach ~0.1% FNIR on mugshot 1:N, which also shows the
gap between the best commercial engines and open weights on hard data). Expect a large
real-world drop on distant CCTV faces vs cooperative enrollment photos regardless of model;
camera placement (choke-point, face height, lighting) matters as much as the encoder.

### 1.3 Candidate encoders compared

| Model | Backbone / training data | Dim | Standing | License reality |
|---|---|---|---|---|
| AdaFace (CVLface) | ResNet-100 or ViT-KP-RPE / WebFace12M | 512 | Best openly documented mixed/low-quality results; IJB-C TAR@FAR=1e-4 ≈ 97.7% (IR-101) | Code open; check weight/dataset terms per release before commercial ship |
| ArcFace `antelopev2` (InsightFace) | ResNet-100 / Glint360K | 512 | Highest-accuracy InsightFace pack, very widely deployed | **Weights non-commercial research-only; commercial license sold by InsightFace** |
| ArcFace `buffalo_l` (InsightFace) | ResNet-50 / WebFace600K | 512 | Default InsightFace pack, LFW ~99.8% | Same non-commercial weight restriction |
| EdgeFace (-XS/-S) | EdgeNeXt hybrid + LoRA / WebFace | 512 | Best tiny/edge option: 1.77M-param variant scores LFW 99.73%, IJB-C 94.85% | Open research release; verify weight terms |
| SFace (OpenCV Zoo) | Lightweight CNN | 128 | A notch below ArcFace-R50, LFW ~99.3–99.6%, ample for cooperative attendance | **Apache-2.0 weights — cleanest commercial story** |
| FaceNet / dlib `face_recognition` | Inception / ResNet-29 | 128 | Obsolete baseline; do not build on it | Permissive, but accuracy not competitive |
| Commercial NIST leaders (IDEMIA, NEC, ROC…) | Proprietary | 512+ | Top FRTE scores, support, legal indemnity | Paid SDK/API |

**Recommendation, by deployment tier:**

- **Default server encoder:** AdaFace ResNet-100 (WebFace12M) via CVLface/UniFace ONNX export —
  best low-quality behavior for camera faces, 512-d, standard alignment, swappable later
  because the core treats the encoder as a versioned node (see §2.4: embeddings from different
  models are *not* comparable; store `model_id` with every vector and re-enroll on model change).
- **Edge / CPU-only / many cameras:** EdgeFace-XS or MobileFaceNet-class ArcFace; accept the
  IJB-C drop in exchange for ~10–30 ms/face on modest hardware.
- **If commercial licensing must be clean with zero negotiation:** YuNet + SFace now, and budget
  later for either an InsightFace commercial license or training AdaFace yourself on a dataset
  you have rights to. Do not quietly ship `buffalo_l` in a paid product.
- **InsightFace is still the best prototyping vehicle** (detector + landmarks + encoder in one
  ONNX pack). Use it for the spike/MVP under research terms, behind the encoder-node interface,
  so swapping the production encoder is a config change, not a rewrite.

### 1.4 Face detection (the other half of encoding)

The detector must return 5 landmarks, not just boxes, or alignment quality collapses.

| Detector | Standing | Notes |
|---|---|---|
| SCRFD-10GF (InsightFace) | Best accuracy/speed balance; WIDER FACE AP ≈ 95.2/93.9/83.1 (easy/med/hard) | Default in InsightFace packs; weight license follows the pack restriction |
| RetinaFace (ResNet-50 / MobileNet) | Max benchmark accuracy lineage, heavier | Good when GPU is plentiful; MobileNet variant for edge |
| YuNet (OpenCV Zoo) | Tiny (~75–230 KB), fast on CPU, a step below SCRFD on hard/small faces | MIT weights — commercial-safe default |
| BlazeFace / MediaPipe | Very fast, front-facing/selfie-oriented | Weak for CCTV angles and small faces; not recommended as core detector |
| MTCNN | Obsolete | Do not use |

Practical rule: faces smaller than ~40–60 px between the eyes are usually not reliably
identifiable no matter the encoder. The core should compute a per-face quality/size gate and
only encode faces that pass it — this also saves most encoder compute in wide shots.

### 1.5 Quality, liveness, and matching infrastructure

- **Face quality assessment:** gate enrollment and matching by quality (blur, pose yaw/pitch,
  size, occlusion). Options: MagFace-style embedding magnitude, SER-FIQ, CR-FIQA. Minimal
  viable version: detector score + face size + yaw estimate + embedding norm. Without a
  quality gate, one bad enrollment photo poisons an identity.
- **Anti-spoofing / liveness (PAD):** attendance is the classic photo-spoof target ("buddy
  punching" with a phone photo). Passive single-frame PAD with MiniFASNet (Silent-Face-
  Anti-Spoofing, Apache-2.0, ~3.6 ms/face) is the standard open option and should be an
  optional node in front of matching. It reduces casual spoofing; it is not certified
  high-assurance liveness (that needs depth/IR hardware or challenge–response).
- **Vector search tiers:** exact brute-force cosine in NumPy is fine to ~10k identities and
  simplest to make correct; FAISS in-process to ~1M vectors; PostgreSQL + pgvector (HNSW)
  when embeddings must live with relational data and transactions; Qdrant/Milvus when you
  reach multi-million galleries or need filtered ANN as a service. Start with pgvector or
  FAISS-behind-an-interface; the matching node should not know which.
- **Privacy by design:** store templates (vectors), not face photos, where possible; encrypt
  templates at rest; support deletion/re-enrollment; log matching decisions with score +
  model version for audit. Biometric templates are still personal data — see Part 3.

---

## Part 2 — Composable pipeline platform core

### 2.1 What "compose a pipeline" must mean

Your example decomposes cleanly, and the decomposition is the architecture:

> "When a person moves from marked area A to marked area B, treat it as an entry and record attendance"

- *person* → object detector + tracker (persistent `track_id` across frames)
- *marked area A / B* → user-drawn polygons in normalized camera coordinates
- *moves from A to B* → per-track state machine over zone membership history
  (e.g. `IN_A → (nowhere|threshold) → IN_B` within a time window, using the track's foot point)
- *entry / attendance* → a business event emitted once, debounced, enriched with identity
  if a face on that track was recognized, then sent to a sink (DB row, webhook, message bus)

None of that is a neural network except the first bullet. This is why successful platforms
(Roboflow Workflows, NVIDIA DeepStream, Intel DL Streamer, MediaPipe graphs, PeekingDuck)
all converge on the same shape: **a DAG of typed blocks**, compiled/validated up front, with
stateful blocks allowed (trackers, zone state machines, counters). Learn from them:

| Platform | Model | Lesson for your core |
|---|---|---|
| NVIDIA DeepStream | GStreamer C++ plugins, TensorRT, multi-stream batching | Fastest on NVIDIA; steep complexity, vendor lock-in. Adopt later as an execution backend, not as the authoring model |
| Intel DL Streamer | GStreamer + OpenVINO elements | Same lesson, Intel-flavored |
| Roboflow Inference / Workflows | Python blocks, JSON workflow → DAG, typed kinds, visual editor | Best authoring model to copy: versioned blocks with manifests, typed inputs/outputs, per-block config |
| MediaPipe | C++/Python calculators in a graph, timestamped packets | Strong stream/timestamp semantics; heavy to extend for end users |
| PeekingDuck (AI Singapore) | Python nodes, YAML pipeline | Proof a small YAML node system is enough for many analytics; limited perf/type safety |
| Frigate / Viseron | Opinionated NVR analytics, zones per camera | Proof users think in "camera + zones + events", not graphs — hide the DAG behind use-case templates |

### 2.2 Proposed core: three planes

```
FRAME PLANE (heavy, per frame, GPU)        EVENT PLANE (cheap, per track event)     BUSINESS PLANE
┌─────────┐  ┌──────────┐  ┌─────────┐    ┌──────────────┐  ┌─────────────┐        ┌────────────┐
│ Source  │→ │ Detector │→ │ Tracker │ →  │ Zone / Line  │→ │ Sequence /  │  →     │ Attendance │
│ RTSP/file│  │ person…  │  │ ByteTrk │    │ state machines│  │ Dwell /     │  event │ Intrusion  │
└─────────┘  └──────────┘  └────┬────┘    └──────────────┘  │ Debounce    │        │ Loitering  │
                                │ crops                     └─────────────┘        └─────┬──────┘
                     ┌──────────▼───────────┐                                             │ sinks
                     │ FaceDetect→Embed→Match│  identity bound to track_id               ▼
                     └──────────────────────┘                                     DB / webhook / MQTT
```

Design rules that follow from this split:

1. **Frames never leave the frame plane.** Only structured observations cross into the event
   plane: `{camera_id, ts, track_id, class, bbox, foot_point, zone_ids[], identity?, score?}`.
   This keeps the event plane testable with synthetic track feeds (no video needed in unit tests).
2. **Identity binds to a track, not a frame.** A track accumulates face observations over its
   life; the best-quality embedding wins, with a per-track identity decision (and hysteresis so
   identity does not flicker frame to frame).
3. **Every event is emitted once, with evidence.** `{event_id, type, ts, camera, track, identity,
   from_zone, to_zone, snapshot_ref, pipeline_version}` + idempotency/cooldown keys so a person
   lingering on a boundary produces one attendance record, not forty.
4. **Pipelines are data, not code.** A versioned YAML/JSON document, schema-validated and
   DAG-checked (types, cycles, unknown nodes) before it can run. The visual editor you build
   later is just an editor for this document; the core must not depend on any UI.

### 2.3 Node catalog for v1

Frame plane: `source.rtsp|file|webcam`, `detect.objects` (person/vehicle/…),
`track.bytetrack|botsort`, `face.detect`, `face.embed`, `face.match` (1:N against gallery),
`face.pad` (optional liveness), `crop`, `annotate`/`sink.video` (debugging).

Event plane: `zone.membership` (polygon, foot-point), `line.crossing` (directed segment),
`sequence` (ordered zone visits within window — your A→B), `dwell` (time-in-zone ≥ T),
`count.in_zone`, `debounce`/`cooldown`, `schedule.gate` (only 07:00–09:00), `identity.gate`
(only recognized / only unknown), `emit.event`.

Business plane (templates composed from the above, shipped as presets):
`attendance.entry_exit` (A→B = in, B→A = out), `security.intrusion`, `safety.loitering`,
`retail.footfall`. Users start from a template and tweak zones/thresholds — most will never
hand-assemble raw nodes, and the core should be designed for that reality.

Each node declares a manifest: `name@version`, typed input/output ports
(`Frame`, `Detections`, `Tracks`, `FaceEmbeddings`, `TrackEvents`, `BusinessEvents`),
a JSON-Schema for its config, whether it is stateful, and its resource needs (GPU, model file).
That manifest is what makes pipelines composable, validatable, and later editable in a UI.

### 2.4 Detection and tracking choices

- **Person/object detector:** YOLO11-class and RT/RF-DETR-class models are the current
  real-time frontier (RF-DETR reports >60 AP on COCO, first real-time model to do so; YOLO11/26
  remain the ecosystem default). **License fork in the road:** Ultralytics YOLO is AGPL-3.0
  (commercial license available); RF-DETR is Apache-2.0. The detector sits behind the same
  node interface either way — pick per your commercial answer, export to ONNX.
- **Tracker:** ByteTrack as default (motion-only, fast, best quality/speed balance, ~80 MOTA
  on MOT17); BoT-SORT (with camera-motion compensation, optional Re-ID embeddings) when
  scenes are crowded/occluded or cameras move; OC-SORT/Deep OC-SORT as alternatives. Do not
  build on DeepSORT in 2026. Tracking quality, more than detector AP, decides whether A→B
  logic works — every ID switch inside the A→B path is a missed or duplicated attendance.
- **Person Re-ID** (appearance embedding for cross-camera identity without faces) is a later
  node, not v1: useful, but it multiplies gallery/privacy complexity.
- **Embeddings are model-bound.** Vectors from encoder v1 cannot be matched against v2.
  Store `model_id + model_version` on every template; plan re-enrollment tooling.

### 2.5 Runtime and stack recommendation

- **Language/runtime for the core: Python first.** Fastest ecosystem access (ONNX Runtime,
  supervision-style geometry, tracker implementations), and the frame plane is GPU-bound, not
  Python-bound, at few-camera scale. Isolate the hot path so nodes can later be re-hosted in
  GStreamer/DeepStream or Rust/C++ without changing pipeline documents.
- **One inference engine: ONNX Runtime** with execution providers (CUDA/TensorRT on NVIDIA,
  OpenVINO on Intel CPU, CoreML/DirectML elsewhere). Every model ships as ONNX. This is the
  portability layer that keeps you off any single vendor's SDK.
- **Stream handling:** per-camera bounded queues with frame-drop (latest-frame-wins) —
  analytics must degrade by dropping frames, never by accumulating latency. Decode via
  FFmpeg/OpenCV first; hardware decode (NVDEC) when camera count grows.
- **State:** pipeline definitions + zones + identities + events in PostgreSQL (pgvector for
  embeddings); snapshots/clips in object/file storage; events out via webhook + a message
  bus option (MQTT/NATS/Kafka depending on your deployment answer).
- **APIs:** REST (or gRPC) for pipeline CRUD/deploy, enrollment, and event query; WebSocket/SSE
  for live events. Zone drawing is a frontend concern that produces normalized polygons in
  pipeline config.
- **Observability:** per-node latency/FPS/queue metrics and per-event audit log from day one;
  a CV platform you cannot measure is a demo, not a platform.

### 2.6 Your attendance scenario, end to end (worked example)

Pipeline preset `attendance.entry_exit` on `camera: lobby`:

1. `source.rtsp` → `detect.objects(classes=[person])` → `track.bytetrack`
2. In parallel per track: crop head/upper-body region → `face.detect` → quality gate →
   `face.embed(AdaFace)` → best-embedding-per-track → `face.match(gallery=employees,
   threshold=calibrated)` → identity bound to `track_id` (may stay `unknown`)
3. `zone.membership(zones: A=outside, TH=doorway, B=inside, anchor=foot_point)` — three zones
   beat two: the threshold band absorbs boundary jitter that causes A↔B flicker
4. `sequence(A → TH → B, within=30s)` emits `entry`; `sequence(B → TH → A)` emits `exit`
5. `debounce(identity_or_track, cooldown=5min)` → `emit.event(attendance)` →
   sink: DB insert + webhook to HR system, with snapshot evidence

Failure modes this design explicitly handles: track ID switch mid-doorway (mitigate with
BoT-SORT-ReID + threshold zone + identity-based dedup window), face never visible (emit
`unknown entry` event for review instead of dropping it), tailgating (two tracks, one identity —
count and flag), boundary loitering (hysteresis + cooldown), and photo spoofing (optional PAD node).

### 2.7 Phased roadmap

- **Spike (throwaway, ~days):** one video file → InsightFace pack + YOLO/ByteTrack + two
  polygons + A→B state machine → printed attendance events. Purpose: feel the real accuracy
  of detection/tracking/encoding on *your* footage before committing to the design.
- **MVP core:** pipeline document + validator + runner; nodes in §2.3; enrollment API;
  pgvector matching; events to DB/webhook; one preset (`attendance.entry_exit`); metrics.
- **v1 platform:** multi-camera scheduling, pipeline versioning/deploy, zone editor API,
  identity/track review UI hooks, PAD node, threshold-calibration tool, retention/privacy controls.
- **Scale phase (only if needed):** DeepStream/GStreamer execution backend, FAISS/Qdrant
  gallery service, cross-camera Re-ID, edge deployment packaging.

---

## Part 3 — Legal and ethical constraints (design inputs, not afterthoughts)

- Face templates used to identify people are **special-category biometric data under GDPR
  (Art. 4(14), 9)** and similar regimes (e.g. Illinois BIPA written consent). Workplace
  attendance is one of the most scrutinized uses; regulators (e.g. UK ICO vs Serco, 2024) have
  ordered employers to stop systems where necessity/proportionality was not shown.
- Under the **EU AI Act**, remote biometric identification is high-risk (with phased
  obligations), real-time remote biometric ID by law enforcement is largely prohibited, and
  **workplace emotion recognition has been banned since 2 Feb 2025** — keep emotion inference
  out of the node catalog entirely.
- Core features this forces: explicit consent/enrollment records, purpose binding per gallery,
  retention limits + automatic deletion, template encryption, subject deletion/export,
  human-review path for disputed matches, and a non-biometric fallback (card/PIN) so
  attendance never *requires* a face. Jurisdiction varies — get local legal review before
  production; this report is engineering guidance, not legal advice.

---

## Decisions recorded 2026-11-01

- **Use type: product to sell.** Shipped models must be commercially clean. InsightFace packs
  (`buffalo_l`/`antelopev2`) and Ultralytics YOLO (AGPL-3.0) are prototype-only unless a
  commercial license is bought. Default shipped stack becomes: RF-DETR (Apache-2.0) for
  person/object detection, YuNet (MIT) + SFace (Apache-2.0) as the clean face baseline, with a
  costed upgrade path to (a) a commercial InsightFace license or (b) a self-trained AdaFace
  encoder on data we have rights to. The core needs a model-provenance registry: every model
  file records source, license, and allowed use, checked at pipeline-deploy time.
- **Deployment: edge box per site.** The core must run offline-first on one box per site:
  local gallery, local event store with store-and-forward sync, no cloud dependency in the
  hot path. Prefer SQLite + in-process FAISS (or sqlite-vec) on the box over PostgreSQL;
  keep PostgreSQL/pgvector for an optional central server that aggregates sites.
- **Hardware: NVIDIA GPU.** On an edge box this means Jetson Orin-class or a small x86+RTX
  server. Use ONNX Runtime with TensorRT/CUDA execution providers, FP16 TensorRT engines,
  hardware decode where available. Full AdaFace R100-class encoders are viable on Orin NX/AGX;
  EdgeFace-class encoders for Orin Nano or high camera counts.

Still open: first flagship use case order (attendance preset first assumed), target cameras
per box, and whether a central multi-site server is in scope for v1.

### Update 2026-11-01 — dev machine vs site

- **Site:** 60 cameras, 3 Jetsons → 20 cameras per Jetson. Jetson model still to be confirmed.
  At 20/box, every camera running detection+tracking+face at full frame rate is not feasible.
  The core therefore needs tiered analytics from the start: low-FPS person detection/tracking
  on all cameras, face encoding only on designated choke-point cameras and only when a track
  is inside a relevant zone, plus per-node latency budgets so cameras-per-box is measured, not guessed.
- **Dev machine (this PC, inspected):** x86_64, 8 cores, 30 GiB RAM, Intel Iris Xe iGPU,
  **no NVIDIA GPU**, system Python 3.14 only, no CV libraries, no webcam, ffmpeg present,
  Docker CLI present but daemon not reachable from the agent sandbox.
- **Consequence for the POC:** it must run CPU-first here (ONNX Runtime CPU, small models,
  1–2 file/RTSP streams at reduced analytics FPS) and move to Jetson later by switching the
  ONNX execution provider to TensorRT — same pipeline documents, same ONNX models. Performance
  on Jetson cannot be proven here; what the POC proves is pipeline composition, A→B event
  correctness, face enrollment/matching, and a per-node cost table used to size cameras/box.
- **Environment rule:** project-local `.venv` managed by uv with a pinned Python 3.12
  (uv fetches the interpreter), dependencies pinned to wheels available on both x86_64 and
  aarch64. Never the system Python 3.14, never `--break-system-packages`.
- **POC inputs:** video files and synthetic track feeds (unit tests for the event plane),
  since this PC has no camera. RTSP is a config change on the same source node, exercised at site.

### Update 2026-11-01 — machine-side scope and payload contract

- The project covers **the machine side only**: ingestion, object detection, tracking, face
  recognition against a local gallery, pipeline composition, metrics, and emitting payloads.
  Dashboards, attendance/HR rules, reporting, and any central system are downstream consumers,
  out of scope. The machine reports what it saw; downstream decides what it means.
- Payload transport (decided): **HTTP webhook POST**, contract defined by us. One JSON object
  per composed event (never per-frame detections), envelope `schema: "vision.event/1"` with
  `event_id` (uuid4, downstream dedup key), `type` from pipeline config, source-clock `ts`,
  `machine_id`, `camera_id`, `pipeline`, `track_id`, `identity`/`identity_score` (null when
  unknown — the event is still emitted), and `data` (e.g. `from_zone`, `to_zone`, `via`).
- Delivery (decided): **store-and-forward** via a local SQLite outbox; at-least-once with
  retry/backoff, marked sent only on HTTP 2xx, `event_id` stable across retries/restarts.
  A file emitter with the byte-identical payload is the POC default on the dev PC.

### Update 2026-11-01 — classroom operating mode (when does the pipeline "trigger"?)

- A deployed pipeline **runs continuously but in tiers**. Sampling + person detection +
  tracking run at a configured analytics FPS (classroom: ~1–5, not camera FPS), further
  reduced by a motion gate when the scene is static. This tier produces no payloads.
- **Face recognition is per-track, not per-frame.** Track lifecycle:
  `NEW → CAPTURING` (collect best-quality face embeddings for a few seconds)
  `→ IDENTIFIED` (identity locked to the track; encoding stops except a slow periodic
  re-check or after track loss) `→ LEFT`. A seated, identified student costs no face
  compute for the rest of the lesson. Encoding every face every frame
  (30 students × 5 FPS = 150 encodes/s/camera) is explicitly not the design.
- **Payloads fire only on state transitions**: track entered room zone, identity resolved,
  track left — debounced as before. Steady-state lesson = zero payloads except optional
  periodic occupancy/presence snapshots if downstream wants them.
- **Schedule gate:** pipelines can be armed only during class hours; outside the window the
  frame plane idles.
- **Classroom alternative — snapshot mode:** instead of (or besides) door transitions, a
  schedule/trigger starts a 30–60 s recognition burst over everyone in the room zone and
  emits one `presence_snapshot` payload (identified + unknown counts). This fits
  "attendance = seated at check time" better than A→B door logic.
- **Honest limit:** one wide camera will not reliably identify back rows (small, occluded,
  downcast faces). Reliable who-is-who needs a front camera facing students at reasonable
  distance and/or a door choke-point camera for identity, with the wide camera used for
  counts/presence. Counting present ≠ identifying present; the pipeline should say which
  it is doing per camera.
- Minors are a heightened-sensitivity biometric population in most jurisdictions and school
  policies; consent, retention, and a non-biometric fallback need institutional/legal review
  before any classroom deployment (see Part 3).

### Update 2026-11-01 — attendance = entered the room; teacher reconciliation SOP

- **Definition (decided):** attendance is based on the student **entering the room**
  (door-zone transition), not on a seated snapshot check. Snapshot mode remains an optional
  extra, not the attendance source.
- **SOP (downstream / human, out of machine scope):** the teacher sees a live counter
  comparing **counted** (persons who entered) vs **attended** (entries resolved to a known
  student identity). If counted > attended, some entrants are unresolved; the teacher asks
  them to make themselves clear and visible to the camera (face it for a few seconds,
  ideally at a marked identification spot near the door/front), and the counter reconciles.
- **What the machine must emit to make that SOP work (in scope):**
  1. `person_entered` for every entry track, identity attached if already known, else null.
  2. `identity_resolved` when an unresolved in-room track is later identified — same
     `track_id`, carrying the original entry `event_id` as `data.entry_event_id`, so
     downstream merges it into the existing attendance row instead of double-counting.
  3. `data.unresolved_reason` on unresolved entries from the quality gate:
     `no_face_detected | face_too_small | poor_quality | below_threshold`.
  4. Position context for the teacher (`data.zone`, last bbox or seat-area label) and an
     optional snapshot reference, subject to school privacy policy.
- **Retry rule (machine):** an unresolved track that is still in the room keeps retrying
  identification at a slow cadence and immediately when its face quality improves or it
  enters a designated identification zone. Once resolved, retries stop (Tier-2 lifecycle).
- **Counter semantics are downstream:** downstream defines the class session window and
  computes counted = entries − exits (or total entries) and attended = entries with an
  identity (including late `identity_resolved` merges). The machine knows no timetable,
  class list expectations, or lateness rules.
- Duplicate-identity guard stays downstream-visible: if one identity resolves on two live
  tracks, the second resolution carries `data.duplicate_identity: true` for teacher review
  rather than being silently accepted.

### Update 2026-11-01 — single wide camera per classroom: zone layout

Constraint (decided): **one wide camera watches the whole room.** No door camera.
Recommended hybrid, all zones are polygons in the one frame:

1. **`door` zone** — the doorway area inside the frame (or the frame edge where entrants
   first appear if the door leaf itself is out of view). A track *born* in `door` that then
   moves into `room` = `person_entered` (the count). Tracks born mid-room (occlusion
   splits, someone standing up from behind a desk) do not count as entries.
2. **`room` zone** — the seating interior. Identity resolution happens here
   opportunistically over time: seated students face the front camera for minutes, so the
   per-track best-face capture gets many chances without any student action.
3. **`id_spot` zone** — a small polygon at the front of the room, closest to the camera,
   matched by a physical floor marking (tape box) and a look-at cue near the camera.
   Only *unresolved* students are ever sent here by the teacher (SOP above). Entering
   `id_spot` forces an immediate identification retry at maximum face size.
4. **Exclusion / staff areas** (optional) — e.g. teacher desk area excluded from student
   entry counting; coarse area labels (`front`, `middle`, `back`, left/right) attached to
   unresolved entries so the teacher can locate the student. No per-seat polygons.

Supporting rules:

- **Never downscale away the faces:** person detection may run on a resized frame, but
  face detection/encoding must run on native-resolution head/upper-body crops (tiling).
  On a 1080p wide shot, a back-row face can be ~35–40 px wide — below reliable
  identification (~60–80 px minimum, 100+ preferred). Front/middle rows are usually fine;
  back rows are count-and-retry territory, resolved via `id_spot` when needed.
- Camera facts that must be verified per room before trusting identification rates:
  resolution (4K roughly doubles face pixels vs 1080p), mounting height/pitch (2.2–2.5 m
  front wall beats a high ceiling corner), whether the door is inside the frame, and a
  measured face-pixel count at the farthest seat. One sample frame from a real classroom
  camera settles all four.
- Physical marking is needed for **`id_spot` only** (floor tape). `door`/`room` are
  digital polygons; seating needs no marking.

### Update 2026-11-01 — measured findings from sample frame "Labkom 2" (29-10-2026)

Sample frame inspected at native 1920×1080 (kept outside the repo for student privacy;
annotated draft generated at `/tmp/labkom_zones.png`). Computer lab, ~30 cubicle seats,
camera high in a front corner with wide-angle distortion, bright even lighting.

- **Face sizes (visual estimates from 3× crops, to be detector-verified in the POC):**
  front row ≈ 40–55 px face width, middle ≈ 25–35 px, back rows ≈ 15–25 px. Even the front
  row is below the 60–80 px reliability floor.
- **Bigger problem is angle, not size:** steep top-down pitch plus students looking down at
  monitors. Seated identification from this camera will be unreliable for most seats, most
  of the time — regardless of encoder choice. Seated faces are a bonus signal.
- **Where identification can work:** standing students (head nearer the camera, face larger),
  especially (a) during the walk from door to seat — capture the best frame across the whole
  transit, (b) during a start-of-class collective "look up at the camera" moment of 5–10 s,
  which fixes the pitch problem for front/middle rows, and (c) at the front `id_spot`
  (≈120+ px face expected) for anyone still unresolved. Revised SOP order: walk-in capture →
  class look-up burst → `id_spot` only for the remainder. Never a whole-class spot queue.
- **Enrollment implication:** enroll each student with a frontal photo *plus* an in-situ
  capture from this camera at the `id_spot` looking up — same angle and lighting as
  operational captures; store both in the template (§1.1 multi-photo aggregation).
- **Detection implication:** seated students are heavily occluded by cubicle partitions and
  monitors; a COCO person detector may undercount them. POC must evaluate detection on this
  frame type; a head/shoulder detection node is the likely fallback for occupancy. Walking
  (entry) detection in the aisles is unobstructed and should be reliable.
- **Zone anchors:** foot-point works for walking tracks in aisles; seated tracks need a
  head/body-center anchor. `zone.membership` needs a per-pipeline anchor config.
- **Entry door (confirmed): the glass door at the back.** The black door far-left is not a
  student entry and is excluded from entry counting. Entrants therefore appear at maximum
  distance from the camera and walk toward it — best-face capture must span the whole
  door-to-seat transit, and students seated at the back have the shortest, farthest transit
  (they are the most likely `id_spot` candidates). Teacher desk bottom-left is an exclusion
  zone; OSD overlays (timestamp top-left, label bottom-right) should be masked in config.
  Updated annotated draft: `/tmp/labkom_zones_v2.png`.

### Update 2026-11-01 — fleet composition: classrooms are a minority of the 60 cameras

- The site's 60 cameras are the **entire** camera fleet (all buildings/purposes). Classroom
  cameras are **not even half** (< 30). Exact classroom count and the camera→Jetson
  assignment are still open (site survey input).
- Sizing consequence: the heavy pipeline (entry tracking + face identification) runs only
  on classroom cameras. Remaining cameras are candidates for light pipelines (person
  detection / zone events at low FPS with motion gating, no face encoding) or no analytics
  at all in v1. Average load per Jetson ≈ 20 cameras total, of which < 10 classroom —
  but only if assignment is balanced; a Jetson wired to a classroom-heavy area could carry
  more. Capacity planning must use the worst box, not the average.
- The synchronized peak still matters: all classrooms start together (~07:00), so entry
  tracking and identification bursts happen on every classroom camera at once. The bench
  estimator (POC Task 8) therefore reports both steady-state and entry-burst budgets, with
  `face_cameras_fraction` set from the real inventory once known (planning bound: < 0.5).

### Update 2026-11-01 — check-in vs check-out

- **Attendance as defined (student entered the room) requires check-in only.**
  `person_entered` (+ later `identity_resolved`) fully determines attended. No check-out
  activity, by student or system, is part of the attendance record.
- **Check-out (`person_left`, reverse sequence `room → door`) is a cheap optional machine
  capability** — same sequence node, reversed order, enabled per pipeline. Emit it only if
  a downstream purpose exists: live room occupancy (entered − left), early-leave/duration
  rules, evacuation muster lists, or room-turnover detection between back-to-back classes.
- Cautions: mass exit at class end is the hardest tracking moment (crowd occlusion), so
  exit counts will be less accurate than entry counts; a failed exit must never erase a
  valid check-in downstream. Exit/return trips mid-lesson (toilet) are noise for
  attendance — downstream dedups repeat entries by identity within its session window.
- Session separation does not require check-out: downstream timetable boundaries (or an
  optional machine `room_emptied` event) close a session. Collecting per-student exit
  times for minors without a stated purpose also fails data-minimization expectations.
- **Recommendation:** POC proves check-in end-to-end; the reverse sequence is covered by
  synthetic event-plane tests and ships as a config option (`person_left`, default off for
  classroom attendance pipelines).

### Update 2026-11-01 — product definition reaffirmed: a composable capability core

- The system is **not an attendance product**. It is a machine-side core to which
  computer-vision capabilities are added over time. Attendance/classroom is the first
  worked example and the POC proving ground, nothing more.
- Capability taxonomy for the registry: **sources** (file, RTSP), **perception**
  (person/object detection, face encode/match, later: pose, vehicle plate/OCR, PPE,
  attributes, segmentation), **association** (tracking, later cross-camera Re-ID),
  **activities** (zone sequence, dwell/loitering, line crossing, counting, absence,
  co-occurrence), **conditions** (zone, schedule, class, score/quality thresholds,
  identity known/unknown, boolean combination, debounce/cooldown), **emitters**
  (webhook, file, later MQTT/relay).
- Composition contract: pipelines are validated DAG documents over versioned node
  manifests with typed ports. Users compose registered capabilities + activities +
  conditions by config only — arbitrary user code is never executed on the machine.
  Adding a capability must not change the validator, runner, outbox, or payload schema;
  the POC proves this with a second non-attendance pipeline (dwell alert, no face nodes)
  running on the identical core.

### Update 2026-11-01 — production runtime strategy: Python throughout, swappable frame plane

- **Language decision (decided):** stay with Python for production too. (A Rust core was
  considered and set aside.) Model compute and video decode dominate cost, not host
  language, and both remaining backends are Python-native: ONNX Runtime directly, and
  DeepStream via NVIDIA's official Python bindings (`pyds`), where Python builds the
  GStreamer pipeline and probes metadata while decode/inference run in native code.
- **The swap boundary is the observation stream, not individual nodes.** Frame
  representations do not abstract cleanly across runtimes (CPU arrays here vs
  `NvBufSurface`/GStreamer buffers in DeepStream — zero-copy would be lost). So backends
  replace the whole *frame plane segment*: it consumes stream + frame-plane node configs
  and emits the standard `TrackObservation` stream. The event plane, validator, outbox,
  and payload contract are identical for every backend.
- **Runtime profiles:** `dev-cpu` (this PC: Python + ONNX Runtime CPU), `jetson-ort`
  (same nodes, TensorRT/CUDA execution providers — config-only switch), `jetson-deepstream`
  (frame-plane subgraph compiled to DeepStream elements: nvstreammux → nvinfer → nvtracker
  → secondary inference for faces → probe/appsink emitting observations). Node manifests
  declare `backends: [...]`; the validator rejects a pipeline on a profile that cannot
  run one of its nodes, instead of failing at runtime.
- **Production shape:** one Python codebase. The frame plane runs behind the observation
  boundary and is selected per machine by `runtime.profile`; event plane, validator,
  outbox and emitters are shared verbatim across profiles. Multi-camera scaling on the
  ORT profiles is process-per-pipeline-group (native inference releases the GIL); on
  DeepStream it is one batched multi-stream pipeline per box.
- **Honest abstraction assessment:**
  - *Event plane / payload / outbox:* ~100% portable. Pure Python, golden fixtures.
  - *`dev-cpu` → `jetson-ort`:* near drop-in. Real risks are packaging, not code:
    Jetson `onnxruntime-gpu` comes from NVIDIA/JetPack builds rather than plain PyPI,
    TensorRT engine compilation time on first run, and providers silently falling back
    to CPU — so startup must assert the *active* provider list, not the requested one.
  - *`jetson-ort` → `jetson-deepstream`:* a frame-plane translation project, not a
    switch. Detect+track maps well (generated `nvinfer` configs, `nvtracker` with
    NvDCF), but outputs differ in detail (NMS clustering, tracker ID behavior), so
    backend conformance tests compare observations within tolerance on golden clips.
  - *Weakest link — face alignment+encoding:* `nvinfer` cannot do 5-point alignment.
    Options are a custom C++ `nvdspreprocess` library (full-native, most work) or a
    hybrid probe: DeepStream detects, Python crops/aligns and runs the encoder via ORT
    only for unresolved tracks. The tiered face policy keeps that hybrid cost bounded,
    and it is the recommended first DeepStream form.
  - *Unverifiable from this PC:* DeepStream needs an NVIDIA GPU, so that backend can
    only be proven once a Jetson is in hand. Sequence the rollout accordingly:
    POC `dev-cpu` → production v1 `jetson-ort` (same code as POC) → `jetson-deepstream`
    only if bench numbers demand it.
- Golden synthetic fixtures (JSON tracks in → events out) remain required from day one —
  now as *backend parity* tests rather than language-port tests.

## Open decisions (needed before the design doc)

1. **Deployment shape:** single on-prem server for a handful of cameras, multi-site/cloud, or
   edge boxes per site? (Decides runtime, bus, gallery service.)
2. **Commercial intent:** internal tool, research, or a product you will sell/deploy for clients?
   (Decides InsightFace/YOLO licensing path vs SFace/RF-DETR/self-trained.)
3. **Hardware you can count on:** NVIDIA GPU server, Intel CPU-only, Jetson-class edge, or
   "must run on whatever the client has"? (Decides encoder size and execution providers.)
4. **First use case to prove:** is attendance (A→B + face identity) the flagship, with generic
   zone analytics second — or the reverse? (Decides MVP preset order.)

## Sources

- AdaFace paper (CVPR 2022): https://arxiv.org/abs/2204.00964 — quality-adaptive margin; IJB-B/C gains
- CVLface (AdaFace successor, ViT/KP-RPE + WebFace12M): https://github.com/mk-minchul/CVLface
- EdgeFace (IJCB EFaR winner, edge encoder): https://arxiv.org/abs/2307.01838
- InsightFace model licensing (packs non-commercial, commercial license offered): https://insightface.ai/
- OpenCV Zoo YuNet + SFace (MIT/Apache-2.0): https://github.com/opencv/opencv_zoo
- UniFace model benchmarks (SCRFD/RetinaFace WIDER FACE, AdaFace IR-101 IJB-C 97.66%): https://github.com/yakhyo/uniface
- NIST FRTE results coverage (IDEMIA/NEC/SenseTime/ROC top tier): https://www.biometricupdate.com/202608/nist-frte-1n-shows-face-recognition-race-shifting-beyond-accuracy
- Roboflow Workflows (block DAG authoring model): https://deepwiki.com/roboflow/inference/3-workflow-system
- NVIDIA DeepStream: https://developer.nvidia.com/deepstream-sdk
- Tracker comparisons (ByteTrack / BoT-SORT / OC-SORT, MOT17): https://trackers.roboflow.com
- RF-DETR (Apache-2.0, >60 COCO AP real-time): https://github.com/roboflow/rf-detr
- Silent-Face-Anti-Spoofing / MiniFASNet (Apache-2.0 PAD): https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- GDPR/EU AI Act biometric attendance analysis: https://staraudit.org/news/the-price-of-a-face

