# AGENTS.md — resume guide for coding agents (and humans)

This file is the entry point for continuing work on a fresh machine without
any chat history. It summarizes what this project is, every decision already
made, verified facts, and what to do next. Deeper detail lives in:

- `README.md` — user/developer guide: architecture, usage, pipeline authoring, payload contract, known limits.
- `docs/2026-11-01-cv-platform-and-face-encoding-research.md` — full research report; **the "Update 2026-11-01" sections at the end are the decision log** (read them first).
- `docs/superpowers/plans/2026-11-01-cpu-poc-plan.md` — the executed POC plan (all 9 tasks complete).
- `models/README.md` — model provenance: sources, licenses, SHA-256, fetch commands.
- `.superpowers/sdd/cpu-poc/progress.md` — execution ledger with rulings made during implementation.

## What this is

A **machine-side, composable computer-vision core** (Python 3.12, POC stage):

- Input: video streams/files. Processing: object detection, tracking, face
  recognition against a local gallery. Output: composed **event payloads**
  (`vision.event/1` JSON) delivered store-and-forward to an HTTP webhook
  (SQLite outbox; file emitter for development with byte-identical payloads).
- It is **not** an attendance product and has no UI/downstream business
  logic. Attendance in a classroom is the first example pipeline
  (`pipelines/attendance_poc.yaml`); `pipelines/dwell_alert.yaml` proves
  generality. The word "attendance" must never appear in `src/`.
- Users compose pipelines as validated YAML DAGs over registered nodes
  (capabilities + activities + conditions). Config only — arbitrary user
  code never runs on the machine.

## Setup on a new machine

```bash
git clone https://github.com/matoapunchstudio/vison.git
cd vison
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is absent
uv venv --python 3.12 && uv sync                 # env is ./.venv, never system python
# fetch weights (not in git) — exact commands + hashes in models/README.md
uv run pytest                                    # expect: 72 passed
```

Docker alternative: `docker build -f Dockerfile.cpu -t vision-platform:cpu .`
(models must be volume-mounted into `/app/models`).

## Architecture in one paragraph

Frame plane (`source → detect → track → face`) emits `TrackObservation`s;
the event plane (`zone.sequence`/`zone.dwell` activities, `condition.gate`,
`debounce`) turns them into `MachineEvent`s; emitters persist to the outbox
and deliver. Pipelines are data (YAML), nodes are versioned manifests with
typed ports and declared backends, and `runtime.profile` selects the frame-
plane backend: `dev-cpu` (implemented), `jetson-ort` (same nodes + CUDA /
TensorRT providers, not yet exercised), `jetson-deepstream` (future
frame-plane translation; manifests honestly reject it today). Payload,
validator, event plane, and outbox are backend-independent.

## Decisions already made (do not relitigate without new evidence)

- **Scope:** machine side only. Teacher counter UI, attendance rules,
  timetables, HR/reporting are downstream and out of scope.
- **Attendance means "student entered the room"** (check-in only).
  `person_left` is an optional per-pipeline reversed sequence, default off.
- **Product form:** general composable platform; new CV capabilities are
  added over time as registered nodes (manifest + node + runner wiring +
  tests) without changing validator/runner/outbox/payload schema.
- **Commercial product** → shipped models must be license-clean:
  YOLOX (Apache-2.0), YuNet (MIT), SFace (Apache-2.0), all from OpenCV Zoo.
  **Never** commit or depend on InsightFace weights (non-commercial) or
  Ultralytics (AGPL) — the POC acceptance check greps the repo for them.
- **Face encoding technique:** margin-loss embeddings are the SOTA family
  (ArcFace/AdaFace). SFace (128-d) is the shipped baseline; a licensed or
  self-trained AdaFace behind the `FaceEncoder` protocol is the upgrade
  path. Embeddings are model-bound; galleries store `model_id` and
  cross-model matching raises. Thresholds must be calibrated per site —
  `0.5` anywhere is a placeholder.
- **Delivery:** HTTP webhook, envelope `vision.event/1`, at-least-once via
  SQLite outbox, `event_id` is the downstream dedup key. Composed events
  only — never per-frame detections.
- **Language/runtime:** Python for POC and production; the frame plane is
  the swappable segment (CPU here → TensorRT/DeepStream on Jetson). Rust
  was considered and set aside by the owner.
- **Deployment target:** edge box per site with an NVIDIA GPU (Jetson
  class), offline-first with store-and-forward sync.

## Site facts (verified from the owner's sample frame)

- Dev PC (this repo's origin machine): x86_64, Intel iGPU only, **no NVIDIA
  GPU**; POC therefore runs CPU. It does have a webcam (`/dev/video0`).
- Site: **60 cameras total across the whole site, 3 Jetsons**; classroom
  cameras are **fewer than half** (<30). Exact classroom count,
  camera→Jetson assignment, and **Jetson model are still unknown** — ask,
  don't assume.
- Reference room "Labkom 2" (computer lab): **one wide camera** high in a
  front corner, 1920×1080; student entry is the **glass door at the back**
  (black side door is not an entry). Zone draft lives in
  `pipelines/attendance_poc.yaml`.
- Measured on that frame: seated faces are 15–55 px and steep top-down —
  only **2 pass the identification gate**; identification must happen at
  standing moments (walk-in capture → class look-up burst → front
  `id_spot` for unresolved students, reconciled by the teacher downstream).
  Tiled (2×2) YOLOX inference with **BGR** input order is required for
  usable person recall; CPU latency ≈539 ms detect / 79 ms face / 0.7 ms
  track per analyzed frame. These are dev-PC CPU numbers only — never
  present them as Jetson expectations.

## Working rules

- **TDD:** failing test first for every behavior change; run the full suite
  (`uv run pytest`) before claiming anything passes. Model-dependent tests
  are marked `model` and skip with a reason when weights/sample assets are
  absent — a skip is not a pass; say so.
- **Privacy:** never commit sample frames, clips, crops, galleries, or any
  student images/templates. Keep such material in `/tmp` or `out/`
  (both ignored). Biometrics of minors are special-category data —
  production deployment needs institutional/legal review (see research
  report Part 3).
- **Honesty in estimates:** `vpoc bench` prints its assumptions; keep it
  that way. Never claim Jetson/RTSP/DeepStream behavior that was not
  measured on target hardware.
- **Commits:** the owner approves history writes; don't push without being
  asked. Weights (`models/*.onnx`) and `out/` stay out of git.
- Keep this file and the research report's decision log updated when a
  decision above changes — future sessions resume from here.

## Next steps, in order

1. **Real motion footage proof:** a 30–60 s Labkom clip around class start;
   run `vpoc enroll` + `vpoc run` and confirm real `person_entered` /
   `identity_resolved` payloads (currently proven synthetically + on stills).
2. **Jetson bring-up** once the model is known: pin the L4T base in
   `Dockerfile.jetson`, install the JetPack-matched `onnxruntime-gpu`,
   run the same pipelines under `jetson-ort`, and bench. Decide on
   DeepStream only from those numbers.
3. **Threshold calibration** per camera from a small labeled set.
4. **Production layers:** RTSP source with reconnect, multi-camera
   supervision on one box, then new capability nodes as use cases arrive.
