"""Implementations behind the `vpoc` CLI (kept out of cli.py so --help stays light)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

from vision_platform.detect import YoloxOnnxDetector
from vision_platform.emit import FileEmitter, Outbox, WebhookEmitter
from vision_platform.face import YuNetSFaceEncoder
from vision_platform.gallery import Gallery
from vision_platform.pipeline import PipelineRunner, PipelineSpecError, load_pipeline
from vision_platform.sources import VideoFileSource

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETECTOR = ROOT / "models" / "object_detection_yolox_2022nov.onnx"
DEFAULT_FACE_DET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_FACE_REC = ROOT / "models" / "face_recognition_sface_2021dec.onnx"


def _err(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


def build_runner(spec, args, emitter, recorder=None):
    """Construct detector/encoder/gallery for a pipeline spec from CLI args.

    Raises FileNotFoundError/ValueError with the offending path named; callers
    convert those to CLI errors. Validates everything before any outbox write.
    """
    providers = list(spec.runtime.get("providers") or ["CPUExecutionProvider"])
    det_nodes = spec.nodes_of_type("detect.persons")
    if not det_nodes:
        raise ValueError("pipeline has no detect.persons node; runner needs detections")
    det_cfg = det_nodes[0].config
    tile = det_cfg.get("tile_grid") or [1, 1]
    detector = YoloxOnnxDetector(
        getattr(args, "detector_model", None) or DEFAULT_DETECTOR,
        providers=providers,
        score_threshold=float(det_cfg.get("score_threshold", 0.4)),
        tile_grid=(int(tile[0]), int(tile[1])))
    active = detector.active_providers
    wanted = providers[0]
    if wanted != "CPUExecutionProvider" and wanted not in active:
        raise ValueError(f"requested provider {wanted} not active (active: {active}); "
                         "refusing silent CPU fallback")
    encoder = gallery = None
    if spec.nodes_of_type("face.encode"):
        encoder = YuNetSFaceEncoder(
            getattr(args, "face_detector_model", None) or DEFAULT_FACE_DET,
            getattr(args, "face_recognizer_model", None) or DEFAULT_FACE_REC)
        gallery_path = getattr(args, "gallery", None)
        if gallery_path:
            gallery = Gallery.load(Path(gallery_path))
            if gallery.model_id != encoder.model_id:
                raise ValueError(f"gallery model_id {gallery.model_id!r} does not match "
                                 f"encoder model_id {encoder.model_id!r}")
        else:
            gallery = Gallery(model_id=encoder.model_id)  # empty: everyone unknown
    return PipelineRunner(spec, detector=detector, encoder=encoder, gallery=gallery,
                          emitter=emitter, machine_id=getattr(args, "machine_id", "dev-pc"),
                          recorder=recorder)


def _preflight_files(spec, args) -> None:
    """Check every referenced file exists BEFORE the outbox is created."""
    wanted = [Path(args.source)]
    if spec.nodes_of_type("detect.persons"):
        wanted.append(Path(getattr(args, "detector_model", None) or DEFAULT_DETECTOR))
    if spec.nodes_of_type("face.encode"):
        wanted.append(Path(getattr(args, "face_detector_model", None) or DEFAULT_FACE_DET))
        wanted.append(Path(getattr(args, "face_recognizer_model", None) or DEFAULT_FACE_REC))
    if getattr(args, "gallery", None):
        wanted.append(Path(args.gallery))
    for p in wanted:
        if not p.exists():
            raise FileNotFoundError(f"required file not found: {p}")


def _emitter_for(spec, args, outbox):
    """Delivery target is decided by the pipeline's emit node, not silently by flags."""
    emit_type = spec.nodes_of_type("emit.webhook", "emit.file")[0].type
    if emit_type == "emit.webhook":
        if not args.webhook:
            raise ValueError("pipeline declares emit.webhook but --webhook URL was not given")
        if args.events_file:
            raise ValueError("pipeline declares emit.webhook but --events-file was given; "
                             "fix the pipeline document or drop --events-file")
        return WebhookEmitter(outbox, args.webhook)
    if args.webhook:
        raise ValueError("pipeline declares emit.file but --webhook was given; "
                         "fix the pipeline document or drop --webhook")
    return FileEmitter(outbox, Path(args.events_file or "out/events.jsonl"))


def cmd_run(args) -> int:
    if args.max_frames is not None and args.max_frames <= 0:
        return _err("--max-frames must be a positive integer")
    try:
        spec = load_pipeline(Path(args.pipeline))
        _preflight_files(spec, args)
        outbox = Outbox(Path(args.outbox))
        emitter = _emitter_for(spec, args, outbox)
        runner = build_runner(spec, args, emitter)
        with VideoFileSource(args.source,
                             analytics_fps=float(spec.source.get("analytics_fps", 5))) as src:
            frames = src if not args.max_frames else _take(src, args.max_frames)
            stats = runner.run(frames)
    except (PipelineSpecError, FileNotFoundError, ValueError, KeyError) as e:
        return _err(str(e))
    print(json.dumps({"frames": stats.frames, "events": stats.events}))
    return 0


def _take(source, n):
    for i, item in enumerate(source):
        if i >= n:
            return
        yield item


def cmd_enroll(args) -> int:
    gallery_path = Path(args.gallery)
    try:
        encoder = YuNetSFaceEncoder(args.face_detector_model, args.face_recognizer_model)
        gallery = Gallery.load(gallery_path) if gallery_path.exists() \
            else Gallery(model_id=encoder.model_id)
        if gallery.model_id != encoder.model_id:
            return _err(f"gallery model_id {gallery.model_id!r} != encoder {encoder.model_id!r}")
        # Enroll into the in-memory copy only; the file is written after ALL
        # photos pass the exactly-one-face check, so a bad photo changes nothing.
        for photo in args.photos:
            img = cv2.imread(photo)
            if img is None:
                return _err(f"cannot read photo: {photo}")
            hits = encoder.embed(img)
            if len(hits) != 1:
                return _err(f"photo {photo}: expected exactly 1 face, found {len(hits)}")
            hit = hits[0]
            width = hit.bbox[2] - hit.bbox[0]
            if width < 40 or hit.quality < 0.6:
                return _err(f"photo {photo}: face too small/poor for enrollment "
                            f"(width={width:.0f}px, quality={hit.quality:.2f}); "
                            "use a closer, frontal photo")
            gallery.enroll(args.identity, hit.embedding)
    except (FileNotFoundError, ValueError) as e:
        return _err(str(e))
    gallery.save(gallery_path)
    print(json.dumps({"identity": args.identity, "gallery": str(gallery_path),
                      "identities": len(gallery)}))
    return 0


def cmd_bench(args) -> int:
    from vision_platform.bench import LatencyRecorder, estimate_cameras_per_box

    try:
        spec = load_pipeline(Path(args.pipeline))
        outbox = Outbox(Path("out/bench_outbox.db"))
        emitter = FileEmitter(outbox, Path("out/bench_events.jsonl"))
        recorder = LatencyRecorder()
        runner = build_runner(spec, args, emitter, recorder=recorder)
        with VideoFileSource(args.source,
                             analytics_fps=float(spec.source.get("analytics_fps", 5))) as src:
            stats = runner.run(_take(src, args.frames))
    except (PipelineSpecError, FileNotFoundError, ValueError) as e:
        return _err(str(e))
    summary = recorder.summary()
    estimate = estimate_cameras_per_box(
        summary, analytics_fps=float(spec.source.get("analytics_fps", 5)),
        face_cameras_fraction=args.face_cameras_fraction)
    print(json.dumps({"label": "this PC, CPU", "frames": stats.frames,
                      "events": stats.events, "nodes": summary,
                      "estimate": estimate}, indent=2))
    return 0


def cmd_flush(args) -> int:
    if bool(args.webhook) == bool(args.events_file):
        return _err("give exactly one of --webhook or --events-file")
    if not Path(args.outbox).exists():
        return _err(f"outbox not found: {args.outbox} (nothing to flush; check the path)")
    outbox = Outbox(Path(args.outbox))
    if args.webhook:
        sent = WebhookEmitter(outbox, args.webhook).flush(limit=100000, force=True)
    else:
        sent = FileEmitter(outbox, Path(args.events_file)).flush()
    print(json.dumps({"sent": sent}))
    return 0
