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
        enc_cfg = spec.nodes_of_type("face.encode")[0].config
        encoder = YuNetSFaceEncoder(
            getattr(args, "face_detector_model", None) or DEFAULT_FACE_DET,
            getattr(args, "face_recognizer_model", None) or DEFAULT_FACE_REC,
            min_face_px=int(enc_cfg.get("min_face_px", 40)))
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


def cmd_run(args) -> int:
    try:
        spec = load_pipeline(Path(args.pipeline))
        outbox = Outbox(Path(args.outbox))
        emitter = (WebhookEmitter(outbox, args.webhook) if args.webhook
                   else FileEmitter(outbox, Path(args.events_file or "out/events.jsonl")))
        runner = build_runner(spec, args, emitter)
        with VideoFileSource(args.source,
                             analytics_fps=float(spec.source.get("analytics_fps", 5))) as src:
            frames = src if not args.max_frames else _take(src, args.max_frames)
            stats = runner.run(frames)
    except (PipelineSpecError, FileNotFoundError, ValueError) as e:
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
        encoder = YuNetSFaceEncoder(args.face_detector_model, args.face_recognizer_model,
                                    min_face_px=1)
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
            gallery.enroll(args.identity, hits[0].embedding)
    except (FileNotFoundError, ValueError) as e:
        return _err(str(e))
    gallery.save(gallery_path)
    print(json.dumps({"identity": args.identity, "gallery": str(gallery_path),
                      "identities": len(gallery)}))
    return 0


def cmd_flush(args) -> int:
    emitter = WebhookEmitter(Outbox(Path(args.outbox)), args.webhook)
    print(json.dumps({"sent": emitter.flush(limit=100000)}))
    return 0
