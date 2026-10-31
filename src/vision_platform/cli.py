"""Command-line entry point (`vpoc`).

Subcommands:
    run     Run a pipeline document against a video file (Task 7).
    enroll  Enroll face photos into a gallery file (Task 7).
    flush   Re-send pending outbox payloads to a webhook (Task 7).
    bench   Measure per-node latency and estimate cameras per box (Task 8).
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vpoc", description="Vision platform POC core")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a pipeline on a video file")
    p_run.add_argument("--pipeline", required=True)
    p_run.add_argument("--source", required=True)
    p_run.add_argument("--outbox", default="out/outbox.db")
    p_run.add_argument("--webhook")
    p_run.add_argument("--events-file")
    p_run.add_argument("--gallery")
    p_run.add_argument("--machine-id", default="dev-pc")
    p_run.add_argument("--detector-model", help="YOLOX ONNX path (omit for no-detection dry wiring is an error; see vpoc run docs)")
    p_run.add_argument("--face-detector-model")
    p_run.add_argument("--face-recognizer-model")
    p_run.add_argument("--max-frames", type=int)

    p_enroll = sub.add_parser("enroll", help="enroll face photos into a gallery")
    p_enroll.add_argument("--gallery", required=True)
    p_enroll.add_argument("--identity", required=True)
    p_enroll.add_argument("--photos", nargs="+", required=True)
    p_enroll.add_argument("--face-detector-model", required=True)
    p_enroll.add_argument("--face-recognizer-model", required=True)

    p_flush = sub.add_parser("flush", help="re-send pending outbox payloads")
    p_flush.add_argument("--outbox", required=True)
    p_flush.add_argument("--webhook", required=True)

    p_bench = sub.add_parser("bench", help="measure per-node latency")
    p_bench.add_argument("--pipeline", required=True)
    p_bench.add_argument("--source", required=True)
    p_bench.add_argument("--frames", type=int, default=300)
    p_bench.add_argument("--face-cameras-fraction", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Bodies are wired in Tasks 7-8; import lazily so --help stays dependency-light.
    from vision_platform import cli_impl

    handler = getattr(cli_impl, f"cmd_{args.command}", None)
    if handler is None:
        print(f"not implemented yet: {args.command}", file=sys.stderr)
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
