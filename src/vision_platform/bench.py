"""Latency recording and cameras-per-box estimation.

The estimator is deliberately transparent: it divides one second of a single
sequential worker by the measured per-frame cost times analytics FPS, with
face cost weighted by the fraction of cameras that run face nodes. Every
input that could flatter the result (accelerator speedup above all) is an
explicit argument and is printed in `assumptions` — the estimator never
invents a Jetson number. Measure on the target box before trusting it.
"""

from __future__ import annotations

import math
from typing import Any

FACE_NODE = "face.encode+match"


class LatencyRecorder:
    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}

    def record(self, node: str, ms: float) -> None:
        self._samples.setdefault(node, []).append(float(ms))

    def summary(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for node, xs in self._samples.items():
            xs_sorted = sorted(xs)
            p95_idx = max(0, math.ceil(0.95 * len(xs_sorted)) - 1)
            out[node] = {
                "count": len(xs_sorted),
                "mean_ms": sum(xs_sorted) / len(xs_sorted),
                "p95_ms": xs_sorted[p95_idx],
            }
        return out


def estimate_cameras_per_box(summary: dict[str, dict[str, Any]], analytics_fps: float,
                             face_cameras_fraction: float,
                             accelerator_speedup: float = 1.0) -> dict[str, Any]:
    if not summary:
        raise ValueError("empty latency summary")
    bottleneck = max(summary, key=lambda n: summary[n]["mean_ms"])
    per_frame_ms = sum(s["mean_ms"] for n, s in summary.items() if n != FACE_NODE)
    per_frame_ms += face_cameras_fraction * summary.get(FACE_NODE, {"mean_ms": 0.0})["mean_ms"]
    per_camera_per_s = per_frame_ms * analytics_fps
    cameras = int(1000.0 * accelerator_speedup // per_camera_per_s) if per_camera_per_s > 0 else 0
    burst_per_frame = sum(s["mean_ms"] for s in summary.values())
    burst_per_s = burst_per_frame * analytics_fps
    burst = int(1000.0 * accelerator_speedup // burst_per_s) if burst_per_s > 0 else 0
    return {
        "cameras": cameras,
        "entry_burst_cameras": burst,
        "bottleneck_node": bottleneck,
        "per_camera_ms_per_frame": per_frame_ms,
        "assumptions": [
            f"accelerator speedup={accelerator_speedup} (1.0 = the machine just measured; "
            "any other value is a hypothesis to re-measure on target hardware)",
            f"analytics_fps={analytics_fps} for every camera, continuously",
            f"face cost weighted by face_cameras_fraction={face_cameras_fraction}",
            "single sequential worker; decode, disk, network and OS overhead excluded",
            "entry burst scenario: entry_burst_cameras assumes every camera pays full "
            "face cost simultaneously (synchronized class start); size for the burst "
            "number, not the steady-state one, when starts are synchronized",
        ],
    }
