"""Payload emission: versioned envelope, SQLite outbox, webhook/file emitters.

Delivery contract (vision.event/1): every composed event is written to the
local outbox before any network attempt, delivered at-least-once, and marked
sent only after success. `event_id` never changes across retries or process
restarts — it is the downstream dedup key.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vision_platform.types import MachineEvent

SCHEMA = "vision.event/1"


def to_payload(event: MachineEvent, machine_id: str, pipeline: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "event_id": event.event_id,
        "type": event.type,
        "ts": event.ts,
        "machine_id": machine_id,
        "camera_id": event.camera_id,
        "pipeline": pipeline,
        "track_id": event.track_id,
        "identity": event.identity,
        "identity_score": event.identity_score,
        "data": event.data,
    }


class Outbox:
    """Append-only local store of payloads awaiting (or having completed) delivery."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "event_id TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                "status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0, "
                "created REAL NOT NULL, sent REAL)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def put(self, event: MachineEvent, machine_id: str, pipeline: str) -> dict[str, Any]:
        """Persist the payload (idempotent on event_id) and return it."""
        payload = to_payload(event, machine_id, pipeline)
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO events(event_id, payload, created) VALUES (?,?,?)",
                      (event.event_id, json.dumps(payload), event.ts))
            row = c.execute("SELECT payload FROM events WHERE event_id=?",
                            (event.event_id,)).fetchone()
        return json.loads(row[0])

    def pending(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT payload FROM events WHERE status='pending' ORDER BY created LIMIT ?",
                (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def mark_sent(self, event_id: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE events SET status='sent', sent=strftime('%s','now') WHERE event_id=?",
                      (event_id,))

    def mark_failed(self, event_id: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE events SET attempts=attempts+1 WHERE event_id=?", (event_id,))


def _http_post(url: str, payload: dict[str, Any], timeout: float) -> int:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (operator URL)
        return int(resp.status)


class WebhookEmitter:
    """Outbox-backed HTTP webhook delivery (stdlib urllib; no extra dependency).

    A failed flush starts an exponential-backoff cooldown (1s, 2s, … capped
    at 30s) during which emit() only persists — a down receiver can never
    stall frame processing by more than one timeout per cooldown window.
    flush(force=True) (the `vpoc flush` recovery path) ignores the cooldown.
    """

    def __init__(self, outbox: Outbox, url: str, timeout_s: float = 5.0,
                 post: Callable[[str, dict, float], int] | None = None):
        import time
        self.outbox = outbox
        self.url = url
        self.timeout_s = timeout_s
        self._post = post or _http_post
        self._clock = time.monotonic
        self._retry_after = 0.0
        self._failures = 0

    def emit(self, event: MachineEvent, machine_id: str, pipeline: str) -> None:
        self.outbox.put(event, machine_id, pipeline)
        self.flush()

    def flush(self, limit: int = 100, force: bool = False) -> int:
        """Attempt pending payloads; returns how many were accepted (2xx) this call."""
        if not force and self._clock() < self._retry_after:
            return 0
        sent = 0
        failed = False
        for payload in self.outbox.pending(limit):
            try:
                status = self._post(self.url, payload, self.timeout_s)
            except Exception:
                status = 0
            if 200 <= status < 300:
                self.outbox.mark_sent(payload["event_id"])
                sent += 1
            else:
                self.outbox.mark_failed(payload["event_id"])
                failed = True
                break  # receiver is down; do not burn a timeout per pending row
        if failed:
            self._failures += 1
            self._retry_after = self._clock() + min(30.0, 2.0 ** (self._failures - 1))
        else:
            self._failures = 0
            self._retry_after = 0.0
        return sent


class FileEmitter:
    """Appends payloads as JSONL — byte-identical to webhook bodies (POC default)."""

    def __init__(self, outbox: Outbox, path: str | Path):
        self.outbox = outbox
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: MachineEvent, machine_id: str, pipeline: str) -> None:
        self.outbox.put(event, machine_id, pipeline)
        self.flush()

    def flush(self, limit: int = 100000) -> int:
        """Append every pending payload (crash-gap recovery), then mark sent.

        A crash between append and mark_sent can duplicate a line on the next
        flush — the file path is at-least-once, same as the webhook."""
        pending = self.outbox.pending(limit)
        if not pending:
            return 0
        with self.path.open("a") as fh:
            for payload in pending:
                fh.write(json.dumps(payload) + "\n")
        for payload in pending:
            self.outbox.mark_sent(payload["event_id"])
        return len(pending)
