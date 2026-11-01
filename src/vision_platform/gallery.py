"""Identity gallery: per-person mean templates with cosine matching.

Embeddings are model-bound: a gallery stores the `model_id` that produced its
vectors, and matching with another model's vectors raises instead of silently
comparing across incompatible spaces. Persistence is a single .npz file —
identities as one stacked array plus a JSON sidecar inside the archive.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("embedding has zero norm")
    return v / n


class Gallery:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.templates: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}

    def enroll(self, identity: str, embedding: np.ndarray) -> None:
        """Fold one enrollment embedding into the identity's running-mean template."""
        emb = _normalize(embedding)
        if identity in self.templates:
            n = self._counts[identity]
            mean = (self.templates[identity] * n + emb) / (n + 1)
            self.templates[identity] = _normalize(mean)
            self._counts[identity] = n + 1
        else:
            self.templates[identity] = emb
            self._counts[identity] = 1

    def identify(self, embedding: np.ndarray, threshold: float) -> tuple[str, float] | None:
        """Nearest template by cosine similarity, or None when below threshold (unknown)."""
        if not self.templates:
            return None
        emb = _normalize(embedding)
        names = list(self.templates)
        mat = np.stack([self.templates[n] for n in names])
        scores = mat @ emb
        i = int(np.argmax(scores))
        if float(scores[i]) < threshold:
            return None
        return names[i], float(scores[i])

    def identify_with_model(self, embedding: np.ndarray, model_id: str,
                            threshold: float) -> tuple[str, float] | None:
        if model_id != self.model_id:
            raise ValueError(
                f"embedding model_id {model_id!r} does not match gallery model_id "
                f"{self.model_id!r}; embeddings from different models are not comparable")
        return self.identify(embedding, threshold)

    def save(self, path: Path) -> None:
        """Atomically write to exactly `path` (np.savez on a path would append .npz)."""
        import os

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = list(self.templates)
        mat = (np.stack([self.templates[n] for n in names]) if names
               else np.zeros((0, 0), dtype=np.float32))
        meta = json.dumps({"model_id": self.model_id, "names": names,
                           "counts": [self._counts[n] for n in names]})
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez(fh, templates=mat, meta=np.array(meta))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "Gallery":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"gallery file not found: {path}")
        try:
            with np.load(path) as data:
                meta = json.loads(str(data["meta"]))
                mat = np.array(data["templates"])
        except Exception as e:
            raise ValueError(f"cannot load gallery {path}: {e}") from e
        g = cls(model_id=meta["model_id"])
        for name, count, row in zip(meta["names"], meta["counts"], mat):
            g.templates[name] = _normalize(row)
            g._counts[name] = int(count)
        return g

    def __len__(self) -> int:
        return len(self.templates)
