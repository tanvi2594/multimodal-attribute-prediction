"""Sharded, resumable embedding cache.

Embeddings are stored as float16 .npy shards:

    cache/train/clip_image/shard_00000.npy   (shard_size, dim)
    cache/train/clip_image/manifest.json     {"shard_size": ..., "dim": ..., "n_rows": ...}

A run can be interrupted and resumed: completed shards are detected and skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ShardedCache:
    def __init__(self, root: str | Path, key: str, n_rows: int, dim: int, shard_size: int = 8192):
        self.dir = Path(root) / key
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n_rows = n_rows
        self.dim = dim
        self.shard_size = shard_size
        self.n_shards = (n_rows + shard_size - 1) // shard_size
        self._manifest_path = self.dir / "manifest.json"
        self._write_manifest()

    def _write_manifest(self):
        self._manifest_path.write_text(
            json.dumps({"n_rows": self.n_rows, "dim": self.dim, "shard_size": self.shard_size})
        )

    def shard_path(self, shard_idx: int) -> Path:
        return self.dir / f"shard_{shard_idx:05d}.npy"

    def completed_shards(self) -> set[int]:
        done = set()
        for i in range(self.n_shards):
            p = self.shard_path(i)
            if p.exists():
                expected = min(self.shard_size, self.n_rows - i * self.shard_size)
                try:
                    arr = np.load(p, mmap_mode="r")
                    if arr.shape == (expected, self.dim):
                        done.add(i)
                except Exception:
                    pass
        return done

    def write_shard(self, shard_idx: int, data: np.ndarray):
        expected = min(self.shard_size, self.n_rows - shard_idx * self.shard_size)
        assert data.shape == (expected, self.dim), f"bad shard shape {data.shape}"
        tmp = self.shard_path(shard_idx).with_suffix(".tmp.npy")
        np.save(tmp, data.astype(np.float16))
        tmp.rename(self.shard_path(shard_idx))

    @staticmethod
    def load_all(root: str | Path, key: str) -> np.ndarray:
        """Load a full cached matrix (float32) by concatenating shards in order."""
        d = Path(root) / key
        manifest = json.loads((d / "manifest.json").read_text())
        shards = sorted(d.glob("shard_*.npy"))
        n_expected = (manifest["n_rows"] + manifest["shard_size"] - 1) // manifest["shard_size"]
        if len(shards) != n_expected:
            raise RuntimeError(
                f"{key}: cache incomplete ({len(shards)}/{n_expected} shards). Re-run extraction."
            )
        return np.concatenate([np.load(p).astype(np.float32) for p in shards], axis=0)
