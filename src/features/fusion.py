"""Embedding fusion: per-block L2 normalization -> concat -> PCA.

Per-block normalization matters: raw embedding norms differ wildly across
backbones (DINOv2 CLS norms >> CLIP norms), and without it PCA directions
are dominated by whichever model has the largest scale.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

from src.embeddings.cache import ShardedCache

DEFAULT_KEYS = ["clip_image", "clip_text", "siglip_image", "siglip_text", "dinov2_image"]


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def load_fused(cache_dir: str | Path, keys: list[str] = None, normalize_blocks: bool = True) -> np.ndarray:
    keys = keys or DEFAULT_KEYS
    blocks = []
    for k in keys:
        block = ShardedCache.load_all(cache_dir, k)
        blocks.append(_l2(block) if normalize_blocks else block)
    return np.concatenate(blocks, axis=1)


class PCAReducer:
    def __init__(self, dim: int = 512, seed: int = 42):
        self.pca = PCA(n_components=dim, random_state=seed)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.pca.fit_transform(x).astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.pca.transform(x).astype(np.float32)

    @property
    def explained_variance(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())

    def save(self, path: str | Path):
        joblib.dump(self.pca, path)

    @classmethod
    def load(cls, path: str | Path) -> "PCAReducer":
        obj = cls.__new__(cls)
        obj.pca = joblib.load(path)
        return obj
