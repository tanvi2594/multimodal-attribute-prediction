"""Train PCA + per-attribute LightGBM heads on cached embeddings.

Usage:
    python -m src.train --cache-dir cache/train --labels data/train.csv \
        --attributes material pattern sleeve_type --pca-dim 512 --out-dir runs/exp1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.fusion import PCAReducer, load_fused
from src.models.lgbm_head import AttributeHead


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--labels", required=True, help="CSV with sample_id + attribute columns")
    p.add_argument("--attributes", nargs="+", required=True)
    p.add_argument("--pca-dim", type=int, default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = config["train"]["seed"]
    rng = np.random.default_rng(seed)

    print("Loading fused embeddings from cache...")
    x = load_fused(args.cache_dir, normalize_blocks=config["fusion"]["l2_normalize_blocks"])
    df = pd.read_csv(args.labels)
    assert len(df) == len(x), f"label rows ({len(df)}) != cached rows ({len(x)})"

    pca_dim = args.pca_dim or config["fusion"]["pca_dim"]
    print(f"Fitting PCA: {x.shape[1]} -> {pca_dim}")
    reducer = PCAReducer(dim=pca_dim, seed=seed)
    x = reducer.fit_transform(x)
    reducer.save(out_dir / "pca.joblib")
    print(f"Explained variance: {reducer.explained_variance:.3f}")

    # Shared train/val split across attributes
    n = len(x)
    idx = rng.permutation(n)
    n_val = int(n * config["train"]["val_fraction"])
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    metrics = []
    for attr in args.attributes:
        if attr not in df.columns:
            raise ValueError(f"attribute column '{attr}' not in labels CSV")
        labeled = df[attr].notna().to_numpy()
        tr = train_idx[labeled[train_idx]]
        va = val_idx[labeled[val_idx]]
        print(f"\n[{attr}] train={len(tr)} val={len(va)} classes={df[attr].nunique()}")

        head = AttributeHead(attr, config["lightgbm"])
        m = head.fit(x[tr], df[attr].to_numpy()[tr], x[va], df[attr].to_numpy()[va])
        head.save(out_dir / "heads")
        metrics.append(m)
        print(f"[{attr}] val_acc={m['val_accuracy']:.4f} macro_f1={m['val_macro_f1']:.4f}")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "run_config.json").write_text(json.dumps({
        "cache_dir": str(args.cache_dir),
        "attributes": args.attributes,
        "pca_dim": pca_dim,
        "fusion_keys": "default",
    }, indent=2))
    print(f"\nSaved run to {out_dir}")


if __name__ == "__main__":
    main()
