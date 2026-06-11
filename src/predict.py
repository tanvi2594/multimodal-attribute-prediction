"""Batch inference on new products: extract (or reuse cached) embeddings,
apply saved PCA, run every trained attribute head.

Usage:
    python -m src.predict --csv data/test.csv --image-dir data/test_images \
        --run-dir runs/exp1 --out predictions.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.features.fusion import PCAReducer, load_fused
from src.models.lgbm_head import AttributeHead


def ensure_embeddings(csv, image_dir, cache_dir, config_path):
    """Extract embeddings for the inference set if not already cached."""
    try:
        load_fused(cache_dir)
        print("Using existing embedding cache.")
        return
    except Exception:
        pass
    print("Extracting embeddings for inference set...")
    subprocess.run(
        [sys.executable, "-m", "src.embeddings.extract",
         "--csv", csv, "--image-dir", image_dir, "--out-dir", cache_dir,
         "--config", config_path, "--fp16"],
        check=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", default=None, help="defaults to <run-dir>/infer_cache")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--with-proba", action="store_true", help="also write max class probability")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cache_dir = args.cache_dir or str(run_dir / "infer_cache")
    config = yaml.safe_load(Path(args.config).read_text())

    ensure_embeddings(args.csv, args.image_dir, cache_dir, args.config)

    x = load_fused(cache_dir, normalize_blocks=config["fusion"]["l2_normalize_blocks"])
    reducer = PCAReducer.load(run_dir / "pca.joblib")
    x = reducer.transform(x)

    df = pd.read_csv(args.csv)
    out = pd.DataFrame({"sample_id": df["sample_id"]})

    for head_path in sorted((run_dir / "heads").glob("*.joblib")):
        head = AttributeHead.load(head_path)
        out[head.attribute] = head.predict(x)
        if args.with_proba:
            out[f"{head.attribute}_confidence"] = head.predict_proba(x).max(axis=1)
        print(f"Predicted {head.attribute}")

    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
