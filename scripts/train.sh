#!/usr/bin/env bash
# Train PCA + LightGBM heads on cached embeddings (CPU is fine)
python -m src.train \
  --cache-dir cache/train \
  --labels data/train.csv \
  --attributes material pattern sleeve_type occasion \
  --pca-dim 512 \
  --out-dir runs/exp1
