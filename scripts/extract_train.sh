#!/usr/bin/env bash
# One-time GPU embedding extraction for the training set (run on g5.2xlarge)
python -m src.embeddings.extract \
  --csv data/train.csv \
  --image-dir data/images \
  --out-dir cache/train \
  --models clip siglip dinov2 \
  --batch-size 256 --num-workers 8 --fp16
