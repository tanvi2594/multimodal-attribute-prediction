# Multimodal Product Attribute Prediction

Predicting structured product attributes (e.g. material, pattern, sleeve type, occasion) from **product images + text descriptions** on a large-scale e-commerce dataset.

Instead of fine-tuning a large vision-language model end-to-end, this pipeline extracts **frozen embeddings from CLIP, SigLIP, and DINOv2**, fuses them, reduces dimensionality with PCA, and trains a **LightGBM head** on top. This matches or beats end-to-end fine-tuning baselines on attribute accuracy at a fraction of the compute cost.

## Why frozen embeddings + GBDT?

| Approach | Accuracy (macro-F1, internal benchmark) | GPU hours |
|---|---|---|
| ViT-B/16 fine-tuned end-to-end | 0.93 | ~38 |
| CLIP linear probe | 0.88 | ~2 |
| **CLIP + SigLIP + DINOv2 → PCA → LightGBM (ours)** | **0.94** | **~3** |

- **Complementary representations**: CLIP/SigLIP capture text-aligned semantics ("floral", "formal"), DINOv2 captures fine-grained visual structure (texture, weave, silhouette) that text-aligned models miss.
- **Embeddings are extracted once and cached** — every downstream experiment (feature ablations, hyperparameter sweeps, new attribute heads) runs on CPU in minutes.
- **LightGBM handles tabular fused features** better than a linear probe and trains in minutes, with native multiclass support and good calibration.

## Architecture

```
 product image ─┬─> CLIP ViT-L/14 image encoder ──┐
                ├─> SigLIP so400m image encoder ──┤
                └─> DINOv2 ViT-L/14 ──────────────┤
                                                  ├─> concat ─> PCA (d=512) ─> LightGBM (per attribute)
 product text ──┬─> CLIP text encoder ────────────┤
                └─> SigLIP text encoder ──────────┘
```

## Repo structure

```
src/
  data/dataset.py        # dataset loading, image fetching, text cleaning
  embeddings/extract.py  # batched GPU extraction for CLIP / SigLIP / DINOv2
  embeddings/cache.py    # memory-mapped .npy embedding cache, resumable
  features/fusion.py     # embedding fusion + PCA
  models/lgbm_head.py    # LightGBM multiclass head, per-attribute
  train.py               # end-to-end training entrypoint
  predict.py             # batch inference on new products
configs/config.yaml      # models, batch sizes, PCA dims, LightGBM params
scripts/                 # convenience shell scripts
```

## Dataset

Trained and evaluated on a public large-scale e-commerce product dataset (~200K products with images, titles/descriptions, and multi-valued categorical attribute labels). The pipeline is dataset-agnostic — any CSV with `sample_id, image_path, text` plus attribute columns works (see `src/data/dataset.py`).

## Setup

```bash
pip install -r requirements.txt
```

GPU strongly recommended for embedding extraction (tested on AWS `g5.2xlarge`, A10G 24GB). Training the LightGBM head and inference on cached embeddings run fine on CPU.

## Usage

**1. Extract and cache embeddings** (one-time, GPU):

```bash
python -m src.embeddings.extract \
  --csv data/train.csv \
  --image-dir data/images \
  --out-dir cache/train \
  --models clip siglip dinov2 \
  --batch-size 256 --fp16
```

Extraction is resumable — interrupted runs pick up from the last completed shard.

**2. Train the head** (CPU is fine):

```bash
python -m src.train \
  --cache-dir cache/train \
  --labels data/train.csv \
  --attributes material pattern sleeve_type \
  --pca-dim 512 \
  --out-dir runs/exp1
```

**3. Batch inference:**

```bash
python -m src.predict \
  --csv data/test.csv \
  --image-dir data/test_images \
  --run-dir runs/exp1 \
  --out predictions.csv
```

## Performance engineering notes

- **fp16 inference + pinned-memory DataLoader** with parallel image decoding (`num_workers=8`, PIL-SIMD compatible) — extraction throughput ~1,400 images/s for CLIP ViT-L on a single A10G.
- **Shard-based caching**: embeddings written as memory-mapped `.npy` shards keyed by content hash, so re-runs skip already-extracted rows and multiple model variants share the same decoded-image pipeline.
- **Single decode, multi-encode**: each image is decoded and resized once, then fed to all three vision backbones in the same batch loop, cutting preprocessing time ~3x vs. running extractors independently.

## License

MIT
