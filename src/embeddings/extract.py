"""Batched GPU extraction of CLIP / SigLIP / DINOv2 embeddings.

Design notes:
- Single decode, multi-encode: the DataLoader decodes each image once; all
  requested backbones consume the same PIL batch, cutting preprocessing ~3x
  vs. running three independent extraction jobs.
- fp16 autocast + pinned memory for throughput on A10G (g5 instances).
- Sharded, resumable cache (see cache.py) so hyperparameter experiments never
  pay for extraction twice.

Usage:
    python -m src.embeddings.extract --csv data/train.csv --image-dir data/images \
        --out-dir cache/train --models clip siglip dinov2 --batch-size 256 --fp16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ProductDataset, load_metadata, pil_collate
from src.embeddings.cache import ShardedCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Backbone wrappers — each exposes encode_images(pil_list) / encode_texts(list)
# --------------------------------------------------------------------------

class CLIPBackbone:
    keys = ("clip_image", "clip_text")

    def __init__(self, cfg):
        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg["name"], pretrained=cfg["pretrained"]
        )
        self.tokenizer = open_clip.get_tokenizer(cfg["name"])
        self.model = self.model.to(DEVICE).eval()
        self.image_dim = cfg["image_dim"]
        self.text_dim = cfg["text_dim"]

    @torch.no_grad()
    def encode_images(self, images):
        x = torch.stack([self.preprocess(im) for im in images]).to(DEVICE, non_blocking=True)
        return self.model.encode_image(x)

    @torch.no_grad()
    def encode_texts(self, texts):
        toks = self.tokenizer(texts).to(DEVICE)
        return self.model.encode_text(toks)


class SigLIPBackbone:
    keys = ("siglip_image", "siglip_text")

    def __init__(self, cfg):
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(cfg["name"])
        self.model = AutoModel.from_pretrained(cfg["name"]).to(DEVICE).eval()
        self.image_dim = cfg["image_dim"]
        self.text_dim = cfg["text_dim"]

    @torch.no_grad()
    def encode_images(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(DEVICE)
        return self.model.get_image_features(**inputs)

    @torch.no_grad()
    def encode_texts(self, texts):
        inputs = self.processor(
            text=texts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(DEVICE)
        return self.model.get_text_features(**inputs)


class DINOv2Backbone:
    keys = ("dinov2_image",)

    def __init__(self, cfg):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(cfg["name"])
        self.model = AutoModel.from_pretrained(cfg["name"]).to(DEVICE).eval()
        self.image_dim = cfg["image_dim"]

    @torch.no_grad()
    def encode_images(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(DEVICE)
        out = self.model(**inputs)
        return out.last_hidden_state[:, 0]  # CLS token


BACKBONES = {"clip": CLIPBackbone, "siglip": SigLIPBackbone, "dinov2": DINOv2Backbone}


# --------------------------------------------------------------------------

def extract(csv_path, image_dir, out_dir, model_names, batch_size, num_workers, fp16, config):
    df = load_metadata(csv_path, image_dir)
    n = len(df)
    shard_size = config["extraction"]["shard_size"]

    dataset = ProductDataset(df, image_size=config["extraction"]["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pil_collate,
        pin_memory=True,
    )

    backbones, caches = {}, {}
    for name in model_names:
        bb = BACKBONES[name](config["embedding_models"][name])
        backbones[name] = bb
        for key in bb.keys:
            dim = bb.text_dim if key.endswith("_text") else bb.image_dim
            caches[key] = ShardedCache(out_dir, key, n_rows=n, dim=dim, shard_size=shard_size)

    # Resume support: only extract shards not yet complete for ALL keys.
    done = set.intersection(*(c.completed_shards() for c in caches.values()))
    n_shards = (n + shard_size - 1) // shard_size
    todo = [i for i in range(n_shards) if i not in done]
    if not todo:
        print("All shards already cached — nothing to do.")
        return
    print(f"Extracting {len(todo)}/{n_shards} shards on {DEVICE} (resumed: {len(done)} done)")

    buffers = {k: [] for k in caches}
    rows_in_shard = 0
    shard_cursor = todo[0]
    todo_set = set(todo)

    autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16 and DEVICE == "cuda")

    for batch in tqdm(loader, total=len(loader)):
        first_row = int(batch["index"][0])
        shard_idx = first_row // shard_size
        if shard_idx not in todo_set:
            continue  # whole batch belongs to a completed shard (batch_size divides shard_size)

        with autocast:
            for name, bb in backbones.items():
                feats = bb.encode_images(batch["images"]).float().cpu().numpy()
                buffers[f"{name}_image"].append(feats)
                if f"{name}_text" in caches:
                    tfeats = bb.encode_texts(batch["texts"]).float().cpu().numpy()
                    buffers[f"{name}_text"].append(tfeats)

        rows_in_shard += len(batch["index"])
        shard_rows = min(shard_size, n - shard_cursor * shard_size)
        if rows_in_shard >= shard_rows:
            for key, cache in caches.items():
                cache.write_shard(shard_cursor, np.concatenate(buffers[key], axis=0))
                buffers[key] = []
            rows_in_shard = 0
            shard_cursor += 1

    print("Done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--models", nargs="+", default=["clip", "siglip", "dinov2"], choices=list(BACKBONES))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    assert config["extraction"]["shard_size"] % args.batch_size == 0, \
        "batch size must divide shard size for clean resume boundaries"

    extract(args.csv, args.image_dir, args.out_dir, args.models,
            args.batch_size, args.num_workers, args.fp16, config)


if __name__ == "__main__":
    main()
