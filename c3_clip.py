"""
c3_clip.py — the discriminator for c3: CLIP zero-shot labels on the 20 fps
MASTER timeline.

Every master frame (20 fps x 10 s x 5 sources = 1000) gets a CLIP label using
c2's exact caption set and model (openai/clip-vit-base-patch32, cached,
HF_HUB_OFFLINE=1, fp16 on GPU if free). Every c3 cell samples ITS frames from
this master labeling, so all 12 cells share one consistent ground truth —
"CLIP labels per frame at the highest available fps".

Runs in the torch env (~/venvs/elephant-gpu), serialized (only GPU step in c3).

Writes:
  c3_frames/clip_labels.npz : labels (n,), probs (n,12), margin (n,)
  c3_frames/clip_summary.json : margin stats + argmax counts (no filtering:
      c3 keeps ALL frames — label noise is identical across cells and the
      framerate-vs-density comparison is paired)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from c2_clip import BATCH, CAPTIONS, MODEL_ID, log  # reuse: captions, model id, batch  # noqa: E402

import torch  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402


def main() -> None:
    frames = np.load("c3_frames/frames_u8.npy")
    with open("c3_frames/feed_meta.json") as fh:
        meta = json.load(fh)
    n = frames.shape[0]
    log(f"{n} frames {frames.shape[1:]} @ {meta['fps']} fps, {len(CAPTIONS)} captions")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    log(f"model on {device} ({dtype})")

    with torch.no_grad():
        tok = processor(text=CAPTIONS, return_tensors="pt", padding=True)
        tok = {k: v.to(device) for k, v in tok.items()}
        # transformers 5.x: projected 512-d CLIP embeddings land in .pooler_output
        txt = model.get_text_features(**{k: tok[k] for k in ("input_ids", "attention_mask")}).pooler_output
        txt = txt / txt.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp().item()

        embs, probs = [], []
        for i in range(0, n, BATCH):
            batch = [np.ascontiguousarray(f) for f in frames[i : i + BATCH]]
            img = processor(images=batch, return_tensors="pt")
            img = {k: v.to(device, dtype=dtype) for k, v in img.items()}
            emb = model.get_image_features(**img).pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embs.append(emb.float().cpu())
            sims = (emb.float() @ txt.float().T) * logit_scale
            probs.append(torch.softmax(sims, dim=-1).cpu())
            if (i // BATCH) % 4 == 0:
                log(f"  {min(i + BATCH, n)}/{n} frames")

    probs = torch.cat(probs).numpy().astype(np.float32)
    labels = probs.argmax(axis=1).astype(np.int64)
    srt = np.sort(probs, axis=1)
    margin = srt[:, -1] - srt[:, -2]

    np.savez("c3_frames/clip_labels.npz", labels=labels, probs=probs, margin=margin)
    pct = {p: float(np.percentile(margin, p)) for p in (0, 5, 25, 50, 75, 95, 100)}
    names = [CAPTIONS[i] for i in range(len(CAPTIONS))]
    counts = {names[c]: int((labels == c).sum()) for c in np.unique(labels)}
    agreements = {}
    start = 0
    for si, m in enumerate(meta["sources"]):
        cnt = m["n_frames"]
        agreements[m["name"]] = float((labels[start : start + cnt] == si).mean())
        start += cnt
    summary = {
        "model": MODEL_ID, "device": device, "n_frames": int(n),
        "fps": meta["fps"], "captions": CAPTIONS, "logit_scale": logit_scale,
        "margin": {"percentiles": pct, "mean": float(margin.mean())},
        "argmax_caption_counts": counts,
        "per_source_argmax_agreement": agreements,
    }
    with open("c3_frames/clip_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log("argmax counts: " + json.dumps(counts, indent=1))
    log("per-source agreement: " +
        json.dumps({m["name"]: round(agreements[m["name"]], 3) for m in meta["sources"]}))
    log("-> c3_frames/clip_labels.npz")


if __name__ == "__main__":
    main()
