"""
c2_clip.py — the discriminator: CLIP zero-shot as ground truth.

For every feed frame, score against the fixed caption set (5 source captions +
7 distractors, 12 total) with openai/clip-vit-base-patch32 (cached locally from
E15; HF_HUB_OFFLINE=1). Label = argmax caption. Margin = p(top1) - p(top2)
under softmax(cos_sim * logit_scale) over ALL 12 captions — low-margin frames
are CLIP being unsure, and c2_sweep.py drops them.

Also writes the normalized CLIP image embeddings — the raw-pixels control in
the sweep is a linear probe on exactly these (same reader class as the text
readers, only the feature space changes).

Runs in the torch env (~/venvs/elephant-gpu), fp16 on GPU if free.

Writes:
  c2_frames/clip_labels.npz : labels (n,), probs (n,12), margin (n,),
                              img_emb (n,512) float32 L2-normalized
prints the margin distribution + per-caption counts to stderr.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

MODEL_ID = "openai/clip-vit-base-patch32"
BATCH = 64
MARGIN_DROP_THRESHOLD = 0.10  # pre-registered: c2_sweep drops margin < this

CAPTIONS = [
    # the 5 source captions (order matches c2_feed.SOURCES)
    "a video test pattern with moving bars and grids",
    "a television color calibration test pattern with solid colored bars",
    "a colorful fractal zooming in",
    "a black and white cellular automaton simulation of cells evolving",
    "a smooth colorful gradient slowly shifting and blending",
    # 7 distractors
    "a person walking down a city street",
    "a cat sitting on a couch",
    "a night sky full of stars",
    "a blank white screen",
    "a page of text on a computer screen",
    "a busy highway with cars",
    "a crowd of people at a concert",
]


def log(msg: str) -> None:
    print(f"[c2-clip] {msg}", file=sys.stderr, flush=True)


def main() -> None:
    frames = np.load("c2_frames/frames_u8.npy")
    with open("c2_frames/feed_meta.json") as fh:
        meta = json.load(fh)
    n = frames.shape[0]
    log(f"{n} frames {frames.shape[1:]}, {len(CAPTIONS)} captions")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    log(f"model on {device} ({dtype})")

    with torch.no_grad():
        tok = processor(text=CAPTIONS, return_tensors="pt", padding=True)
        tok = {k: v.to(device) for k, v in tok.items()}
        # transformers 5.x: get_*_features returns BaseModelOutputWithPooling;
        # the projected 512-d CLIP embeddings are in .pooler_output
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
            if (i // BATCH) % 2 == 0:
                log(f"  {min(i + BATCH, n)}/{n} frames")

    img_emb = torch.cat(embs).numpy().astype(np.float32)
    probs = torch.cat(probs).numpy().astype(np.float32)
    labels = probs.argmax(axis=1).astype(np.int64)
    srt = np.sort(probs, axis=1)
    margin = srt[:, -1] - srt[:, -2]

    pct = {p: float(np.percentile(margin, p)) for p in (0, 5, 25, 50, 75, 95, 100)}
    kept = int((margin >= MARGIN_DROP_THRESHOLD).sum())
    names = [CAPTIONS[i] for i in range(len(CAPTIONS))]
    counts = {names[c]: int((labels == c).sum()) for c in np.unique(labels)}

    np.savez("c2_frames/clip_labels.npz", labels=labels, probs=probs,
             margin=margin, img_emb=img_emb)
    agreements = {}
    for si, m in enumerate(meta["sources"]):
        lo = si * m["n_frames"]
        agreements[m["name"]] = float((labels[lo : lo + m["n_frames"]] == si).mean())
    summary = {
        "model": MODEL_ID, "device": device, "n_frames": int(n),
        "captions": CAPTIONS, "logit_scale": logit_scale,
        "margin": {"percentiles": pct, "mean": float(margin.mean()),
                   "n_below_threshold": int(n - kept),
                   "threshold": MARGIN_DROP_THRESHOLD,
                   "kept_after_threshold": kept},
        "argmax_caption_counts": counts,
        "per_source_argmax_agreement": agreements,
    }
    with open("c2_frames/clip_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log(json.dumps(summary["margin"], indent=2))
    log("argmax counts: " + json.dumps(counts, indent=1))
    log("per-source agreement with intended caption: " +
        json.dumps({m["name"]: round(summary["per_source_argmax_agreement"][m["name"]], 3)
                    for m in meta["sources"]}))
    log("-> c2_frames/clip_labels.npz")


if __name__ == "__main__":
    main()
