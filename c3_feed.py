"""
c3_feed.py — the 20 fps MASTER feed for c3: the framerate-vs-pixel-loss grid.

Seed claim under test: "chiaroscuro can crank framerate to make up for pixel
loss." To compare framerates against glyph resolution at a fixed token budget
we need ONE master timeline that every framerate cell samples from — so all
cells see the SAME 10 seconds of the same sources, just at different temporal
strides.

Reuses c2_feed by import: the five lavfi sources + captions + read_source().
The graphs are rebuilt at 20 fps (master rate): `rate=10` -> `rate=20`, and
mandelbrot's `end_pts` scaled so the zoom still spans the same wall-clock 10 s.

Writes (not committed, see .gitignore):
  c3_frames/feed_meta.json    source order + per-source frame count + captions
  c3_frames/frames_u8.npy     (1000, 240, 320, 3) uint8 RGB  (~230 MB)

Usage:  python3 c3_feed.py
        C3_SMOKE=1 python3 c3_feed.py   # 2 s per source (40 frames each)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import c2_feed  # reuse: SOURCES (names, 10fps graphs, captions), read_source, W, H

MASTER_FPS = 20
W, H = c2_feed.W, c2_feed.H
SECONDS = 2 if os.environ.get("C3_SMOKE") else 10


def log(msg: str) -> None:
    print(f"[c3-feed] {msg}", file=sys.stderr, flush=True)


def master_graph(graph10: str) -> str:
    """Rebuild a c2 10fps lavfi graph at the 20 fps master rate."""
    g = graph10.replace("rate=10", f"rate={MASTER_FPS}")
    assert g != graph10, f"no rate= in graph: {graph10}"
    if "end_pts=" in g:  # mandelbrot: zoom end measured in frames
        g2 = g.replace(f"end_pts={10 * 10}", f"end_pts={SECONDS * MASTER_FPS}")
        assert g2 != g, f"unexpected end_pts in graph: {g}"
        g = g2
    return g


def main() -> None:
    n_per = SECONDS * MASTER_FPS
    frames, meta = [], []
    for name, graph10, caption in c2_feed.SOURCES:
        graph = master_graph(graph10)
        f = c2_feed.read_source(graph, n_per)
        frames.append(f)
        meta.append({"name": name, "lavfi": graph, "n_frames": int(f.shape[0]),
                     "intended_caption": caption})
        log(f"{name:10s} {f.shape[0]} frames @ {MASTER_FPS} fps {f.shape[1:]} (intended: {caption!r})")

    arr = np.concatenate(frames, axis=0)
    os.makedirs("c3_frames", exist_ok=True)
    np.save("c3_frames/frames_u8.npy", arr)
    with open("c3_frames/feed_meta.json", "w") as fh:
        json.dump({"fps": MASTER_FPS, "width": W, "height": H,
                   "seconds_per_source": SECONDS, "master": True,
                   "sources": meta}, fh, indent=2)
    log(f"total {arr.shape[0]} frames -> c3_frames/frames_u8.npy ({arr.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
