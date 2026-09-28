"""
c2_feed.py — the animation feed for c2: ffmpeg lavfi time-varying sources.

"The video is the supervision — any feed of people or animations."
(No webcam needed yet: pure on-box synthetic feeds.)

Five sources, 10 s each at 10 fps, 320x240 (4:3 to match the c1 aspect
convention) -> 500 frames total, assembled into frames_u8.npy in feed order:

  testsrc2   moving bars + grids + rolling color square   (the "test pattern")
  smptebars  static SMPTE color calibration bars          (the "solid color bars")
  mandelbrot zooming fractal                              (the "fractal zoom")
  life       Conway cellular automaton, white on black    (the "cells evolving")
  gradients  smooth animated color gradients              (the "gradient sweep")

Each source carries its intended caption (what CLIP *should* see); the actual
label set is whatever CLIP's argmax says (see c2_clip.py).

Writes (not committed, see .gitignore):
  c2_frames/feed_meta.json    source order + per-source frame count + intended captions
  c2_frames/frames_u8.npy     (500, 240, 320, 3) uint8 RGB

Usage:  python3 c2_feed.py          (system python3: needs ffmpeg + cv2 for nothing
                                     here — raw frames come straight off the pipe)
        C2_SMOKE=1 python3 c2_feed.py   # 2 s per source (20 frames each)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

FFMPEG = os.environ.get("C2_FFMPEG", "ffmpeg")
FPS = 10
W, H = 320, 240
SECONDS = 10 if not os.environ.get("C2_SMOKE") else 2

# (name, lavfi graph, intended caption). frame count derived from SECONDS * FPS.
SOURCES = [
    ("testsrc2",
     f"testsrc2=size={W}x{H}:rate={FPS}",
     "a video test pattern with moving bars and grids"),
    ("smptebars",
     f"smptebars=size={W}x{H}:rate={FPS}",
     "a television color calibration test pattern with solid colored bars"),
    ("mandelbrot",
     f"mandelbrot=size={W}x{H}:rate={FPS}:start_scale=3.0:end_scale=0.4:end_pts={SECONDS * FPS}",
     "a colorful fractal zooming in"),
    ("life",
     f"life=size={W}x{H}:rate={FPS}:ratio=0.08:life_color=#FFFFFF:death_color=#000000",
     "a black and white cellular automaton simulation of cells evolving"),
    ("gradients",
     f"gradients=size={W}x{H}:rate={FPS}:speed=0.03",
     "a smooth colorful gradient slowly shifting and blending"),
]


def log(msg: str) -> None:
    print(f"[c2-feed] {msg}", file=sys.stderr, flush=True)


def read_source(graph: str, n_frames: int) -> np.ndarray:
    """ffmpeg lavfi -> raw rgb24 bytes on stdout -> (n, H, W, 3) uint8."""
    cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", graph,
           "-frames:v", str(n_frames), "-pix_fmt", "rgb24",
           "-f", "rawvideo", "pipe:1"]
    raw = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    got = len(raw) // (H * W * 3)
    if got < n_frames:
        raise RuntimeError(f"lavfi '{graph}': wanted {n_frames} frames, got {got}")
    return np.frombuffer(raw[: n_frames * H * W * 3], dtype=np.uint8).reshape(n_frames, H, W, 3).copy()


def main() -> None:
    n_per = SECONDS * FPS
    frames, meta = [], []
    for name, graph, caption in SOURCES:
        f = read_source(graph, n_per)
        frames.append(f)
        meta.append({"name": name, "lavfi": graph, "n_frames": int(f.shape[0]),
                     "intended_caption": caption})
        log(f"{name:10s} {f.shape[0]} frames {f.shape[1:]} (intended: {caption!r})")

    arr = np.concatenate(frames, axis=0)
    os.makedirs("c2_frames", exist_ok=True)
    np.save("c2_frames/frames_u8.npy", arr)
    with open("c2_frames/feed_meta.json", "w") as fh:
        json.dump({"fps": FPS, "width": W, "height": H, "seconds_per_source": SECONDS,
                   "sources": meta}, fh, indent=2)
    log(f"total {arr.shape[0]} frames -> c2_frames/frames_u8.npy ({arr.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
