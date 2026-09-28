"""
c3_sweep.py — framerate vs pixel loss at a fixed CHARACTER budget.

The SEED claim: "chiaroscuro can crank framerate to make up for pixel loss."
Quantified here as: at a fixed chars/sec token budget, is information better
carried by MORE frames at LOWER glyph resolution, or FEWER frames at HIGHER
glyph resolution?

Design (pre-registered 2026-09-28):

  master feed : c3_feed.py — c2's five lavfi sources, 10 s each at 20 fps
                (1000 frames, 320x240). CLIP zero-shot labels EVERY master
                frame (c3_clip.py, c2's caption set verbatim).
  cells       : fps {2.5, 5, 10, 20} x density {40, 80, 160} = 12 cells.
                A cell samples every MASTER_FPS/fps-th master frame (strides
                8/4/2/1) and renders those frames through the chiaroscuro
                GLYPH pipeline at c2's default mix dials (c2_sweep.mix
                defaults, ramp=classic). The ONLY things that vary across
                cells are frames/sec and chars/frame.
  token budget: chars/sec = fps * cols * rows, rows = density*3/4 (4:3 aspect,
                renderer.rows_for), fixed-width frame — blank cells count:
                they are part of the channel. The grid contains FOUR exact
                matched-budget pairs, in each the high-fps cell has 4x the
                framerate and 1/4 the glyph area:
                  12,000 c/s : (2.5 fps, d80)  vs (10 fps, d40)
                  24,000 c/s : (5 fps,   d80)  vs (20 fps, d40)
                  48,000 c/s : (2.5 fps, d160) vs (10 fps, d80)
                  96,000 c/s : (5 fps,   d160) vs (20 fps, d80)
  reader      : text-only, IDENTICAL reader class and feature dimension for
                every cell: the rendered text frame is cut into a fixed 8x6
                grid of spatial blocks; each block contributes a histogram
                over the glyph vocabulary (classic ramp + edge glyph set,
                V=23). Feature dim = 8*6*23 = 1104 for every cell, so reader
                capacity never confounds the budget comparison.
  variants    : single — features of the current frame's text only.
                stacked — the last k=4 frames' text concatenated (temporal
                context). Fixed k gives LOW-fps cells a LONGER temporal reach
                per stack (1.6 s at 2.5 fps vs 0.2 s at 20 fps) — their fair
                shot. Stacked budget = 4x single, so matched pairs stay
                matched.
  task        : predict the CLIP label of each sampled frame from its text.
                Readers are c2's exact readers (dual ridge one-hot + PCA->MLP),
                reused by import. Blocked 5-fold CV: each source's sub-sampled
                frames split into 5 contiguous temporal blocks; fold = test on
                block i across all sources, train on the rest; ridge alpha in
                {10,100,1000} picked on a val slice (last 15% of the training
                frames, per source). Reported accuracy pools the test folds —
                every sampled frame is a test target exactly once, and no
                reader ever trains on its test block.
  labels      : NO margin filter — label noise is identical across cells and
                the comparison is paired; margin stats reported for context.

Gates (pre-registered):
  delta(pair) = acc(high-fps cell) - acc(low-fps cell), accuracy points.
  FRAMERATE-WINS  : >= 2 of 4 matched pairs with delta >= +5 AND more pairs
                    favor high-fps than favor low-fps.
  RESOLUTION-WINS : the reverse (>= 2 pairs with delta <= -5, more than favor
                    high-fps).
  TIE otherwise. Headline verdict = the STACKED variant (low-fps cells got
  their fair shot); the single-frame variant is reported as an ablation.

Usage:  python3 c3_sweep.py            # stdout = ONE JSON verdict
        C3_SMOKE=1 python3 c3_sweep.py # tiny plumbing check
Progress -> stderr. Requires c3_feed.py + c3_clip.py outputs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

from renderer import EDGE_DIRS, RAMPS, Settings, render
from c2_sweep import ALPHAS as C2_ALPHAS  # (10.0, 100.0, 1000.0)
from c2_sweep import mlp_reader, ridge_onehot, standardize

MASTER_FPS = 20
FPSS = (2.5, 5.0, 10.0, 20.0)
DENSITIES = (40, 80, 160)
ASPECT = 4.0 / 3.0
K_STACK = 4
N_FOLDS = 5
VAL_FRAC = 0.15
ALPHAS = C2_ALPHAS
N_COL_BLOCKS, N_ROW_BLOCKS = 8, 6
SMOKE = bool(os.environ.get("C3_SMOKE"))
GATE_PTS = 5.0
GATE_MIN_PAIRS = 2

# c2_sweep.mix() default dials — the render settings held FIXED across all cells
MIX_DIALS = dict(ramp="classic", tone_amt=100, edge_amt=60, edge_thresh=4,
                 edge_gain=10, relief=70, contrast=25, gamma=100)

# matched (high_fps_cell, low_fps_cell, budget) — see docstring
PAIRS: Tuple[Tuple[Tuple[float, int], Tuple[float, int], int], ...] = (
    ((2.5, 80), (10.0, 40), 12000),
    ((5.0, 80), (20.0, 40), 24000),
    ((2.5, 160), (10.0, 80), 48000),
    ((5.0, 160), (20.0, 80), 96000),
)

VOCAB = sorted(set(RAMPS["classic"]) | {c for row in EDGE_DIRS for c in row})
V = len(VOCAB)
CODE = {c: i for i, c in enumerate(VOCAB)}


def log(msg: str) -> None:
    print(f"[c3-sweep] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# text -> features (the reader sees ONLY the characters)
# ---------------------------------------------------------------------------
def text_features(text: List[str], cols: int, rows: int) -> np.ndarray:
    """Block-pooled glyph histograms of a rendered text frame.

    Fixed 8x6 spatial blocks; per block, fraction of each vocabulary glyph.
    Feature dim is identical for every (fps, density) cell: 8*6*V = 1104.
    """
    arr = np.array([list(row.ljust(cols)) for row in text], dtype="<U1")
    col_e = np.linspace(0, cols, N_COL_BLOCKS + 1).astype(int)
    row_e = np.linspace(0, rows, N_ROW_BLOCKS + 1).astype(int)
    feats = np.empty(N_ROW_BLOCKS * N_COL_BLOCKS * V, dtype=np.float32)
    k = 0
    for bi in range(N_ROW_BLOCKS):
        for bj in range(N_COL_BLOCKS):
            blk = arr[row_e[bi]:row_e[bi + 1], col_e[bj]:col_e[bj + 1]].ravel()
            u, cnt = np.unique(blk, return_counts=True)
            h = np.zeros(V, dtype=np.float32)
            for ch, c in zip(u, cnt):
                j = CODE.get(str(ch))
                if j is not None:
                    h[j] = float(c)
            feats[k:k + V] = h / blk.size
            k += V
    return feats


def stack_features(F: np.ndarray, n_per: List[int], k: int) -> np.ndarray:
    """Concatenate the last k frames' features within each source (pad by
    repeating the earliest frame at source start). Order: oldest -> newest."""
    D = F.shape[1] * k
    out = np.empty((F.shape[0], D), dtype=F.dtype)
    pos = 0
    for n in n_per:
        Fs = F[pos:pos + n]
        for t in range(n):
            out[pos + t] = np.concatenate(
                [Fs[max(0, t - (k - 1 - j))] for j in range(k)], axis=0)
        pos += n
    return out


# ---------------------------------------------------------------------------
# sampling + folds
# ---------------------------------------------------------------------------
def sub_indices(fps: float, per_source: List[int]) -> np.ndarray:
    """Master-frame indices for one cell framerate (source-major order)."""
    stride = int(round(MASTER_FPS / fps))
    out, base = [], 0
    for cnt in per_source:
        out.append(np.arange(base, base + cnt, stride))
        base += cnt
    return np.concatenate(out)


def make_folds(n_per: List[int]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Blocked 5-fold CV over positional indices (source-major order).

    Each source's frames split into N_FOLDS contiguous temporal blocks;
    fold = test on block i (all sources), train on the rest, val = last
    VAL_FRAC of the training frames per source (contiguous tail).
    """
    bounds = np.concatenate([[0], np.cumsum(n_per)])
    chunks = [np.array_split(np.arange(bounds[s], bounds[s + 1]), N_FOLDS)
              for s in range(len(n_per))]
    folds = []
    for f in range(N_FOLDS):
        te = np.concatenate([c[f] for c in chunks])
        tr_parts, va_parts = [], []
        for c in chunks:
            trc = np.concatenate([c[j] for j in range(N_FOLDS) if j != f])
            n_va = int(round(VAL_FRAC * len(trc)))
            if len(trc) > 1:
                n_va = min(n_va, len(trc) - 1)
            va_parts.append(trc[-n_va:] if n_va > 0 else trc[:0])
            tr_parts.append(trc[:-n_va] if n_va > 0 else trc)
        folds.append((np.concatenate(tr_parts), np.concatenate(va_parts), te))
    return folds


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def eval_cell(X: np.ndarray, y: np.ndarray,
              folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict:
    """Pooled CV accuracy with c2's exact readers. Every sample is a test
    target exactly once; the val slice picks the ridge alpha per fold."""
    y = y.astype(np.int64)
    preds = {"ridge": np.full_like(y, -1), "mlp": np.full_like(y, -1)}
    for tr, va, te in folds:
        Xtr, Xva, Xte = standardize(X[tr], X[va], X[te])
        ytr, yva, yte = y[tr], y[va], y[te]
        _, _, p = ridge_onehot(Xtr, ytr, Xva, yva, Xte, yte, ALPHAS)
        preds["ridge"][te] = p
        if len(tr) >= 30 and len(np.unique(ytr)) > 1:
            _, mp = mlp_reader(Xtr, ytr, Xte, yte)
            preds["mlp"][te] = mp
    accs = {name: float((pr == y).mean()) for name, pr in preds.items()
            if (pr >= 0).all()}
    best_name = max(accs, key=accs.get)
    return {"ridge_test_acc": round(accs.get("ridge", float("nan")), 4),
            "mlp_test_acc": round(accs.get("mlp", float("nan")), 4),
            "best_reader": best_name,
            "best_test_acc": round(accs[best_name], 4)}


def run_gate(cells: List[Dict], key: str) -> Dict:
    lut = {(c["fps"], c["density"]): c for c in cells}
    rows, hi_wins, lo_wins = [], 0, 0
    for (hi, lo, budget) in PAIRS:
        c_hi, c_lo = lut[hi], lut[lo]
        a_hi, a_lo = c_hi[key]["best_test_acc"], c_lo[key]["best_test_acc"]
        delta = round((a_hi - a_lo) * 100, 2)
        rows.append({
            "budget_chars_per_sec": budget,
            "high_fps_cell": {"fps": hi[0], "density": hi[1], "acc": a_hi},
            "low_fps_cell": {"fps": lo[0], "density": lo[1], "acc": a_lo},
            "delta_acc_pts_high_minus_low": delta,
            "favors": "high-fps" if delta >= GATE_PTS else
                      ("low-fps" if delta <= -GATE_PTS else "even")})
        if delta >= GATE_PTS:
            hi_wins += 1
        elif delta <= -GATE_PTS:
            lo_wins += 1
    if hi_wins >= GATE_MIN_PAIRS and hi_wins > lo_wins:
        verdict = "FRAMERATE-WINS"
    elif lo_wins >= GATE_MIN_PAIRS and lo_wins > hi_wins:
        verdict = "RESOLUTION-WINS"
    else:
        verdict = "TIE"
    return {"pairs": rows, "pairs_favoring_high_fps": hi_wins,
            "pairs_favoring_low_fps": lo_wins, "verdict": verdict}


# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    frames = np.load("c3_frames/frames_u8.npy")
    clip = np.load("c3_frames/clip_labels.npz")
    labels = clip["labels"]
    with open("c3_frames/feed_meta.json") as fh:
        feed_meta = json.load(fh)
    clip_summary = json.load(open("c3_frames/clip_summary.json"))
    per_source = [m["n_frames"] for m in feed_meta["sources"]]
    log(f"master {frames.shape[0]} frames, labels {labels.shape}, vocab V={V} "
        f"(feature dim/cell = {N_ROW_BLOCKS * N_COL_BLOCKS * V})")

    # ---- render + featurize every master frame at each density -------------
    FEATS: Dict[int, np.ndarray] = {}
    for d in DENSITIES:
        st = Settings(density=d, aspect=ASPECT, **MIX_DIALS)
        rows = st.rows_for()
        F = np.empty((frames.shape[0], N_ROW_BLOCKS * N_COL_BLOCKS * V), dtype=np.float32)
        ts = time.time()
        for i in range(frames.shape[0]):
            F[i] = text_features(render(frames[i], st).text, d, rows)
        FEATS[d] = F
        log(f"rendered+featurized density={d} ({d}x{rows} chars/frame, "
            f"{frames.shape[0]} frames) in {time.time() - ts:.1f}s")

    # ---- cells --------------------------------------------------------------
    cells: List[Dict] = []
    for fps in FPSS:
        stride = int(round(MASTER_FPS / fps))
        sub = sub_indices(fps, per_source)
        n_per = [len(range(0, cnt, stride)) for cnt in per_source]
        folds = make_folds(n_per)
        y = labels[sub]
        prior = float(np.bincount(y).max() / len(y))
        for d in DENSITIES:
            st = Settings(density=d, aspect=ASPECT, **MIX_DIALS)
            cpf = d * st.rows_for()
            log(f"cell fps={fps} density={d}: {len(sub)} samples, "
                f"{cpf} chars/frame, {fps * cpf:.0f} chars/sec")
            F = FEATS[d][sub]
            single = eval_cell(F, y, folds)
            stacked = eval_cell(stack_features(F, n_per, K_STACK), y, folds)
            cells.append({
                "fps": fps, "density": d, "chars_per_frame": cpf,
                "chars_per_sec": round(fps * cpf), "n_samples": int(len(sub)),
                "prior_majority": round(prior, 4),
                "single": single,
                "stacked": {**stacked, "chars_per_sec": round(K_STACK * fps * cpf)}})
            log(f"  single: ridge={single['ridge_test_acc']:.3f} "
                f"mlp={single['mlp_test_acc']:.3f} best={single['best_test_acc']:.3f} | "
                f"stacked: best={stacked['best_test_acc']:.3f}")

    cells.sort(key=lambda c: c["chars_per_sec"])

    # ---- gates --------------------------------------------------------------
    gate_single = run_gate(cells, "single")
    gate_stacked = run_gate(cells, "stacked")
    verdict = gate_stacked["verdict"]  # headline: stacked = low-fps fair shot
    detail = (f"stacked gate: {gate_stacked['verdict']} "
              f"({gate_stacked['pairs_favoring_high_fps']} pairs favor high-fps, "
              f"{gate_stacked['pairs_favoring_low_fps']} favor low-fps, "
              f"need >= {GATE_MIN_PAIRS} at >= {GATE_PTS:.0f} pts); "
              f"single-frame ablation: {gate_single['verdict']} "
              f"({gate_single['pairs_favoring_high_fps']}/"
              f"{gate_single['pairs_favoring_low_fps']})")
    if gate_single["verdict"] != gate_stacked["verdict"]:
        detail += " — VARIANTS DISAGREE"

    out = {
        "experiment": "c3-framerate-vs-pixels",
        "repo": "chiaroscuro-embedding",
        "question": ("at a fixed character budget (chars/sec), is information better "
                     "carried by MORE frames at LOWER glyph resolution, or FEWER frames "
                     "at HIGHER glyph resolution? (SEED: 'chiaroscuro can crank framerate "
                     "to make up for pixel loss')"),
        "feed": {"sources": [m["name"] for m in feed_meta["sources"]],
                 "master_fps": MASTER_FPS, "seconds_per_source": feed_meta["seconds_per_source"],
                 "n_master_frames": int(frames.shape[0]), "resolution": "320x240",
                 "cell_strides": {str(fps): int(round(MASTER_FPS / fps)) for fps in FPSS}},
        "ground_truth": {
            "model": clip_summary["model"],
            "rule": "CLIP zero-shot argmax caption per MASTER frame (20 fps); every cell samples labels from this one labeling",
            "captions_n": len(clip_summary["captions"]),
            "margin": clip_summary["margin"],
            "argmax_caption_counts": clip_summary["argmax_caption_counts"],
            "per_source_agreement_with_intended": clip_summary["per_source_argmax_agreement"],
            "margin_filter": "NONE (pre-registered): label noise identical across cells; comparison is paired",
        },
        "protocol": {
            "render_dials": MIX_DIALS,
            "text_features": f"8x6 block-pooled histograms over glyph vocabulary (V={V}); dim = {N_ROW_BLOCKS * N_COL_BLOCKS * V} for EVERY cell (reader capacity constant)",
            "variants": {"single": "current frame's text only",
                         "stacked": f"last k={K_STACK} frames' text concatenated (low-fps cells get a longer temporal reach per stack; budget = {K_STACK}x single)"},
            "readers": "c2's exact readers, reused by import: dual ridge one-hot (alpha in {10,100,1000} on val) + PCA->MLP(64)",
            "cv": f"blocked {N_FOLDS}-fold: contiguous temporal blocks per source; every sampled frame is a test target exactly once; val = last {VAL_FRAC:.0%} of train per source",
            "budget_rule": "chars/sec = fps * cols * rows (fixed-width text frame, blanks count)",
        },
        "gates": {"rule": f"FRAMERATE-WINS iff >= {GATE_MIN_PAIRS}/4 matched pairs have acc(high-fps) - acc(low-fps) >= +{GATE_PTS:.0f} pts and more pairs favor high-fps; RESOLUTION-WINS the reverse; else TIE. Headline = stacked variant.",
                  "matched_pairs": [{"budget_chars_per_sec": b,
                                     "high": {"fps": h[0], "density": h[1]},
                                     "low": {"fps": l[0], "density": l[1]}}
                                    for (h, l, b) in PAIRS],
                  "single": gate_single, "stacked": gate_stacked},
        "verdict": verdict,
        "verdict_detail": detail,
        "cells": cells,
        "runtime_s": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=2))  # THE verdict — stdout stays pure JSON
    log(f"done in {out['runtime_s']}s -> {verdict}")


if __name__ == "__main__":
    main()
