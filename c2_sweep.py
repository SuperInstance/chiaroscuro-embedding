"""
sweep for c2 — renderer sweep vs VLM ground truth.

The GAN-ish loop from SEED.md: a vision model (CLIP ViT-B/32, zero-shot)
interprets each feed frame -> its argmax caption is the ground truth -> the
chiaroscuro renderer renders the SAME frames to text at N settings -> a
text-only reader (ridge on native glyph features + PCA->MLP classifier)
predicts the vision model's label from the rendered text alone. The generator
sweeps; the discriminator is a real interpreter.

Controls / floors (pre-registered):
  caption prior  = majority-class frequency in the test split (chance for accuracy)
  pixels control = the SAME ridge-one-hot reader on CLIP image embeddings
                   (raw-pixels probe: what a linear read of pixel-space gets)

Splits are TEMPORAL per source (first 70% / mid 15% / last 15% of each
source's frames in feed order, after the margin filter) — no random shuffling:
frames within a feed are autocorrelated, so a random split would leak the
future into training. Readers must generalize forward in time.

Verdict gate (pre-registered, from the task):
  KEEP  iff some setting's best text reader beats the caption prior by >= 15
        accuracy points AND reaches >= 60% of the raw-pixels control accuracy.
  KILL  iff NO setting beats the prior by >= 15 points (text carries no
        interpretable content for the vision model's labels).
  INCONCLUSIVE otherwise (beats prior but can't reach 60% of pixel control).

Usage:
  python3 c2_sweep.py                # full: 34 settings (c1 grid), stdout = ONE JSON verdict
  C2_SMOKE=1 python3 c2_sweep.py     # smoke: subset grid
Progress -> stderr. Requires c2_feed.py + c2_clip.py outputs.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from renderer import RAMPS, Settings, render

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SMOKE = bool(os.environ.get("C2_SMOKE"))
MARGIN_THRESHOLD = 0.10          # must match c2_clip.py
SPLIT = (0.70, 0.15, 0.15)       # temporal train/val/test fraction per source
ALPHAS = (10.0, 100.0, 1000.0)
CONTROL_ALPHAS = (1.0, 10.0, 100.0)
PCA_DIMS = 100
KEEP_MARGIN_PTS = 15.0           # accuracy points over prior
KEEP_CONTROL_RATIO = 0.60        # fraction of pixels-control accuracy
TOP_K = 10


def log(msg: str) -> None:
    print(f"[c2-sweep] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# sweep grid — c1's exact grid conventions (34 settings)
# ---------------------------------------------------------------------------
def build_grid() -> List[Tuple[str, Settings]]:
    g: List[Tuple[str, Settings]] = []

    def mix(**kw) -> Settings:
        d = dict(density=100, aspect=4 / 3, ramp="classic", tone_amt=100,
                 edge_amt=60, edge_thresh=4, edge_gain=10, relief=70,
                 contrast=25, gamma=100)
        d.update(kw)
        return Settings(**d)

    for ramp in RAMPS:
        g.append((f"tone|ramp={ramp}", mix(edge_amt=0, ramp=ramp)))
    for thr in (2, 4, 8):
        g.append((f"edge|thr={thr}", mix(tone_amt=0, edge_amt=100, edge_thresh=thr)))
    g.append(("edge|style=emboss", mix(tone_amt=0, edge_amt=100, edge_style="emboss")))
    g.append(("edge|style=laplace", mix(tone_amt=0, edge_amt=100, edge_style="laplace")))
    for ramp in ("classic", "fine", "blocks", "minimal", "binary"):
        g.append((f"mix|ramp={ramp}", mix(ramp=ramp)))
    for dcol in (40, 60, 80, 130):
        g.append((f"mix|density={dcol}", mix(density=dcol)))
    for ea in (10, 25, 50):
        g.append((f"mix|edgeAmt={ea}", mix(edge_amt=ea)))
    g.append(("preset|sculptor", mix(ramp="fine", edge_amt=100, edge_gain=8, relief=55)))
    g.append(("preset|blueprint", mix(ramp="minimal", tone_amt=25, edge_amt=100,
                                      edge_thresh=3, edge_gain=14, relief=70)))
    g.append(("preset|wireframe", mix(ramp="minimal", tone_amt=0, edge_amt=100,
                                      edge_thresh=5, edge_gain=15, relief=80)))
    g.append(("preset|green-matrix", mix(ramp="classic", edge_amt=20, edge_gain=10, relief=25)))
    g.append(("mix|contrast=10", mix(contrast=10)))
    g.append(("mix|contrast=60", mix(contrast=60)))
    g.append(("tone|posterize=3", mix(edge_amt=0, posterize=3)))
    g.append(("tone|posterize=6", mix(edge_amt=0, posterize=6)))
    return g


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_split() -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, Dict]:
    """Margin filter, then temporal 70/15/15 split per source. Returns
    (frames per split, labels per split, raw frame indices per split, info)."""
    frames = np.load("c2_frames/frames_u8.npy")
    clip = np.load("c2_frames/clip_labels.npz")
    with open("c2_frames/feed_meta.json") as fh:
        meta = json.load(fh)
    labels, margin = clip["labels"], clip["margin"]

    keep = margin >= MARGIN_THRESHOLD
    per_source = [m["n_frames"] for m in meta["sources"]]
    splits: List[List[int]] = [[], [], []]
    start = 0
    for cnt in per_source:
        idx = np.arange(start, start + cnt)[keep[start : start + cnt]]
        n = len(idx)
        n_tr = int(round(SPLIT[0] * n))
        n_va = int(round(SPLIT[1] * n))
        splits[0] += list(idx[:n_tr])
        splits[1] += list(idx[n_tr : n_tr + n_va])
        splits[2] += list(idx[n_tr + n_va :])
        start += cnt

    out_frames = [frames[np.array(ix, dtype=np.int64)] for ix in splits]
    out_labels = [labels[np.array(ix, dtype=np.int64)] for ix in splits]
    out_idx = [np.array(ix, dtype=np.int64) for ix in splits]
    info = {
        "n_raw": int(len(labels)),
        "n_kept": int(keep.sum()),
        "n_dropped_low_margin": int((~keep).sum()),
        "split_sizes": [int(len(ix)) for ix in splits],
        "kept_per_source": {},
        "kept_label_counts": {int(c): int((out_labels[0] == c).sum() +
                                          (out_labels[1] == c).sum() +
                                          (out_labels[2] == c).sum())
                              for c in np.unique(np.concatenate(out_labels))},
    }
    start = 0
    for m in meta["sources"]:
        cnt = m["n_frames"]
        info["kept_per_source"][m["name"]] = int(keep[start : start + cnt].sum())
        start += cnt
    return out_frames, out_labels, out_idx, info


def featurize(frames: List[np.ndarray], s: Settings) -> np.ndarray:
    feats = None
    for i, fr in enumerate(frames):
        r = render(fr, s)
        f = r.channels.reshape(-1).astype(np.float32)
        if feats is None:
            feats = np.empty((len(frames), f.size), dtype=np.float32)
        feats[i] = f
    return feats


def standardize(Xtr: np.ndarray, *rest: np.ndarray) -> Tuple[np.ndarray, ...]:
    """z-score ALL splits with TRAIN stats (no val/test leakage)."""
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return tuple((X - mu) / sd for X in (Xtr,) + rest)


# ---------------------------------------------------------------------------
# readers: identical reader class for text features AND the pixels control
# ---------------------------------------------------------------------------
def ridge_onehot(Xtr, ytr, Xval, yval, Xte, yte, alphas) -> Tuple[float, float, np.ndarray]:
    """Least-squares classification: dual ridge on one-hot targets, argmax.
    Alpha picked on val accuracy. Returns (val_acc, test_acc, test_pred)."""
    classes = np.unique(ytr)
    cmap = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(ytr), len(classes)), dtype=np.float64)
    for i, y in enumerate(ytr):
        Y[i, cmap[y]] = 1.0
    K = Xtr @ Xtr.T
    w, V = np.linalg.eigh(K)
    Vy = V.T @ Y
    Kval, Kte = Xval @ Xtr.T, Xte @ Xtr.T

    def preds(Kx):
        coef = V @ (Vy / (w + alphas[0])[:, None])
        return classes[(Kx @ coef).argmax(axis=1)]

    best_a, best_val = alphas[0], -1.0
    for a in alphas:
        coef = V @ (Vy / (w + a)[:, None])
        pv = classes[(Kval @ coef).argmax(axis=1)]
        acc = float((pv == yval).mean())
        if acc > best_val:
            best_a, best_val = a, acc
    coef = V @ (Vy / (w + best_a)[:, None])
    pte = classes[(Kte @ coef).argmax(axis=1)]
    return best_val, float((pte == yte).mean()), pte


def mlp_reader(Xtr, ytr, Xte, yte, seed=0) -> Tuple[float, np.ndarray]:
    n_comp = min(PCA_DIMS, Xtr.shape[0] - 1, Xtr.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed).fit(Xtr)
    Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
    sc = StandardScaler().fit(Ptr)
    net = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=1e-2,
                        max_iter=300, random_state=seed, tol=1e-3,
                        learning_rate_init=1e-3).fit(sc.transform(Ptr), ytr)
    pte = net.predict(sc.transform(Pte))
    return float((pte == yte).mean()), pte


# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    frames, labels, idxs, split_info = load_split()
    ytr, yval, yte = labels
    n_class = len(split_info["kept_label_counts"])
    prior_majority = float(np.bincount(yte, minlength=n_class).max() / len(yte))
    log(f"splits {split_info['split_sizes']} (train/val/test), "
        f"{n_class} classes, test majority prior = {prior_majority:.3f}")

    # ---- pixels control: same reader, CLIP image embeddings ---------------
    clip_summary = json.load(open("c2_frames/clip_summary.json"))
    E = np.load("c2_frames/clip_labels.npz")["img_emb"]
    Etr, Eval, Ete = standardize(*(E[ix] for ix in idxs))
    c_val, control_acc, _ = ridge_onehot(Etr, ytr, Eval, yval, Ete, yte, CONTROL_ALPHAS)
    log(f"pixels control (ridge-onehot on CLIP img emb): val={c_val:.3f} test={control_acc:.3f}")

    grid = build_grid()
    if SMOKE:
        grid = [g for i, g in enumerate(grid) if i % 4 == 0][:8]
    log(f"sweeping {len(grid)} settings ...")

    results = []
    for k, (name, st) in enumerate(grid):
        tset = time.time()
        Xtr, Xval, Xte = standardize(featurize(frames[0], st),
                                     featurize(frames[1], st),
                                     featurize(frames[2], st))
        r_val, r_acc, _ = ridge_onehot(Xtr, ytr, Xval, yval, Xte, yte, ALPHAS)
        m_acc, _ = mlp_reader(Xtr, ytr, Xte, yte)
        best = max(r_acc, m_acc)
        results.append({
            "name": name,
            "dials": {k2: st.as_dict()[k2] for k2 in
                      ("density", "ramp", "tone_amt", "edge_amt", "edge_thresh",
                       "edge_gain", "edge_style", "relief", "contrast", "gamma", "posterize")},
            "n_text_features": int(Xtr.shape[1]),
            "ridge_val_acc": r_val,
            "ridge_test_acc": r_acc,
            "mlp_test_acc": m_acc,
            "best_test_acc": best,
            "beats_prior_pts": round((best - prior_majority) * 100, 2),
            "ratio_of_control": round(best / max(control_acc, 1e-9), 3),
            "secs": round(time.time() - tset, 1),
        })
        log(f"  [{k+1:2d}/{len(grid)}] {name:26s} ridge={r_acc:.3f} mlp={m_acc:.3f} "
            f"best={best:.3f} (+{(best - prior_majority) * 100:.1f} pts vs prior, "
            f"{best / max(control_acc, 1e-9):.0%} of control)")

    results.sort(key=lambda e: e["best_test_acc"], reverse=True)
    best_margin = max(e["beats_prior_pts"] for e in results)
    best_ratio = max(e["ratio_of_control"] for e in results)
    n_beating = sum(e["beats_prior_pts"] >= KEEP_MARGIN_PTS for e in results)
    any_keep = any(e["beats_prior_pts"] >= KEEP_MARGIN_PTS and
                   e["ratio_of_control"] >= KEEP_CONTROL_RATIO for e in results)
    if any_keep:
        verdict = "KEEP"
    elif best_margin < KEEP_MARGIN_PTS:
        verdict = "KILL"
    else:
        verdict = "INCONCLUSIVE"

    out = {
        "experiment": "c2-vlm-ground-truth",
        "repo": "chiaroscuro-embedding",
        "question": "which chiaroscuro render settings leave a real vision model's (CLIP zero-shot) interpretation of video readable from the rendered TEXT alone?",
        "loop": "ffmpeg lavfi feed -> CLIP zero-shot argmax caption per frame (the discriminator's ground truth) -> chiaroscuro glyph render of the SAME frames -> text-only reader predicts the caption index (the generator sweeps)",
        "feed": {"sources": ["testsrc2", "smptebars", "mandelbrot", "life", "gradients"],
                 "n_raw_frames": split_info["n_raw"], "fps": 10, "resolution": "320x240",
                 "note": "10 s per source; smptebars is static by design (easy class)"},
        "ground_truth": {
            "model": "openai/clip-vit-base-patch32 (cached, zero-shot, fp16 GPU)",
            "captions": clip_summary["captions"],
            "captions_n": n_class,
            "rule": "argmax caption per frame over softmax(cos*100); label = CLIP's interpretation, distractor-caption frames kept as-is",
            "margin_distribution": {"percentiles": clip_summary["margin"]["percentiles"],
                                     "mean": clip_summary["margin"]["mean"]},
            "argmax_caption_counts": clip_summary["argmax_caption_counts"],
            "per_source_agreement_with_intended": clip_summary["per_source_argmax_agreement"],
            "interpretation_note": "CLIP (single-frame) cannot separate testsrc2 from smptebars — both read as the calibration-bars caption; 3 life frames read as 'night sky full of stars'. The label is the discriminator's interpretation, not the generator's intent. Effective classes ~4 + tiny off-label groups.",
            "margin_filter": {"threshold": MARGIN_THRESHOLD,
                              "dropped": split_info["n_dropped_low_margin"],
                              "kept": split_info["n_kept"]},
        },
        "splits": {"scheme": "temporal 70/15/15 per source (feed order, after margin filter) — no random shuffle: adjacent frames are near-duplicates",
                   "sizes": split_info["split_sizes"],
                   "kept_per_source": split_info["kept_per_source"],
                   "label_counts_all": {str(k): v for k, v in split_info["kept_label_counts"].items()}},
        "readers": {"ridge": "least-squares classification: dual ridge on native text features vs one-hot, alpha in [10,100,1000] picked on val accuracy, argmax",
                    "mlp": "PCA-100 -> StandardScaler -> MLPClassifier(64)",
                    "pixels_control": "SAME ridge-onehot reader on L2-normalized CLIP image embeddings (raw-pixels probe)",
                    "text_features": "per cell: tone index / edge level / edge dir x/y (lossless read of the glyph grid)"},
        "scoring": {"prior": f"test majority-class frequency = {prior_majority:.4f}",
                    "gate": f"KEEP iff some setting beats prior by >= {KEEP_MARGIN_PTS:.0f} acc pts AND reaches >= {KEEP_CONTROL_RATIO:.0%} of pixels control; KILL iff nothing beats prior by {KEEP_MARGIN_PTS:.0f} pts; else INCONCLUSIVE"},
        "pixels_control_acc": control_acc,
        "verdict": verdict,
        "verdict_detail": f"best margin over prior = +{best_margin:.1f} pts, best ratio of pixels control = {best_ratio:.3f}, settings beating prior by >= {KEEP_MARGIN_PTS:.0f}: {n_beating}/{len(results)}",
        "n_settings": len(results),
        "top_k": results[:TOP_K],
        "settings": results,
        "runtime_s": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=2))  # THE verdict — stdout stays pure JSON
    log(f"done in {out['runtime_s']}s -> {verdict}")


if __name__ == "__main__":
    main()
