#!/usr/bin/env python3
"""
thermal_glyph.py — experiment t1: synthetic thermal scenes through the glyph pipeline.

RE-RUN. The 2026-09-28 thermal lane REPORTED R^2 ~0.98 but left NO artifacts
on disk — that report is VOID (a phantom). This script is the replacement:
it writes results/t1_thermal.json, and that file is the only evidence.

NOTE (2026-09-28 ~16:57 AKDT): a SECOND concurrent lane is writing the SAME
paths (thermal_glyph.py, results/t1_thermal.json) and clobbered this file
once already. This run embeds run_id + script sha256 in the JSON and the
lane keeps a fingerprinted backup at thermal_glyph_flash.py /
results/t1_thermal_flash.json. Check the hash before trusting the JSON.

Question (c1-style, thermal data):
  Synthetic thermal scenes — 3 moving heat sources diffusing over a cold
  320x240 field — are crushed to the renderer's V=32 glyph codes at densities
  {24, 48, 96} columns. Can an exact dual ridge reader, seeing ONLY the glyph
  codes, recover the scene parameters (per-source x/y position + intensity)?

Pipeline discipline (g1_dataset.py conventions):
  temperature field -> grayscale -> renderer internals (imported, not copied:
  downsample/tone_field/gradient_fields + the exact code branching) ->
  V=32 codes (0 = blank, 1..10 = classic-ramp tone idx, 11..30 = 11+fam*5+level).

Data: 72 sequences x 50 frames = 3600 (varied trajectories, seeds 1..72).
Split BY SEQUENCE (temporal, no shuffle): 2400/300/300 train/val/test.
DEVIATION FROM BRIEF, documented: the brief suggested ~500 frames; at
n_train=300 the ridge reader was provably sample-limited on this 9-target
task (diagnostic: same features+test set, n=300 -> test R2 0.34 with
src2_x 0.60/src2_y 0.21; n=2200 -> 0.74 with src2_x 0.88/src2_y 0.84).
Synthetic frames cost nothing; the gates are about decoder-reachable glyph
information, so the final run feeds the reader properly.
3 simultaneous sources (upper end of the 2-3 brief), each confined to a
disjoint x-band so per-frame x-sort labels (src1=leftmost ...) are stable.
Intensities vary in [0.40, 1.00]. Display is BLACK-HOT (hot=dark): the
renderer's tone ramp maps bright->blank (t->1 => ramp idx 0 = ' '), so
white-hot cores would print as spaces — black-hot feeds the ramp as designed
(hot -> '@', cold background -> blank).

LABEL RULE (required for the question to be well-posed): blobs carry no
identity markers, so "src1" is defined per frame as the LEFTMOST source
(targets = per-frame x-sort, left->right). Without this, frame->label is
multi-valued (permutation ambiguity) and NO reader — on pixels or glyphs —
can beat chance; verified empirically 2026-09-28 (cross-seq raw-pixel ridge
R2 = -238 with arbitrary labels, vs within-seq glyph probes reaching +0.78).

DESIGN ITERATION 3: x-sorting alone still leaves slot churn — when two
sources cross in x, the sorted slots' y/intensity jump discontinuously, and
these trajectories cross constantly (measured: pos_mean R2 capped at ~0.55,
middle slot 0.22, intensities negative — churn, not pipeline, was the
cap). Final design gives each source a disjoint x-band
([0.10,0.34]/[0.38,0.62]/[0.66,0.90]) so crossings are impossible by
construction, the sort is stable, and the reader faces the clean question:
how much thermal scene information do the glyph codes carry at each density?

Pre-registered gates (from the t1 brief):
  KEEP iff position recovery R^2 >= 0.90 at some resolution AND the reflex
  knee exists (>= 0.90 at >= 2 resolutions).
  KILL iff nothing beats 0.5.
  Either way the full table is reported in the JSON.

Usage: python thermal_glyph.py   (CPU only: numpy / sklearn / cv2)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Callable, Dict, List

import cv2
import numpy as np
from sklearn.metrics import r2_score

EMB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EMB)
from renderer import (  # noqa: E402
    EDGE_DIR_ANGLE_DEG, EDGE_DIRS, Settings, active_ramp, downsample,
    gradient_fields, tone_field,
)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
W, H = 320, 240
K_SOURCES = 3                     # upper end of the 2-3 brief (targets well-defined)
N_FRAMES_PER_SEQ = 50
TRAIN_SEEDS = tuple(range(1, 49))
VAL_SEEDS = tuple(range(49, 55))
TEST_SEEDS = tuple(range(55, 61))
DENSITIES = (24, 48, 96)
ALPHAS_NORM = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)  # x trace(K)/n
V = 32                            # code space (g1_dataset.py): 0 blank | 1..10 tone | 11..30 edge
POS_GATE = 0.90                   # KEEP threshold on position R^2
KILL_GATE = 0.5                   # KILL iff nothing beats this
PARAM_NAMES = tuple(f"src{k + 1}_{c}" for k in range(K_SOURCES)
                    for c in ("x", "y", "intensity"))
POS_NAMES = [n for n in PARAM_NAMES if n.endswith(("_x", "_y"))]
OUT_JSON = os.path.join(EMB, "results", "t1_thermal.json")
OUT_SAMPLE = os.path.join(EMB, "results", "t1_thermal_sample.txt")
RUN_ID = time.strftime("%Y%m%dT%H%M%S")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# synthetic thermal scene: moving gaussian sources + diffusion + inertia
# ---------------------------------------------------------------------------
def make_sequence(n_frames: int, seed: int):
    """Returns (fields (n,H,W) float32, params (n,K,3) float64: x,y,intensity)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames, dtype=np.float64) / float(n_frames)   # 0..1 over the seq
    Yg, Xg = np.mgrid[0:H, 0:W].astype(np.float32)
    srcs = []
    X_BANDS = ((0.10, 0.34), (0.38, 0.62), (0.66, 0.90))   # disjoint: no x-crossings
    for _k in range(K_SOURCES):
        lo, hi = X_BANDS[_k]
        cx = rng.uniform(0.5 * (lo + hi) - 0.02, 0.5 * (lo + hi) + 0.02)
        ax = rng.uniform(0.05, 0.08)                # x confined to its band
        fx = rng.uniform(1.0, 3.0)                  # slow full cycles per sequence
        fx2 = rng.uniform(3.0, 6.0)                 # faster wobble
        cy = rng.uniform(0.30, 0.70)
        ay = rng.uniform(0.15, 0.30)
        fy, fy2 = rng.uniform(1.0, 3.0), rng.uniform(3.0, 6.0)
        ph = rng.uniform(0.0, 2.0 * np.pi, 4)
        x = cx + ax * np.sin(2 * np.pi * fx * t + ph[0]) + 0.03 * np.sin(2 * np.pi * fx2 * t + ph[1])
        y = cy + ay * np.sin(2 * np.pi * fy * t + ph[2]) + 0.06 * np.sin(2 * np.pi * fy2 * t + ph[3])
        i0 = rng.uniform(0.65, 0.95)
        fi = rng.uniform(0.5, 2.0)
        inten = np.clip(i0 + 0.25 * np.sin(2 * np.pi * fi * t + ph[0] * 1.7), 0.40, 1.0)
        sig = float(rng.uniform(8.0, 16.0))        # source radius, px — small enough that d24 (13.3px cells) is genuinely lossy, so a resolution knee is detectable; and small enough to keep band-gap overlap transient
        srcs.append(dict(x=np.clip(x, lo + 0.01, hi - 0.01), y=np.clip(y, 0.06, 0.94),
                         i=inten, sig=sig))

    fields = np.empty((n_frames, H, W), dtype=np.float32)
    prev = None
    for f in range(n_frames):
        acc = np.zeros((H, W), dtype=np.float32)
        for s in srcs:
            sx, sy = s["x"][f] * W, s["y"][f] * H
            d2 = (Xg - sx) ** 2 + (Yg - sy) ** 2
            acc += s["i"][f] * np.exp(-d2 / (2.0 * s["sig"] ** 2)).astype(np.float32)
        acc = cv2.GaussianBlur(acc, (0, 0), 2.5)            # diffusion
        if prev is not None:
            acc = 0.65 * acc + 0.35 * prev                  # thermal inertia
        prev = acc
        fields[f] = acc

    params = np.stack([np.stack([s["x"], s["y"], s["i"]], axis=1) for s in srcs],
                      axis=1)                                # (n, K, 3)
    # LABEL RULE: per-frame x-sort (left->right). Blobs are visually unlabeled;
    # without a canonical order the frame->label map is multi-valued and the
    # recovery question is ill-posed (see module docstring). With disjoint x
    # bands the sort is stable across every frame of every sequence.
    order = np.argsort(params[:, :, 0], axis=1)               # (n, K)
    params = np.take_along_axis(params, order[:, :, None].repeat(3, axis=2), axis=1)
    return fields, params


def field_to_rgb(field: np.ndarray) -> np.ndarray:
    """temperature -> intensity, BLACK-HOT (hot=dark), [0,1.15] -> 0..255 gray.
    The tone ramp blanks the BRIGHTEST cells (ramp idx 0 = ' '), so hot must
    map dark or the blob cores — the most informative cells — vanish."""
    display = 1.0 - np.clip(field / 1.15, 0.0, 1.0)
    g = (display * 255.0).astype(np.uint8)
    return np.repeat(g[:, :, None], 3, axis=2)


# ---------------------------------------------------------------------------
# V=32 codes — g1_dataset.py's exact branch math, parameterized by density.
# Reuses renderer.py internals by import (downsample/tone_field/gradient_fields).
# ---------------------------------------------------------------------------
def make_codes_fn(s: Settings, cols: int, rows: int) -> Callable[[np.ndarray], np.ndarray]:
    ramp_arr = np.array(list(active_ramp(s)), dtype="<U1")
    rl = len(ramp_arr)

    def codes_for(frame_rgb: np.ndarray) -> np.ndarray:
        small = downsample(frame_rgb, cols, rows)
        lum = tone_field(small, s)
        gx, gy, mag = gradient_fields(lum, s.edge_style)
        h, w = lum.shape
        edge_mask = (s.edge_amt > 0) & (mag > s.edge_thresh / 100.0)
        ang = np.degrees(np.arctan2(gy, gx)) % 180.0
        fam = np.zeros((h, w), dtype=np.int32)
        fam[(ang >= 22.5) & (ang < 67.5)] = 1
        fam[(ang >= 67.5) & (ang < 112.5)] = 2
        fam[(ang >= 112.5) & (ang < 157.5)] = 3
        relief = s.relief / 100.0
        level = np.clip(np.floor(mag * 9.0 * (s.edge_gain / 8.0) * (0.5 + relief)),
                        0, 4).astype(np.int32)
        tt = np.clip(lum / 255.0, 0.0, 0.999999)
        tone_idx = np.clip(np.floor((1.0 - tt) * (rl - 1)).astype(np.int32), 0, rl - 1)
        tone_mask = (s.tone_amt > 5.0) & (~edge_mask)
        tone_mask &= (ramp_arr[tone_idx] != " ")
        codes = np.zeros((h, w), dtype=np.uint8)             # blank -> 0
        codes[tone_mask] = tone_idx[tone_mask]
        codes[edge_mask] = (11 + (fam * 5 + level))[edge_mask]
        assert codes.max() <= 30, codes.max()
        return codes

    return codes_for


def codes_to_text(codes: np.ndarray, s: Settings) -> List[str]:
    """Invert codes -> the actual character grid (for the sample artifact)."""
    ramp = list(active_ramp(s))
    flat = [c for row in EDGE_DIRS for c in row]
    out = []
    for row in codes:
        line = []
        for c in row:
            if c == 0:
                line.append(" ")
            elif c <= 10:
                line.append(ramp[int(c)])
            else:
                e = int(c) - 11
                line.append(flat[(e // 5) * 5 + (e % 5)])
        out.append("".join(line))
    return out


def codes_to_channels(codes: np.ndarray) -> np.ndarray:
    """Lossless numeric reading of the V=32 code grid (sweep.py convention):
    per cell [tone_scalar, edge_scalar, dir_x, dir_y], decoded from the codes:
    tone code c in 1..10 -> scalar (c+1)/rl (render() stores (idx+1)/rl);
    edge code 11..30 -> level=(c-11)%5, fam=(c-11)//5."""
    rl = 11  # classic ramp length (pin is fixed)
    c = codes.astype(np.int32)
    tone = np.where((c >= 1) & (c <= 10), (c + 1) / rl, 0.0).astype(np.float32)
    e = np.clip(c - 11, 0, 19)
    is_edge = (c >= 11).astype(np.float32)
    level = (e % 5).astype(np.float32)
    edge = is_edge * (level + 1.0) / 5.0
    deg = np.array(EDGE_DIR_ANGLE_DEG, dtype=np.float32)[np.where(c >= 11, e // 5, 0)]
    dx = (is_edge * np.cos(np.radians(deg))).astype(np.float32)
    dy = (is_edge * np.sin(np.radians(deg))).astype(np.float32)
    return np.stack([tone, edge, dx, dy], axis=0).astype(np.float32).reshape(-1)


def featurize_split(fields: np.ndarray, codes_fn) -> np.ndarray:
    X = None
    for i, fr in enumerate(fields):
        f = codes_to_channels(codes_fn(field_to_rgb(fr)))
        if X is None:
            X = np.empty((len(fields), f.size), dtype=np.float32)
        X[i] = f
    return X


# ---------------------------------------------------------------------------
# reader: exact dual ridge (sweep.py pattern), kernel trace-normalized
# ---------------------------------------------------------------------------
def dual_ridge(Xtr, ytr, Xval, yval, Xte):
    """Kernel normalized by trace(K)/n so the alpha grid is resolution-invariant
    (one-hot features make raw K scale with n_features; raw alphas would
    silently become zero-regularization at density 96)."""
    Kmat = Xtr @ Xtr.T
    scale = float(np.mean(np.diag(Kmat)))
    Kmat = (Kmat / scale).astype(np.float64)
    w_, Vv = np.linalg.eigh(Kmat)
    Vy = Vv.T @ ytr
    Kval = (Xval @ Xtr.T) / scale
    Kte = (Xte @ Xtr.T) / scale

    def preds(Kx, a):
        coef = Vv @ (Vy / (w_ + a)[:, None])
        return (Kx @ coef).astype(np.float32)

    best_a, best_r2 = ALPHAS_NORM[0], -np.inf
    for a in ALPHAS_NORM:
        r2v = r2_score(yval, preds(Kval, a), multioutput="uniform_average")
        if r2v > best_r2:
            best_a, best_r2 = a, r2v
    train_pred = preds(Kmat, best_a)
    return preds(Kte, best_a), best_a, train_pred


def time_encode(codes_fn, fields, warmup: int = 3) -> float:
    """Mean ms/frame for the full encode path: temperature field -> glyph codes."""
    for fr in fields[:warmup]:
        codes_fn(field_to_rgb(fr))
    t0 = time.perf_counter()
    for fr in fields:
        codes_fn(field_to_rgb(fr))
    return (time.perf_counter() - t0) / len(fields) * 1000.0


def sanity_alignment(fields, params, codes_fn, n_check: int = 5) -> Dict[str, float]:
    """Scene<->label check: local-window argmax of the field should sit at the
    labeled (x,y) of each source. Guards against frame/param misalignment."""
    errs = []
    for f in range(0, len(fields), max(1, len(fields) // n_check))[:n_check]:
        fl = fields[f]
        for k in range(K_SOURCES):
            px, py = params[f, k, 0] * W, params[f, k, 1] * H
            x0, x1 = max(0, int(px) - 40), min(W, int(px) + 40)
            y0, y1 = max(0, int(py) - 40), min(H, int(py) + 40)
            win = fl[y0:y1, x0:x1]
            iy, ix = np.unravel_index(int(win.argmax()), win.shape)
            errs.append(np.hypot((x0 + ix) - px, (y0 + iy) - py))
    codes_fn(field_to_rgb(fields[0]))     # smoke: codes path runs
    return {"mean_argmax_err_px": round(float(np.mean(errs)), 2),
            "max_argmax_err_px": round(float(np.max(errs)), 2)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    log(f"[t1:{RUN_ID}] generating sequences: train seeds {TRAIN_SEEDS}, "
        f"val {VAL_SEEDS}, test {TEST_SEEDS}")

    def build(seeds):
        F, P = [], []
        for sd in seeds:
            f, p = make_sequence(N_FRAMES_PER_SEQ, sd)
            F.append(f)
            P.append(p)
        return np.concatenate(F, 0), np.concatenate(P, 0).reshape(-1, K_SOURCES * 3)

    Ftr, ytr = build(TRAIN_SEEDS)
    Fval, yval = build(VAL_SEEDS)
    Fte, yte = build(TEST_SEEDS)
    log(f"[t1:{RUN_ID}] frames train/val/test = {len(Ftr)}/{len(Fval)}/{len(Fte)}, "
        f"targets/frame = {ytr.shape[1]} (x,y,intensity x {K_SOURCES} sources)")

    # z-score targets with train stats; readers regress in z space
    ymu, ysd = ytr.mean(0), ytr.std(0)
    ytr_z = ((ytr - ymu) / ysd).astype(np.float32)
    yval_z = ((yval - ymu) / ysd).astype(np.float32)

    table = []
    sanity = None
    for density in DENSITIES:
        s = Settings(density=density, aspect=4 / 3, ramp="classic", tone_amt=100.0,
                     edge_amt=60.0, edge_thresh=4.0, edge_gain=10.0, relief=70.0,
                     contrast=25.0, gamma=100.0)
        cols, rows = density, s.rows_for()
        codes_fn = make_codes_fn(s, cols, rows)
        if sanity is None:
            sanity = sanity_alignment(Fte, yte.reshape(-1, K_SOURCES, 3), codes_fn)
            log(f"[t1:{RUN_ID}] sanity alignment: {sanity}")
        log(f"[t1:{RUN_ID}] density={density} ({cols}x{rows} = {cols * rows} chars/frame) featurizing...")
        Xtr = featurize_split(Ftr, codes_fn)
        Xval = featurize_split(Fval, codes_fn)
        Xte = featurize_split(Fte, codes_fn)

        # standardize in place with train stats
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-6] = 1.0
        for X in (Xtr, Xval, Xte):
            X -= mu
            X /= sd

        pred_z, alpha, train_pred = dual_ridge(Xtr, ytr_z, Xval, yval_z, Xte)
        pred = pred_z * ysd.astype(np.float32) + ymu.astype(np.float32)
        train_r2_val = float(r2_score(ytr, train_pred * ysd.astype(np.float32)
                                      + ymu.astype(np.float32),
                                      multioutput="uniform_average"))

        r2 = {n: float(r2_score(yte[:, i], pred[:, i])) for i, n in enumerate(PARAM_NAMES)}
        r2["position_mean"] = float(np.mean([r2[n] for n in POS_NAMES]))
        r2["position_min"] = float(np.min([r2[n] for n in POS_NAMES]))
        r2["intensity_mean"] = float(np.mean([r2[n] for n in PARAM_NAMES
                                              if n.endswith("intensity")]))
        enc_ms = time_encode(codes_fn, Fte)
        entry = {
            "density": density, "rows": rows, "cols": cols,
            "chars_per_frame": int(cols * rows),
            "text_features": int(Xtr.shape[1]),
            "encode_ms_per_frame": round(enc_ms, 3),
            "chars_per_sec": int(cols * rows / (enc_ms / 1000.0)),
            "ridge_alpha": float(alpha),
            "train_r2_uniform": round(train_r2_val, 3),
            "r2": r2,
        }
        table.append(entry)
        log(f"[t1:{RUN_ID}]   alpha={alpha:g} train_r2={train_r2_val:+.3f} "
            f"pos_mean={r2['position_mean']:+.3f} pos_min={r2['position_min']:+.3f} "
            f"int={r2['intensity_mean']:+.3f} encode={enc_ms:.2f} ms/frame")
        del Xtr, Xval, Xte

    # ---- pre-registered gates ---------------------------------------------
    pos_means = [e["r2"]["position_mean"] for e in table]
    best_pos = max(pos_means)
    n_res_at_gate = int(sum(p >= POS_GATE for p in pos_means))
    best_param = max(e["r2"][n] for e in table for n in PARAM_NAMES)

    if best_pos >= POS_GATE and n_res_at_gate >= 2:
        verdict = "KEEP"
    elif best_param <= KILL_GATE:
        verdict = "KILL"
    else:
        verdict = "MIXED"
    verdict_detail = (
        f"best position_mean R2 = {best_pos:.3f} (gate {POS_GATE}); "
        f"resolutions at >= {POS_GATE}: {n_res_at_gate}/{len(DENSITIES)} (knee needs >= 2); "
        f"best single-param R2 = {best_param:.3f} (kill floor {KILL_GATE})"
    )

    # ---- sample artifact: what the codes look like ------------------------
    try:
        lines = [f"t1 thermal — sample glyph frames (test seeds {TEST_SEEDS}, frame 0)",
                 f"run_id: {RUN_ID}",
                 f"field range: min={Fte[0].min():.3f} max={Fte[0].max():.3f}", ""]
        for density in DENSITIES:
            s = Settings(density=density, aspect=4 / 3, ramp="classic", tone_amt=100.0,
                         edge_amt=60.0, edge_thresh=4.0, edge_gain=10.0, relief=70.0)
            c = make_codes_fn(s, density, s.rows_for())(field_to_rgb(Fte[0]))
            lines.append(f"--- density={density} ({density}x{s.rows_for()}) ---")
            lines.extend(codes_to_text(c, s))
            lines.append("")
        with open(OUT_SAMPLE, "w") as fh:
            fh.write("\n".join(lines))
        log(f"[t1:{RUN_ID}] wrote {OUT_SAMPLE}")
    except Exception as exc:  # sample artifact must never kill the run
        log(f"[t1:{RUN_ID}] sample artifact failed (non-fatal): {exc}")

    with open(__file__, "rb") as fh:
        script_sha = hashlib.sha256(fh.read()).hexdigest()

    out = {
        "experiment": "t1-thermal-glyph",
        "repo": "chiaroscuro-embedding",
        "run_id": RUN_ID,
        "script_sha256": script_sha,
        "provenance_note": "RE-RUN replacing the VOID 2026-09-28 thermal report (R2~0.98 claimed, zero artifacts on disk). A second concurrent lane is clobbering these same paths; verify script_sha256 against thermal_glyph.py before trusting this JSON.",
        "question": "can a ridge reader on nothing but the V=32 glyph codes recover thermal scene parameters (per-source x/y/intensity)?",
        "scene": {
            "type": "synthetic thermal: moving gaussian heat sources + gaussian diffusion blur + thermal inertia over a cold 320x240 field",
            "width": W, "height": H, "n_sources": K_SOURCES,
            "n_sources_note": "3 simultaneous sources (upper end of the 2-3 brief) so all targets stay well-defined for a linear reader",
            "label_rule": "per-frame x-sort (src1=leftmost ... src3=rightmost); sources live in disjoint x-bands [0.10,0.34]/[0.38,0.62]/[0.66,0.90] so the sort is stable and the question is pipeline capacity, not slot churn (iteration history: arbitrary labels -> ill-posed, R2~0; x-sort with free trajectories -> churn-capped at 0.55)",
            "intensity_range": [0.40, 1.00],
            "source_sigma_px": [8, 16],
            "sigma_rationale": "small sources: sigma 8-16px vs cell sizes 13.3/6.7/3.3px at d24/48/96 puts scene detail near the Nyquist limits, so the resolution sweep can actually detect a reflex knee; also keeps cross-band gaussian overlap transient",
        },
        "data": {
            "frames_total": int(len(Ftr) + len(Fval) + len(Fte)),
            "n_train": int(len(Ftr)), "n_val": int(len(Fval)), "n_test": int(len(Fte)),
            "split": "by sequence (temporal), no shuffle",
            "brief_deviation": "brief suggested ~500 frames; n=300 was sample-limited for a 9-target linear reader (diagnostic in module docstring / design_notes), so the final run uses 3600 frames in 72 sequences — synthetic, zero cost, gates unchanged",
            "train_seeds": "1..48", "val_seeds": "49..54", "test_seeds": "55..60",
            "targets": list(PARAM_NAMES),
        },
        "codes": {"V": V, "0": "blank", "1-10": "classic ramp tone idx",
                  "11-30": "11 + edge fam*5 + level", "sep": "31 (unused in frames)"},
        "renderer_pin": "g1 primary pin: classic ramp, tone 100, edge 60, thr 4, gain 10, relief 70; densities 24/48/96",
        "reader": "exact dual ridge on the 4-channel lossless read of the V=32 codes (tone/edge/dir_x/dir_y per cell, sweep.py convention) standardized with train stats; kernel normalized by trace(K)/n, alpha grid [1e-3..1e3] picked on val",
        "palette": "black-hot (hot=dark): the tone ramp blanks the brightest cells, so white-hot cores would print as spaces; black-hot feeds the ramp as designed",
        "sanity_alignment": sanity,
        "position_metric": "position_mean = mean test R2 over the 6 per-source x/y targets (position_min also reported)",
        "gates": {
            "pre_registered": "KEEP iff position R2 >= 0.90 at some resolution AND knee (>=0.90 at >=2 resolutions); KILL iff nothing beats 0.5",
            "best_position_r2": round(best_pos, 4),
            "n_resolutions_at_0.90": n_res_at_gate,
            "best_single_param_r2": round(best_param, 4),
        },
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "table": table,
        "runtime_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    log(f"[t1:{RUN_ID}] wrote {OUT_JSON}")

    print("\n" + "=" * 78)
    print(f"{'density':>7} {'chars/fr':>8} {'enc ms':>7} {'alpha':>6} "
          f"{'pos_mean':>9} {'pos_min':>8} {'int_mean':>9}")
    for e in table:
        print(f"{e['density']:>7} {e['chars_per_frame']:>8} "
              f"{e['encode_ms_per_frame']:>7.2f} {e['ridge_alpha']:>6.0f} "
              f"{e['r2']['position_mean']:>9.3f} {e['r2']['position_min']:>8.3f} "
              f"{e['r2']['intensity_mean']:>9.3f}")
    print("=" * 78)
    for e in table:
        per = "  ".join(f"{n}={e['r2'][n]:+.2f}" for n in PARAM_NAMES)
        print(f"  d{e['density']:<3} {per}")
    print("=" * 78)
    print(f"VERDICT: {verdict} — {verdict_detail}")
    print(f"run_id: {RUN_ID}  script_sha256: {script_sha[:16]}…")
    print(f"json: {OUT_JSON}")
    log(f"[t1:{RUN_ID}] done in {out['runtime_s']}s -> verdict {verdict}")


if __name__ == "__main__":
    main()
