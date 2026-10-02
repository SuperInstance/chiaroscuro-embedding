#!/usr/bin/env python3
"""ie1_motion_codes.py — ie1: temporal fam codes (the lobula-plate build).

Per the insect-eye spec (quilt-gpu-lab/docs/chiaroscuro-insect-eye-2026-09-28.md,
build c / experiment ie1): Reichardt-style elementary motion detectors on
ADJACENT GLYPH CELLS -> four MOTION fam codes (mx+, mx-, my+, my-) pooled per
frame -> the motion-code stream -> a text-only ridge reader recovers the true
drift velocity the scene was rendered with.

Scene: dark field + STATIC distractors (radial key light, 8x8 checkerboard)
+ 2 bright gaussian blobs drifting with a GLOBAL velocity (vx, vy), drawn with
periodic wrap so nothing exits the field. 8 directions x 3 speeds x 2 scenes
= 48 scenes; 20 fps; 2 s each (1 warm-up frame + 40 coded frames).

Correlator (delay-and-multiply with mirror subtraction, baseline = 1 cell,
delay = 1 frame = 50 ms), S = the V=32 glyph code field the text reader sees
(tone 0..10, edge 11+; the codes_for_temp convention from t1):
    Cx(x,y,t) = S(x,y,t)*S(x-1,y,t-1) - S(x,y,t)*S(x+1,y,t-1)
    Cy(x,y,t) = S(x,y,t)*S(x,y-1,t-1) - S(x,y,t)*S(x,y+1,t-1)
    mx+ = sum max(Cx,0)   mx- = sum max(-Cx,0)   my+, my- likewise
(two half-wave-rectified pools per axis = the four T4/T5 populations).

Readers (ridge, t1 house hygiene), all trained per density:
  primary   : 4 pooled motion codes -> (sin theta, cos theta)   [GATE]
              and -> (vx, vy)                                   [reported]
  diagnostic: full per-cell motion field (4 x rows x cols) -> same targets
  reference : same 4-code pooling on the pre-glyph lum field (the cell's
              photoreceptor layer) — locates loss (glyph quantization vs pooling)

Gate (pre-registered, c1 house gate): KEEP iff primary direction mean R^2
>= 0.5 at >= 2 densities; KILL iff no density >= 0.5; else INCONCLUSIVE.

CPU-only (numpy/cv2; system python3). No torch, no GPU.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from renderer import (Settings, active_ramp, downsample, gradient_fields,
                      tone_field)

W, H = 320, 240
FPS = 20
SECONDS = 2.0
N_WARM = 1                     # warm-up frames so every coded frame has a t-1
N_FRAMES = int(FPS * SECONDS)  # 40 coded frames per scene
DIRECTIONS = tuple(float(a) for a in range(0, 360, 45))   # 8 compass dirs
SPEEDS = (0.10, 0.25, 0.45)    # fraction of frame WIDTH per second
REPS = 2                       # scenes per velocity combo
DENSITIES = (24, 48, 96)       # c1-stock densities (t1 precedent: 24/48/96)
ASPECT = 4.0 / 3.0
SEED = 42
N_SCENES = len(DIRECTIONS) * len(SPEEDS) * REPS  # 48

TARGETS = ("sin_theta", "cos_theta", "vx", "vy")


# ---------------------------------------------------------------------------
# the scene: static distractors + globally drifting gaussian blobs
# ---------------------------------------------------------------------------
class DriftScene:
    """Dark field + static key light + static checker + 2 drifting blobs.

    Blobs move with the SAME (vx, vy) — global drift is the ground truth —
    and are drawn with PERIODIC wrap so nothing exits the field (no
    appear/disappear transients in the correlator).
    """

    def __init__(self, theta_deg: float, speed_frac: float, scene_idx: int):
        self.W, self.H = W, H
        Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
        self.X, self.Y = X, Y

        rng = np.random.default_rng(1000 + scene_idx)
        # static radial key light (mock-style, frozen: no param confound)
        lx, ly = W * 0.5, H * 0.5
        d = np.sqrt((X - lx) ** 2 + (Y - ly) ** 2)
        self._bg = np.clip(1.0 - d / (H * 0.85), 0.0, 1.0) * 0.85
        self._bg = (0.04 + 0.92 * self._bg).astype(np.float32)  # 0.04..0.96

        # static checkerboard distractor (mock-style 8x8 patch)
        cb = np.zeros((H, W), dtype=bool)
        cx0, cy0 = int(300 * (W / 640.0)), int(200 * (W / 640.0))
        cell = max(3, int(16 * (W / 640.0)))
        for cy in range(8):
            for cx in range(8):
                if (cx + cy) % 2:
                    y0, x0 = cy0 + cy * cell, cx0 + cx * cell
                    cb[y0:y0 + cell, x0:x0 + cell] = True
        self._checker = cb

        # the two blobs (bright gaussians; per-scene look, shared motion)
        self.blobs = [
            dict(x=rng.uniform(0, W), y=rng.uniform(0, H),
                 sigma=rng.uniform(14, 22), I=rng.uniform(0.65, 1.0))
            for _ in range(2)
        ]

        # the ground truth: global drift velocity
        v_px = speed_frac * W                    # px/s
        th = np.radians(theta_deg)
        self.vx = float(v_px * np.cos(th))
        self.vy = float(v_px * np.sin(th))
        self.sin_t = float(np.sin(th))
        self.cos_t = float(np.cos(th))
        self.theta_deg = float(theta_deg)
        self.speed_frac = float(speed_frac)

    def _blob(self, cx, cy, sigma, intensity):
        dx = np.abs(self.X - cx); dx = np.minimum(dx, self.W - dx)
        dy = np.abs(self.Y - cy); dy = np.minimum(dy, self.H - dy)
        return intensity * np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))

    def light_field(self, t_frame: int) -> np.ndarray:
        """Scene luminance 0..1 at frame index t_frame (deterministic)."""
        t_s = t_frame / FPS
        L = self._bg.copy()
        for b in self.blobs:
            cx = (b["x"] + self.vx * t_s) % self.W
            cy = (b["y"] + self.vy * t_s) % self.H
            L = L + self._blob(cx, cy, b["sigma"], b["I"])
        L = np.where(self._checker, np.maximum(L, 0.85), L)
        return np.clip(L, 0.0, 1.0)

    def frame(self, t_frame: int) -> np.ndarray:
        """RGB uint8 frame for the glyph pipeline."""
        return (np.repeat(self.light_field(t_frame)[:, :, None], 3, axis=2)
                * 255).astype(np.uint8)

    def targets(self) -> np.ndarray:
        return np.array([self.sin_t, self.cos_t, self.vx, self.vy],
                        dtype=np.float64)


# ---------------------------------------------------------------------------
# glyph coding + the Reichardt correlator
# ---------------------------------------------------------------------------
def codes_and_lum(frame_rgb, s: Settings):
    """Glyph pipeline -> (V=32 code field, pre-glyph lum field).

    codes: tone_idx 0..10 on tone cells, 11+fam*5+level on edge cells,
    0 blank (identical convention to thermal_glyph.codes_for_temp).
    """
    cols, rows = int(s.density), s.rows_for()
    small = downsample(frame_rgb.astype(np.float32), cols, rows)
    lum = tone_field(small, s)
    gx, gy, mag = gradient_fields(lum, s.edge_style)
    ramp_arr = np.array(list(active_ramp(s)), dtype="<U1")
    h, w = lum.shape
    edge_mask = (s.edge_amt > 0) & (mag > s.edge_thresh / 100.0)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    fam = np.zeros((h, w), dtype=np.int32)
    fam[(ang >= 22.5) & (ang < 67.5)] = 1
    fam[(ang >= 67.5) & (ang < 112.5)] = 2
    fam[(ang >= 112.5) & (ang < 157.5)] = 3
    relief = s.relief / 100.0
    level = np.clip(np.floor(mag * 9.0 * (s.edge_gain / 8.0)
                             * (0.5 + relief)), 0, 4).astype(np.int32)
    t = np.clip(lum / 255.0, 0.0, 0.999999)
    tone_idx = np.clip(np.floor((1.0 - t) * 10).astype(np.int32), 0, 10)
    tone_mask = (s.tone_amt > 5.0) & (~edge_mask)
    tone_mask &= (ramp_arr[tone_idx] != " ")
    codes = np.zeros((h, w), dtype=np.float32)
    codes[tone_mask] = tone_idx[tone_mask]
    codes[edge_mask] = (11 + (fam * 5 + level))[edge_mask]
    return codes, lum


def reichardt_fields(S_cur: np.ndarray, S_prev: np.ndarray):
    """Signed per-cell correlator fields Cx, Cy (mirror subtraction).

    Motion in +x (image coords: rightward): content at (x-1, t-1) is at
    (x, t) now -> S(x,t)*S(x-1,t-1) dominates -> Cx > 0. Same for +y = down.
    """
    L = np.concatenate([S_prev[:, :1], S_prev[:, :-1]], axis=1)
    R = np.concatenate([S_prev[:, 1:], S_prev[:, -1:]], axis=1)
    U = np.concatenate([S_prev[:1, :], S_prev[:-1, :]], axis=0)
    D = np.concatenate([S_prev[1:, :], S_prev[-1:, :]], axis=0)
    Cx = S_cur * (L - R)
    Cy = S_cur * (U - D)
    return Cx, Cy


def pool4(Cx: np.ndarray, Cy: np.ndarray) -> np.ndarray:
    """Lattice aggregation -> the four MOTION fam codes per frame."""
    return np.array([Cx[Cx > 0].sum(), -Cx[Cx < 0].sum(),
                     Cy[Cy > 0].sum(), -Cy[Cy < 0].sum()], dtype=np.float64)


# ---------------------------------------------------------------------------
# readers (t1 house stock)
# ---------------------------------------------------------------------------
def ridge_fit_predict(Xtr, Ytr, Xte, lam=10.0):
    """Robust ridge: drop near-dead columns, center per column, scale globally.

    Solved in the DUAL (K + lam I) alpha = B — algebraically identical to the
    house primal solve(A^T A + lam I), but tractable for the 96-col motion
    field (27k features, n=1360 rows -> a 1360x1360 Gram matrix).
    """
    sd_col = Xtr.std(0)
    gstd = Xtr.std() + 1e-9
    keep = sd_col > 0.02 * gstd
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    mu = Xtr.mean(0)
    A = ((Xtr - mu) / gstd).astype(np.float64)
    C = ((Xte - mu) / gstd).astype(np.float64)
    mu_y, sd_y = Ytr.mean(0), Ytr.std(0) + 1e-9
    B = ((Ytr - mu_y) / sd_y).astype(np.float64)
    n = A.shape[0]
    alpha = np.linalg.solve(A @ A.T + lam * np.eye(n), B)
    pred = (C @ A.T @ alpha) * sd_y + mu_y
    return pred


LAMS = (10.0, 100.0, 1000.0)   # c1 house convention: alpha picked on val


def pick_lam_and_predict(X, Y, is_fit, is_val, test_pos):
    """c1 house lam selection: fit on fit rows, score mean R2 on val rows for
    each lam in LAMS, refit on fit+val with the winner, return (test predictions
    for rows test_pos, chosen lam)."""
    best_lam, best_score = None, -1e18
    for lam in LAMS:
        pred_v = ridge_fit_predict(X[is_fit], Y[is_fit], X[is_val], lam)
        score = float(np.mean(r2_per_col(Y[is_val], pred_v)))
        if score > best_score:
            best_score, best_lam = score, lam
    fv = is_fit | is_val
    pred = ridge_fit_predict(X[fv], Y[fv], X[test_pos], best_lam)
    return pred, best_lam


def r2_per_col(Y, pred):
    y = np.asarray(Y, float)
    p = np.asarray(pred, float)
    out = []
    for j in range(y.shape[1]):
        ss = ((y[:, j] - p[:, j]) ** 2).sum()
        st = ((y[:, j] - y[:, j].mean()) ** 2).sum() + 1e-12
        out.append(round(float(1 - ss / st), 4))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    scene_defs = [(th, sp, rep) for rep in range(REPS)
                  for sp in SPEEDS for th in DIRECTIONS]
    assert len(scene_defs) == N_SCENES

    # -------- pass 1 (sanity): does the detector point the right way? ------
    sc = DriftScene(0.0, 0.45, 0)   # pure rightward, fast
    s48 = Settings(density=48, ramp="classic", tone_amt=100.0, edge_amt=60.0,
                   edge_thresh=4.0, edge_gain=10.0, relief=70.0)
    prev = None
    sanity = None
    for t in range(N_FRAMES + 1):
        codes, _ = codes_and_lum(sc.frame(t), s48)
        if prev is not None:
            Cx, Cy = reichardt_fields(codes, prev)
            p = pool4(Cx, Cy)
            sanity = dict(theta_deg=0.0, speed_frac=0.45, density=48,
                          mx_plus=round(float(p[0]), 1),
                          mx_minus=round(float(p[1]), 1),
                          my_plus=round(float(p[2]), 1),
                          my_minus=round(float(p[3]), 1))
        prev = codes
    print(f"[ie1 sanity] rightward scene -> mx+ {sanity['mx_plus']} vs "
          f"mx- {sanity['mx_minus']} (want mx+ >> mx-)", flush=True)

    # -------- pass 2: full data build (one clean pass) ---------------------
    per_density = {d: dict(X_pool=[], X_field=[], X_lum=[], Y=[],
                           enc_ms=0.0, t_frames=0) for d in DENSITIES}
    settings = {d: Settings(density=d, ramp="classic", tone_amt=100.0,
                            edge_amt=60.0, edge_thresh=4.0, edge_gain=10.0,
                            relief=70.0) for d in DENSITIES}

    for idx, (th, sp, rep) in enumerate(scene_defs):
        sc = DriftScene(th, sp, idx)
        y = sc.targets()
        frames = [sc.frame(t) for t in range(N_WARM + N_FRAMES)]
        for d in DENSITIES:
            s = settings[d]
            pd = per_density[d]
            t0 = time.time()
            prev_c, prev_l = codes_and_lum(frames[0], s)  # warm-up frame = t-1
            for t in range(1, N_WARM + N_FRAMES):
                codes, lum = codes_and_lum(frames[t], s)
                Cx, Cy = reichardt_fields(codes, prev_c)
                Lx, Ly = reichardt_fields(lum.astype(np.float32),
                                          prev_l.astype(np.float32))
                pd["X_pool"].append(pool4(Cx, Cy))
                pd["X_lum"].append(pool4(Lx, Ly))
                pd["X_field"].append(np.concatenate([
                    np.maximum(Cx, 0).ravel(), np.maximum(-Cx, 0).ravel(),
                    np.maximum(Cy, 0).ravel(), np.maximum(-Cy, 0).ravel()]))
                pd["Y"].append(y)
                prev_c, prev_l = codes, lum
            pd["enc_ms"] += time.time() - t0
            pd["t_frames"] += N_FRAMES
        if (idx + 1) % 12 == 0:
            print(f"[ie1] scenes {idx + 1}/{N_SCENES} rendered", flush=True)

    # -------- readers + gate ------------------------------------------------
    results = {}
    rng = np.random.default_rng(SEED)
    order = rng.permutation(N_SCENES)
    n_train_sc = 34
    train_scenes = set(order[:n_train_sc].tolist())
    # frame rows are appended scene-by-scene in order; recover the split by
    # scene index blocks (N_FRAMES rows per scene, in scene_defs order)
    scene_ids = np.repeat(np.arange(N_SCENES), N_FRAMES)
    speed_row = np.repeat([sp for (th, sp, rep) in scene_defs], N_FRAMES)
    is_train = np.isin(scene_ids, list(train_scenes))
    is_test = ~is_train
    test_pos = np.where(is_test)[0]
    # scene-level validation carve-out from TRAIN (c1 house lam selection)
    rng2 = np.random.default_rng(SEED + 1)
    val_scenes = set(rng2.permutation(sorted(train_scenes))[:6].tolist())
    is_val = is_train & np.isin(scene_ids, list(val_scenes))
    is_fit = is_train & ~is_val

    for d in DENSITIES:
        pd = per_density[d]
        n = len(pd["X_pool"])
        Y = np.stack(pd["Y"])
        splits, lams_used = {}, {}
        for name, Xlist in (("primary_4code_stream", np.stack(pd["X_pool"])),
                            ("diagnostic_full_motion_field", np.stack(pd["X_field"])),
                            ("reference_lum_pool", np.stack(pd["X_lum"]))):
            X = np.stack(Xlist)
            pred, lam = pick_lam_and_predict(X, Y, is_fit[:n], is_val[:n], test_pos)
            splits[name] = dict(zip(TARGETS, r2_per_col(Y[test_pos], pred)))
            lams_used[name] = lam
        pr = splits["primary_4code_stream"]
        dir_r2 = round(float(np.mean([pr["sin_theta"], pr["cos_theta"]])) , 4)
        vel_r2 = round(float(np.mean([pr["vx"], pr["vy"]])), 4)
        # per-speed direction R2 (primary reader, test rows grouped by speed)
        Xp = np.stack(pd["X_pool"])
        pred_full, _ = pick_lam_and_predict(Xp, Y, is_fit[:n], is_val[:n], test_pos)
        per_speed = {}
        for sp in SPEEDS:
            m = is_test[:n] & (speed_row[:n] == sp)
            idx = np.where(m)[0]
            pos_in_test = np.searchsorted(test_pos, idx)
            # recompute r2 on this subgroup directly from the same predictions
            sub_pred = pred_full[pos_in_test]
            sub_r2 = r2_per_col(Y[idx], sub_pred)
            per_speed[str(sp)] = round(float(np.mean([sub_r2[0], sub_r2[1]])), 4)
        results[str(d)] = {
            "primary_direction_mean_r2": dir_r2,
            "primary_velocity_mean_r2": vel_r2,
            "primary_r2_per_target": pr,
            "primary_direction_r2_by_speed": per_speed,
            "diagnostic_r2_per_target": splits["diagnostic_full_motion_field"],
            "reference_lum_r2_per_target": splits["reference_lum_pool"],
            "lam_chosen": lams_used,
            "frames_total": n,
            "frames_train": int(is_train[:n].sum()),
            "frames_val": int(is_val[:n].sum()),
            "frames_test": int(is_test[:n].sum()),
            "motion_field_dim": int(np.stack(pd["X_field"]).shape[1]),
            "encode_ms_per_frame": round(pd["enc_ms"] / pd["t_frames"], 2),
            "cells": [d, int(d / ASPECT)],
        }
        print(f"[ie1] density {d}: direction R2 {dir_r2} | velocity R2 {vel_r2} "
              f"| per-speed {per_speed} | lam {lams_used}", flush=True)

    dir_all = [results[str(d)]["primary_direction_mean_r2"] for d in DENSITIES]
    best, npass = max(dir_all), sum(1 for v in dir_all if v >= 0.5)
    verdict = ("KEEP" if best >= 0.5 and npass >= 2 else
               "KILL" if best < 0.5 else "INCONCLUSIVE")

    out = {
        "experiment": "ie1 temporal fam codes — Reichardt correlators on the glyph lattice (the lobula-plate build)",
        "verdict": verdict,
        "gate": {"metric": "primary reader: mean R2 over direction targets (sin theta, cos theta)",
                 "threshold": ">= 0.5 at >= 2 c1-stock densities (c1 house gate; density set 24/48/96 per t1 precedent)",
                 "direction_mean_r2_by_density": dict(zip(map(str, DENSITIES), dir_all)),
                 "best": best, "n_densities_pass": npass},
        "question": "can a text-only ridge reader recover the true drift velocity of a scene from the Reichardt motion-code stream (4 pooled fam codes/frame) alone?",
        "scene": ("dark field + static radial key light + static 8x8 checker (distractors) "
                  "+ 2 bright gaussian blobs (sigma 14-22 px, I 0.65-1.0) drifting with a GLOBAL velocity, periodic wrap"),
        "data": {"n_scenes": N_SCENES, "velocities": "8 directions x 3 speeds (0.10/0.25/0.45 frame-widths/s) x 2 reps",
                 "fps": FPS, "seconds_per_scene": SECONDS,
                 "frames_per_scene": N_FRAMES, "warmup_frames": N_WARM,
                 "split": f"scene-level {n_train_sc}/{N_SCENES - n_train_sc}, seed {SEED} (no frame leakage: all frames of a scene stay on one side)",
                 "note": "frames within a scene share the target; the reader generalizes across scenes"},
        "correlator": {"baseline_cells": 1, "delay_frames": 1,
                       "delay_ms": round(1000.0 / FPS, 1),
                       "formula": "Cx = S(x,y,t)*S(x-1,y,t-1) - S(x,y,t)*S(x+1,y,t-1); Cy likewise; pool = half-wave-rectified lattice sums -> mx+, mx-, my+, my-",
                       "signal_S": "V=32 glyph code field (tone 0..10, edge 11+; t1 codes_for_temp convention)"},
        "readers": {"primary": "ridge on the 4 pooled motion codes per frame (the pre-registered motion-code stream)",
                    "diagnostic": "ridge on the full per-cell rectified motion field (4 x rows x cols)",
                    "reference": "ridge on the same 4-code pooling applied to the pre-glyph lum field",
                    "hygiene": "t1 ridge (drop near-dead cols, center, global scale) solved in the dual; lam in [10,100,1000] picked per reader on a 6-scene val carve-out from train (c1 house convention)"},
        "sanity_rightward_fast": sanity,
        "results": results,
        "settings_note": "glyph dials = t1 primary pin (classic ramp, tone 100, edge 60, thresh 4, gain 10, relief 70)",
        "wall_clock_s": round(time.time() - t_start, 1),
        "measured": ("direction = drift angle theta = atan2(vy, vx) of the render, encoded (sin, cos); "
                     "velocity = (vx, vy) in px/s; BOTH are per-scene constants recovered frame-by-frame "
                     "by the readers; R2 computed on held-out scenes only"),
    }
    print("===IE1_JSON===")
    print(json.dumps(out, indent=2))
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/ie1_motion.json", "w"), indent=2)
    print("[ie1] saved results/ie1_motion.json")

    # -------- sample artifact: 3 consecutive text frames + codes ------------
    sc = DriftScene(0.0, 0.45, 0)
    lines = ["ie1 sample — scene theta=0deg speed=0.45 fw/s, density 48, frames 20-22 "
             "+ pooled motion fam codes", "=" * 78]
    prev = None
    for t in range(23):
        codes, _ = codes_and_lum(sc.frame(t), s48)
        if prev is not None and 20 <= t <= 22:
            p = pool4(*reichardt_fields(codes, prev))
            lines.append(f"\n--- frame {t}: mx+ {p[0]:.0f}  mx- {p[1]:.0f}  "
                         f"my+ {p[2]:.0f}  my- {p[3]:.0f} ---")
            ramp = active_ramp(s48)
            for row in codes.astype(int):
                chars = []
                for c in row:
                    if c == 0:
                        chars.append(" ")
                    elif c <= 10:
                        chars.append(ramp[c])
                    else:
                        chars.append("·×▚▓╱╲"[min(6, (c - 11) // 5)])
                lines.append("".join(chars))
        prev = codes
    open("results/ie1_motion_sample.txt", "w").write("\n".join(lines) + "\n")
    print("[ie1] saved results/ie1_motion_sample.txt")


if __name__ == "__main__":
    main()
