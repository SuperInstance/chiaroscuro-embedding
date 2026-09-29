#!/usr/bin/env python3
"""thermal_glyph.py — t1: do thermal glyph fields carry the scene?

The REAL t1 (the two lane attempts phantomed — no artifacts). Built in-house,
main session, receipts guaranteed by construction. Per SEED.md t1:

- Scene: K=2 heat sources with gaussian profiles moving on smooth trajectories
  over a cold field; parameters (x1,y1,x2,y2,I1,I2) per frame.
- Render: temperature field -> grayscale RGB frame -> the chiaroscuro glyph
  pipeline (renderer.py's own functions, primary-pin dials) -> V=32 codes.
- Question: can a ridge reader recover the scene parameters from the codes?
- Sweep: densities {24, 48, 96} columns; report R^2 + chars/frame + encode ms.
- Gate (pre-registered): KEEP iff position R^2 >= 0.90 at some resolution AND
  the knee exists (>= 0.90 at >= 2 resolutions). KILL iff nothing beats 0.5.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/eileen/projects/chiaroscuro-embedding")
from renderer import (Settings, active_ramp, downsample, tone_field,
                      gradient_fields)

W, H = 320, 240
RL = 11  # classic ramp length
SEP = 31
N_TRAIN, N_TEST = 600, 200
SEED = 42


def scene_params(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(2):
        fx, fy = rng.uniform(4.0, 12.0, 2)   # MANY cycles per window: a slow
        # trajectory makes the 600/200 time-split dial-RANGE EXTRAPOLATION (the
        # X6 lesson) — test params sit in a narrow band, SS_tot ~ 0, R2 blows up
        # negative regardless of the features. 4-12 cycles/segment keeps both
        # train and test covering the full position range (interpolation).
        px, py = rng.uniform(0, 2 * np.pi, 2)
        for i in range(n):
            t = 2 * np.pi * i / n
            x = 0.5 + 0.33 * np.sin(fx * t + px)
            y = 0.5 + 0.30 * np.sin(fy * t + py)
            I = 0.6 + 0.4 * (0.5 + 0.5 * np.sin(fx * t * 0.7 + py + 1.3))
            out.append((k, x, y, I))
    # build the param matrix: (n, 6) = x1,y1,I1,x2,y2,I2
    P = np.zeros((n, 6))
    for k in range(2):
        rows = [r for r in out if r[0] == k]
        P[:, 3 * k + 0] = [r[1] for r in rows]
        P[:, 3 * k + 1] = [r[2] for r in rows]
        P[:, 3 * k + 2] = [r[3] for r in rows]
    return P


def thermal_frame(P_row):
    """Render the temperature field (H, W) for one param row."""
    yy, xx = np.mgrid[0:H, 0:W]
    T = np.zeros((H, W), dtype=np.float32)
    for k in range(2):
        x, y, I = P_row[3 * k: 3 * k + 3]
        cx, cy = x * W, y * H
        sigma = 30.0
        T += I * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2)
                          / (2 * sigma ** 2)))
    return np.clip(T, 0, 1)


def codes_for_temp(T, s):
    """Temperature field -> V=32 glyph codes (renderer's exact branching)."""
    frame = (np.repeat(T[:, :, None], 3, axis=2) * 255).astype(np.uint8)
    cols, rows = int(s.density), s.rows_for()
    small = downsample(frame.astype(np.float32), cols, rows)
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
    level = np.clip(
        np.floor(mag * 9.0 * (s.edge_gain / 8.0) * (0.5 + relief)), 0,
        4).astype(np.int32)
    t = np.clip(lum / 255.0, 0.0, 0.999999)
    tone_idx = np.clip(np.floor((1.0 - t) * (RL - 1)).astype(np.int32), 0,
                       RL - 1)
    tone_mask = (s.tone_amt > 5.0) & (~edge_mask)
    tone_mask &= (ramp_arr[tone_idx] != " ")
    codes = np.zeros((h, w), dtype=np.uint8)
    codes[tone_mask] = tone_idx[tone_mask]
    codes[edge_mask] = (11 + (fam * 5 + level))[edge_mask]
    return codes


def ridge_fit_predict(Xtr, Ytr, Xte, lam=10.0):
    # robust scaling (v3): the thermal field is mostly-zero background — per-column
    # standardization explodes on near-dead columns (train-std ~1e-3, test leverage
    # huge -> R2 -3114 even on RAW features). Fix: drop columns with train-std
    # < 2% of the global std, center per column, scale GLOBALLY, strong ridge.
    sd_col = Xtr.std(0)
    gstd = Xtr.std() + 1e-9
    keep = sd_col > 0.02 * gstd
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    mu_x = Xtr.mean(0)
    mu_y, sd_y = Ytr.mean(0), Ytr.std(0) + 1e-9
    A = ((Xtr - mu_x) / gstd).astype(np.float64)
    B = ((Ytr - mu_y) / sd_y).astype(np.float64)
    d = A.shape[1]
    w = np.linalg.solve(A.T @ A + lam * np.eye(d), A.T @ B)
    pred = (((Xte - mu_x) / gstd).astype(np.float64) @ w) * sd_y + mu_y
    return pred


def r2_per_col(Y, pred):
    y = np.asarray(Y, float)
    p = np.asarray(pred, float)
    out = []
    for j in range(y.shape[1]):
        ss = ((y[:, j] - p[:, j]) ** 2).sum()
        st = ((y[:, j] - y[:, j].mean()) ** 2).sum() + 1e-12
        out.append(round(float(1 - ss / st), 4))
    return out


def main():
    P = scene_params(N_TRAIN + N_TEST, SEED)
    results = {}
    for density in (24, 48, 96):
        s = Settings(density=density, ramp="classic", tone_amt=100.0,
                     edge_amt=60.0, edge_thresh=4.0, edge_gain=10.0,
                     relief=70.0)
        feats, t0 = [], time.time()
        for i in range(N_TRAIN + N_TEST):
            T = thermal_frame(P[i])
            c = codes_for_temp(T, s)
            feats.append(c.reshape(-1).astype(np.float32))
        enc_ms = (time.time() - t0) / (N_TRAIN + N_TEST) * 1000
        X = np.stack(feats)
        Xtr, Xte = X[:N_TRAIN], X[N_TRAIN:]
        Ytr, Yte = P[:N_TRAIN], P[N_TRAIN:]
        pred = ridge_fit_predict(Xtr, Ytr, Xte)
        r2 = r2_per_col(Yte, pred)
        pos_r2 = [r2[0], r2[1], r2[3], r2[4]]   # x1,y1,x2,y2
        results[str(density)] = {
            "r2_per_param": dict(zip(["x1", "y1", "I1", "x2", "y2", "I2"],
                                     r2)),
            "positions_mean_r2": round(float(np.mean(pos_r2)), 4),
            "intensity_mean_r2": round(float(np.mean([r2[2], r2[5]])), 4),
            "chars_per_frame": int(X.shape[1]),
            "encode_ms": round(enc_ms, 2),
        }
        print(f"[t1] density {density}: pos R2 {results[str(density)]['positions_mean_r2']}"
              f" int R2 {results[str(density)]['intensity_mean_r2']}"
              f" {X.shape[1]} cells {enc_ms:.1f}ms/frame", flush=True)

    pos_all = [v["positions_mean_r2"] for v in results.values()]
    knee = sum(1 for p in pos_all if p >= 0.90)
    verdict = ("KEEP" if max(pos_all) >= 0.90 and knee >= 2 else
               "KILL" if max(pos_all) < 0.5 else "INCONCLUSIVE")
    out = {"experiment": "t1 thermal glyph fields",
           "verdict": verdict,
           "gate": {"positions_best": max(pos_all), "n_resolutions_ge_090": knee},
           "results": results,
           "n_train": N_TRAIN, "n_test": N_TEST, "seed": SEED,
           "note": "built in-house after two phantom lane reports; artifacts on disk by construction"}
    print("===T1_JSON===")
    print(json.dumps(out, indent=2))
    import os
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/t1_thermal.json", "w"), indent=2)
    print("[t1] saved results/t1_thermal.json")


if __name__ == "__main__":
    main()
