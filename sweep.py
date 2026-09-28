"""
sweep.py — experiment c1: renderer sweep vs synthetic ground truth.

Question: when chiaroscuro crushes a frame into a grid of CHARACTERS, which
render settings keep enough information for a *text-only* reader to recover
the scene? We render a synthetic mock scene (moving shaded sphere + rotating
bars + static checkerboard, mirroring chiaroscuro's ?mock=1) at ~34 settings,
and train two readers on nothing but the rendered text:

  ridge — exact dual kernel ridge on native text features (per-cell tone
          index / edge level / edge direction channels derived from the
          characters), alpha picked on a val split
  mlp   — PCA-100 -> small sklearn MLPRegressor

Ground truth is known by construction (scene.py). Chance floor is measured
empirically by permuting labels on the best setting.

Verdict (pre-registered gate from SEED.md):
  KEEP iff >= 1 setting recovers >= 2 of 3 scene params at test R^2 >= 0.5
  (chance ~ 0). Otherwise KILL. Either way the landscape is reported.

Usage:
  python sweep.py                 # full sweep, ONE JSON verdict on stdout
  C1_SMOKE=1 python sweep.py      # 2-min smoke (fewer frames/settings)
Progress goes to stderr; stdout is a single JSON document.
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
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from renderer import RAMPS, Settings, render
from scene import MockScene, TARGET_NAMES

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SMOKE = bool(os.environ.get("C1_SMOKE"))
N_TRAIN, N_VAL, N_TEST = (60, 12, 24) if SMOKE else (280, 40, 80)
R2_GATE = 0.5
PARAMS_NEEDED = 2
TOP_K = 8
ALPHAS = (10.0, 100.0, 1000.0)
PCA_DIMS = 100


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# sweep grid: ~34 settings over ramp / edge threshold / density / edge mix
# ---------------------------------------------------------------------------
def build_grid() -> List[Tuple[str, Settings]]:
    g: List[Tuple[str, Settings]] = []

    def mix(**kw) -> Settings:
        d = dict(density=100, aspect=4 / 3, ramp="classic", tone_amt=100,
                 edge_amt=60, edge_thresh=4, edge_gain=10, relief=70,
                 contrast=25, gamma=100)
        d.update(kw)
        return Settings(**d)

    # 1) ramp landscape, pure tone (the classic ASCII look)
    for ramp in RAMPS:
        g.append((f"tone|ramp={ramp}", mix(edge_amt=0, ramp=ramp)))
    # 2) edge-only: threshold sweep (glyph = slope family, relief levels)
    for thr in (2, 4, 8):
        g.append((f"edge|thr={thr}", mix(tone_amt=0, edge_amt=100, edge_thresh=thr)))
    # 3) edge-only: detector family
    g.append(("edge|style=emboss", mix(tone_amt=0, edge_amt=100, edge_style="emboss")))
    g.append(("edge|style=laplace", mix(tone_amt=0, edge_amt=100, edge_style="laplace")))
    # 4) mixed tone+edge across ramps (bivariate glyph engine, the v2 core)
    for ramp in ("classic", "fine", "blocks", "minimal", "binary"):
        g.append((f"mix|ramp={ramp}", mix(ramp=ramp)))
    # 5) density sweep (1 cell = 1 char; cols=100 is mix|ramp=classic)
    for dcol in (40, 60, 80, 130):
        g.append((f"mix|density={dcol}", mix(density=dcol)))
    # 6) edge mix sweep (tone kept at 100)
    for ea in (10, 25, 50):
        g.append((f"mix|edgeAmt={ea}", mix(edge_amt=ea)))
    # 7) shipped studio presets (their sculpt dials)
    g.append(("preset|sculptor", mix(ramp="fine", edge_amt=100, edge_gain=8, relief=55)))
    g.append(("preset|blueprint", mix(ramp="minimal", tone_amt=25, edge_amt=100,
                                      edge_thresh=3, edge_gain=14, relief=70)))
    g.append(("preset|wireframe", mix(ramp="minimal", tone_amt=0, edge_amt=100,
                                      edge_thresh=5, edge_gain=15, relief=80)))
    g.append(("preset|green-matrix", mix(ramp="classic", edge_amt=20, edge_gain=10, relief=25)))
    # 8) image dials
    g.append(("mix|contrast=10", mix(contrast=10)))
    g.append(("mix|contrast=60", mix(contrast=60)))
    # 9) tone quantization
    g.append(("tone|posterize=3", mix(edge_amt=0, posterize=3)))
    g.append(("tone|posterize=6", mix(edge_amt=0, posterize=6)))
    return g


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------
def featurize_all(scene: MockScene, s: Settings, frames: List[np.ndarray]) -> np.ndarray:
    """Render every frame at this setting -> (n, 4*rows*cols) float32 text features.

    Features are a lossless numeric reading of the CHARACTER GRID: per cell
    (tone index normalized by ramp length, edge level, edge dir x, edge dir y)
    — exactly the channels a text-only model could decode from the glyphs.
    """
    feats = None
    for i, fr in enumerate(frames):
        r = render(fr, s)
        f = r.channels.reshape(-1).astype(np.float32)
        if feats is None:
            feats = np.empty((len(frames), f.size), dtype=np.float32)
        feats[i] = f
    return feats


def standardize(Xtr: np.ndarray, *rest: np.ndarray) -> Tuple[np.ndarray, ...]:
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return tuple((X - mu) / sd for X in (Xtr,) + rest)


def dual_ridge(Xtr: np.ndarray, ytr: np.ndarray, Xval: np.ndarray, yval: np.ndarray,
               Xte: np.ndarray, yte: np.ndarray) -> Tuple[np.ndarray, float, object]:
    """Exact ridge via the dual: solve (K + aI) c = y with K = Xtr Xtr^T.

    One eigendecomposition serves every alpha and all label permutations,
    so the permutation null below is nearly free.
    Returns (test predictions at best alpha, best alpha, eig cache).
    """
    K = Xtr @ Xtr.T
    w, V = np.linalg.eigh(K.astype(np.float64))
    Vy = V.T @ ytr                            # (n, n_targets)
    Kval = Xval @ Xtr.T
    Kte = Xte @ Xtr.T

    def preds(Kx: np.ndarray, alpha: float) -> np.ndarray:
        coef = V @ (Vy / (w + alpha)[:, None])    # (n, n_targets)
        return Kx @ coef

    best_a, best_r2 = ALPHAS[0], -np.inf
    for a in ALPHAS:
        r2 = r2_score(yval, preds(Kval, a).astype(np.float32), multioutput="uniform_average")
        if r2 > best_r2:
            best_a, best_r2 = a, r2
    return preds(Kte, best_a).astype(np.float32), best_a, (w, V, Kte)


def dual_ridge_from_cache(ytr_perm: np.ndarray, cache) -> np.ndarray:
    w, V, Kte = cache
    Vy = V.T @ ytr_perm
    coef = V @ (Vy / (w + ALPHAS[0])[:, None])
    return (Kte @ coef).astype(np.float32)


def mlp_reader(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int = 0
               ) -> Tuple[np.ndarray, PCA]:
    n_comp = min(PCA_DIMS, Xtr.shape[0] - 1, Xtr.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed).fit(Xtr)
    Ptr = pca.transform(Xtr)
    Pte = pca.transform(Xte)
    sc = StandardScaler().fit(Ptr)
    net = MLPRegressor(hidden_layer_sizes=(64,), activation="relu", alpha=1e-2,
                       max_iter=200, random_state=seed, tol=1e-3,
                       learning_rate_init=1e-3).fit(sc.transform(Ptr), ytr)
    return net.predict(sc.transform(Pte)).astype(np.float32), pca


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def per_param_r2(pred: np.ndarray, yte: np.ndarray) -> Dict[str, float]:
    """R^2 per target; the circular angle is min(R2(sin2a), R2(cos2a))."""
    out = {}
    for i, name in enumerate(TARGET_NAMES):
        out[name] = float(r2_score(yte[:, i], pred[:, i]))
    out["bar_angle"] = min(out["bar_angle_sin2"], out["bar_angle_cos2"])
    return out


def main() -> None:
    t0 = time.time()
    scene = MockScene()

    log(f"[c1] generating frames: {N_TRAIN}/{N_VAL}/{N_TEST} (train/val/test)")
    splits = []
    for name, n, seed in (("train", N_TRAIN, 101), ("val", N_VAL, 202), ("test", N_TEST, 303)):
        frames, y, params = scene.sequence(n, seed=seed)
        splits.append({"name": name, "frames": frames, "y": y})
        log(f"  {name}: {n} frames, y{y.shape}")
    ytr, yval, yte = splits[0]["y"], splits[1]["y"], splits[2]["y"]
    # z-score targets with train stats (readers regress in z space)
    ymu, ysd = ytr.mean(0), ytr.std(0)
    ytr_z = (ytr - ymu) / ysd
    yval_z = (yval - ymu) / ysd
    yte_z = (yte - ymu) / ysd

    grid = build_grid()
    if SMOKE:
        grid = [g for i, g in enumerate(grid) if i % 4 == 0][:8]
    log(f"[c1] sweeping {len(grid)} settings ...")

    results = []
    for k, (name, st) in enumerate(grid):
        tset = time.time()
        Xtr = featurize_all(scene, st, splits[0]["frames"])
        Xval = featurize_all(scene, st, splits[1]["frames"])
        Xte = featurize_all(scene, st, splits[2]["frames"])
        Xtr, Xval, Xte = standardize(Xtr, Xval, Xte)

        # reader A: dual ridge on native text features
        pred_r, alpha, _ = dual_ridge(Xtr, ytr_z, Xval, yval_z, Xte, yte_z)
        # un-zscore predictions, score on the true scale
        r_ridge = per_param_r2(pred_r * ysd + ymu, yte)

        # reader B: PCA -> small MLP
        pred_m, pca = mlp_reader(Xtr, ytr_z, Xte)
        r_mlp = per_param_r2(pred_m * ysd + ymu, yte)

        best = {p: max(r_ridge[p], r_mlp[p]) for p in ("sphere_x", "sphere_y", "bar_angle")}
        n_rec = sum(v >= R2_GATE for v in best.values())
        entry = {
            "name": name,
            "dials": {k2: st.as_dict()[k2] for k2 in
                      ("density", "ramp", "tone_amt", "edge_amt", "edge_thresh",
                       "edge_gain", "edge_style", "relief", "contrast", "gamma", "posterize")},
            "n_text_features": int(Xtr.shape[1]),
            "ridge_alpha": float(alpha),
            "ridge_r2": r_ridge,
            "mlp_r2": r_mlp,
            "best_param_r2": best,
            "n_params_recovered_ge_0.5": int(n_rec),
            "mean_best_r2": float(np.mean(list(best.values()))),
        }
        results.append(entry)
        del Xtr, Xval, Xte, pred_r, pred_m, pca
        log(f"  [{k + 1:2d}/{len(grid)}] {name:26s} ridge(a={alpha:g}) sx={r_ridge['sphere_x']:+.2f} "
            f"sy={r_ridge['sphere_y']:+.2f} ang={r_ridge['bar_angle']:+.2f} | "
            f"mlp sx={r_mlp['sphere_x']:+.2f} sy={r_mlp['sphere_y']:+.2f} "
            f"ang={r_mlp['bar_angle']:+.2f} | recovered={n_rec}/3 ({time.time() - tset:.1f}s)")

    # ---- empirical chance floor: permute labels on the best setting -------
    log("[c1] permutation null (20 shuffles) on the top setting ...")
    top = max(results, key=lambda e: (e["n_params_recovered_ge_0.5"], e["mean_best_r2"]))
    st = next(s for n, s in grid if n == top["name"])
    Xtr = standardize(featurize_all(scene, st, splits[0]["frames"]))[0]
    Xval = standardize(featurize_all(scene, st, splits[1]["frames"]))[0]
    Xte = standardize(featurize_all(scene, st, splits[2]["frames"]))[0]
    _, alpha, cache = dual_ridge(Xtr, ytr_z, Xval, yval_z, Xte, yte_z)
    rng = np.random.default_rng(7)
    perm_scores: List[Dict[str, float]] = []
    for _ in range(20):
        idx = rng.permutation(len(ytr_z))
        pred = dual_ridge_from_cache(ytr_z[idx], cache)
        perm_scores.append(per_param_r2(pred * ysd + ymu, yte))
    chance = {p: float(np.mean([s[p] for s in perm_scores]))
              for p in ("sphere_x", "sphere_y", "bar_angle")}
    chance["bar_angle_sin2"] = float(np.mean([s["bar_angle_sin2"] for s in perm_scores]))
    chance["bar_angle_cos2"] = float(np.mean([s["bar_angle_cos2"] for s in perm_scores]))

    # ---- verdict -----------------------------------------------------------
    results.sort(key=lambda e: (e["n_params_recovered_ge_0.5"], e["mean_best_r2"]), reverse=True)
    best_rec = results[0]["n_params_recovered_ge_0.5"]
    verdict = "KEEP" if best_rec >= PARAMS_NEEDED else "KILL"

    out = {
        "experiment": "c1-renderer-sweep",
        "repo": "chiaroscuro-embedding",
        "question": "which chiaroscuro render settings leave enough scene information in the TEXT for a text-only reader to recover?",
        "scene": "mock-mirror (static key light, shaded sphere, 2 rotating bars, static checker; light/checker frozen to avoid confounds)",
        "data": {"n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
                 "sampling": "scene params i.i.d. uniform per frame, disjoint seeds",
                 "targets": list(TARGET_NAMES)},
        "readers": {
            "ridge": "exact dual kernel ridge on native text features, alpha in [10,100,1000] picked on val",
            "mlp": "PCA-100 -> StandardScaler -> sklearn MLPRegressor(64)",
            "text_features": "per cell: tone-ramp index (normalized), edge level, edge dir x/y — a lossless read of the glyph grid",
        },
        "scoring": {"per_param": "max(ridge, mlp) test R^2",
                    "bar_angle": "min(R2(sin2a), R2(cos2a)) — both circular components required",
                    "gate": f"KEEP iff some setting recovers >= {PARAMS_NEEDED}/3 params at test R^2 >= {R2_GATE} (chance ~ 0)"},
        "chance_empirical": {"method": "ridge with permuted train labels, 20 shuffles, best setting",
                             "mean_test_r2": chance},
        "verdict": verdict,
        "verdict_detail": f"best setting '{results[0]['name']}' recovers "
                          f"{best_rec}/3 params at R^2 >= {R2_GATE}",
        "n_settings": len(results),
        "top_k": results[:TOP_K],
        "settings": results,
        "runtime_s": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=2))  # THE verdict — stdout stays pure JSON
    log(f"[c1] done in {out['runtime_s']}s -> verdict {verdict}")


if __name__ == "__main__":
    main()
