"""
thermal_glyph_iid.py — experiment t1-iid: thermal glyph fields, i.i.d. frames.

Companion to thermal_glyph.py (the concurrent re-run of the thermal lane,
whose temporal-split/3-unordered-sources run returned KILL). This script
isolates WHY: same modality, but a controlled design —

  * 2 heat sources (not 3), RELABELED by x-order every frame: the scene ->
    target map is deterministic, so identity can never swap under the reader.
    (Crossing sources with arbitrary labels make targets independent of the
    image — unlearnable by any reader, which is a LABELING artifact, not a
    modality verdict. A 2-source version of that failure reproduced this
    script's only broken smoke run: all R^2 in [-0.5, -0.2], the same
    signature as the companion KILL.)
  * i.i.d. uniform frames (c1 house style) + disjoint seeds — no temporal
    split, no trajectory manifold shift between train and test.
  * ordinal text features (tone index / edge level / edge dir), not one-hot:
    the ramp index is ordinal and ridge exploits monotone structure.
  * readers = dual ridge AND PCA-MLP (c1 style); empirical permutation floor.

Casey's directive (2026-09-28): "we can do this same 2ASCII for more than just
visible light... thermal embedding at any resolution for quick encoding which
could revolutionize reflexes and motion of creatures ML."

Scene:
  cold ambient 12 C (+ static ground gradient + 0.8 C sensor noise),
  2 gaussian heat sources, sigma 9 px FIXED (else intensity<->size confound),
  amplitude A = 18 + 62*i C over ambient (peaks 30..92 C),
  diffusion blur sigma 2.5 px, sources x-ordered (src1 = leftmost).
  Targets per frame: (src1_x, src1_y, src1_i, src2_x, src2_y, src2_i).

Canonical density V=32 (32x24 = 768 chars/frame, G1 reference; aspect 4/3 —
the classic thermal-sensor frame shape, e.g. a FLIR Lepton thumbnail).
Resolution sweep: {24, 32, 48, 96, 192} cols — chars/frame + encode ms/frame.

GATE (pre-registered):
  KILL   iff no setting recovers any parameter at test R^2 > 0.5
  KEEP   iff some setting recovers ALL parameters at test R^2 >= 0.90 AND the
         resolution knee exists (all-params >= 0.90 holds at >= 2 densities)
  else   MARGINAL

Usage:
  python thermal_glyph_iid.py                 # ONE JSON verdict on stdout
  T1_SMOKE=1 python thermal_glyph_iid.py      # ~2-min smoke

Progress goes to stderr; stdout is a single JSON document.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from renderer import Settings, render

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SMOKE = bool(os.environ.get("T1_SMOKE"))
N_TRAIN, N_VAL, N_TEST = (60, 12, 24) if SMOKE else (280, 40, 80)
DENSITIES_SWEEP = (24, 32, 48) if SMOKE else (24, 32, 48, 96, 192)
N_SHUFFLES = 5 if SMOKE else 20
BASE_DENSITY = 32            # V=32, the G1 reference
KEEP_BAR = 0.90
KILL_BAR = 0.50
ALPHAS = (10.0, 100.0, 1000.0)
PCA_DIMS = 100
PARAM_NAMES = ("src1_x", "src1_y", "src1_i", "src2_x", "src2_y", "src2_i")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# the thermal scene: temperature field -> gray RGB for the glyph pipeline
# ---------------------------------------------------------------------------
class ThermalScene:
    """Two moving heat sources (gaussians) diffusing over a cold field.

    sigma 20 px (scene v2): a source spans ~7 cells at V=32 — the reflex-range
    scale (creature/engine at close range; c1's sphere template was ~9 cells).
    Scene v1 used sigma 9 px (~2 cells) and a RAW-FIELD CONTROL showed even the
    un-glyphed field capped at R^2 ~ 0.4 for the linear reader class — the
    template, not the glyphs, was the bottleneck. v2 also kills the sub-cell
    phase x intensity coupling (cell average ~= peak amplitude at this sigma).

    sigma is FIXED across sources/frames so intensity maps 1:1 to peak
    amplitude. Sources are relabeled by x-order EVERY frame: src1 is always
    the leftmost — the identity convention that makes the target map
    deterministic (the reflex layer will need exactly such a convention;
    a tracking prior supplies it in the wild).
    """

    T_LO, T_HI = 2.0, 95.0      # gray mapping window, degrees C

    def __init__(self, width: int = 320, height: int = 240,
                 ambient: float = 12.0, sigma: float = 20.0,
                 blur_sigma: float = 4.0, noise_c: float = 0.8):
        self.w, self.h = int(width), int(height)
        self.ambient = ambient
        self.sigma = sigma
        self.blur_sigma = blur_sigma
        self.noise_c = noise_c
        Y, X = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.X, self.Y = X, Y
        # mild static ground gradient (edges colder), so a source-free frame
        # is not perfectly flat — real night fields have structure
        gx = np.abs(X / self.w - 0.5) * 2.0
        gy = np.abs(Y / self.h - 0.5) * 2.0
        self._cold = self.ambient - 4.0 * np.sqrt(gx ** 2 + gy ** 2)

    def field(self, x1: float, y1: float, i1: float,
              x2: float, y2: float, i2: float,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
        T = self._cold.copy()
        for x, y, i in ((x1, y1, i1), (x2, y2, i2)):
            amp = 18.0 + 62.0 * i
            d2 = (self.X - x * self.w) ** 2 + (self.Y - y * self.h) ** 2
            T += amp * np.exp(-d2 / (2.0 * self.sigma ** 2))
        T = cv2.GaussianBlur(T, (0, 0), self.blur_sigma)   # diffusion step
        if rng is not None and self.noise_c > 0:
            T = T + rng.normal(0.0, self.noise_c, T.shape).astype(np.float32)
        return T

    def to_rgb(self, T: np.ndarray) -> np.ndarray:
        g = np.clip((T - self.T_LO) / (self.T_HI - self.T_LO), 0.0, 1.0) * 255.0
        return np.repeat(g[:, :, None], 3, axis=2).astype(np.uint8)

    def sample_params(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # two independent draws, then ORDERED BY X: identity is deterministic
        # (src1 = leftmost), so the scene -> target map can never swap.
        xa, xb = rng.uniform(0.12, 0.88, n), rng.uniform(0.12, 0.88, n)
        ya, yb = rng.uniform(0.12, 0.88, n), rng.uniform(0.12, 0.88, n)
        ia, ib = rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 1.0, n)
        swap = xa > xb
        xa, xb = np.where(swap, xb, xa), np.where(swap, xa, xb)
        ya, yb = np.where(swap, yb, ya), np.where(swap, ya, yb)
        ia, ib = np.where(swap, ib, ia), np.where(swap, ia, ib)
        return np.stack([xa, ya, ia, xb, yb, ib], axis=1).astype(np.float32)

    def sequence(self, n: int, seed: int) -> Tuple[List[np.ndarray], np.ndarray]:
        rng = np.random.default_rng(seed)
        P = self.sample_params(n, rng)
        frames = [self.to_rgb(self.field(*P[k], rng=rng)) for k in range(n)]
        return frames, P


# ---------------------------------------------------------------------------
# readers (c1 house style, verbatim pattern from sweep.py)
# ---------------------------------------------------------------------------
def featurize_all(scene: ThermalScene, s: Settings, frames: List[np.ndarray],
                  view: str = "native") -> np.ndarray:
    """Render every frame -> (n, D) float32 text features.

    view="native": per cell (tone-ramp index normalized, edge level, edge
    dir x, edge dir y) — the c1 lossless read of the glyph grid.

    view="normass": the same grid re-read LOSSLESSLY as (cell / total ink
    mass, mass). Why: a raw-field control showed varying source intensity
    confounds LINEAR position readout (Sum x_i v_i ~ p*A; a linear reader
    cannot divide by mass), capping even un-glyphed fields at R^2 ~ 0.5.
    The normalized view removes exactly that confound while remaining a pure
    function of the glyph text — one sum + one divide, a reflex-scale op.
    Both views are reported everywhere; the gate uses the per-param max.
    """
    feats = None
    for k, fr in enumerate(frames):
        r = render(fr, s)
        f = r.channels.reshape(-1).astype(np.float32)
        if feats is None:
            feats = np.empty((len(frames), f.size), dtype=np.float32)
        feats[k] = f
    if view == "normass":
        mass = np.maximum(feats.sum(axis=1, keepdims=True), 1e-6)
        feats = np.concatenate([feats / mass, mass / 100.0], axis=1).astype(np.float32)
    return feats


def standardize(Xtr: np.ndarray, *rest: np.ndarray) -> Tuple[np.ndarray, ...]:
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return tuple((X - mu) / sd for X in (Xtr,) + rest)


def dual_ridge(Xtr, ytr, Xval, yval, Xte, yte):
    K = Xtr @ Xtr.T
    w, V = np.linalg.eigh(K.astype(np.float64))
    Vy = V.T @ ytr
    Kval = Xval @ Xtr.T
    Kte = Xte @ Xtr.T

    def preds(Kx, alpha):
        coef = V @ (Vy / (w + alpha)[:, None])
        return Kx @ coef

    best_a, best_r2 = ALPHAS[0], -np.inf
    for a in ALPHAS:
        r2 = r2_score(yval, preds(Kval, a).astype(np.float32),
                      multioutput="uniform_average")
        if r2 > best_r2:
            best_a, best_r2 = a, r2
    return preds(Kte, best_a).astype(np.float32), best_a, (w, V, Kte)


def dual_ridge_from_cache(ytr_perm, cache):
    w, V, Kte = cache
    Vy = V.T @ ytr_perm
    coef = V @ (Vy / (w + ALPHAS[0])[:, None])
    return (Kte @ coef).astype(np.float32)


def mlp_reader(Xtr, ytr, Xte, seed: int = 0):
    n_comp = min(PCA_DIMS, Xtr.shape[0] - 1, Xtr.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="randomized",
              random_state=seed).fit(Xtr)
    Ptr = pca.transform(Xtr)
    Pte = pca.transform(Xte)
    sc = StandardScaler().fit(Ptr)
    net = MLPRegressor(hidden_layer_sizes=(64,), activation="relu", alpha=1e-2,
                       max_iter=200, random_state=seed, tol=1e-3,
                       learning_rate_init=1e-3).fit(sc.transform(Ptr), ytr)
    return net.predict(sc.transform(Pte)).astype(np.float32)


def per_param_r2(pred: np.ndarray, yte: np.ndarray) -> Dict[str, float]:
    out = {n: float(r2_score(yte[:, k], pred[:, k]))
           for k, n in enumerate(PARAM_NAMES)}
    out["min_param"] = float(min(out.values()))
    out["mean_param"] = float(np.mean(list(out.values())[:len(PARAM_NAMES)]))
    return out


# ---------------------------------------------------------------------------
# dial landscape (thermal-adapted) + resolution sweep
# ---------------------------------------------------------------------------
def build_grid() -> List[Tuple[str, Settings]]:
    """9 dials at the canonical V=32. Thermal adaptation: low edge thresholds
    (gaussian heat blobs are smooth, so sobel magnitudes are small), high
    relief/gain. Polarity NOTE (learned the hard way): invert=True maps the
    cold ambient to tone index 0 (' ') which the tone mask BLANKS — only the
    blob cores survive (~57/768 cells). invert=False keeps every cell inked
    (hot = light dent in dense ink), so the tone dials default to it; one
    'hotink' variant is kept to quantify exactly that blanking cost."""

    def mix(**kw) -> Settings:
        d = dict(density=BASE_DENSITY, aspect=4 / 3, ramp="classic",
                 tone_amt=100, edge_amt=60, edge_thresh=2, edge_gain=12,
                 relief=70, contrast=25, gamma=100, invert=False)
        d.update(kw)
        return Settings(**d)

    g: List[Tuple[str, Settings]] = []
    for ramp in ("classic", "blocks", "waves", "minimal"):
        g.append((f"tone|ramp={ramp}", mix(edge_amt=0, ramp=ramp)))
    g.append(("mix|ramp=classic", mix(ramp="classic")))
    g.append(("mix|ramp=blocks", mix(ramp="blocks")))
    g.append(("tone|blocks|hotink", mix(edge_amt=0, ramp="blocks", invert=True)))
    g.append(("edge|thr=1", mix(tone_amt=0, edge_amt=100, edge_thresh=1,
                                edge_gain=14, relief=75)))
    g.append(("edge|emboss", mix(tone_amt=0, edge_amt=100, edge_thresh=1,
                                 edge_style="emboss", edge_gain=14, relief=75)))
    return g


def run_setting(scene: ThermalScene, st: Settings, splits, ymu, ysd,
                ytr_z, yval_z, yte_z, yte) -> Dict[str, object]:
    views: Dict[str, object] = {}
    per_view_best: Dict[str, Dict[str, float]] = {}
    for view in ("native", "normass"):
        Xtr, Xval, Xte = (featurize_all(scene, st, sp["frames"], view=view)
                          for sp in splits)
        Xtr, Xval, Xte = standardize(Xtr, Xval, Xte)
        pred_r, alpha, _ = dual_ridge(Xtr, ytr_z, Xval, yval_z, Xte, yte_z)
        r_ridge = per_param_r2(pred_r * ysd + ymu, yte)
        pred_m = mlp_reader(Xtr, ytr_z, Xte)
        r_mlp = per_param_r2(pred_m * ysd + ymu, yte)
        per_view_best[view] = {n: max(r_ridge[n], r_mlp[n]) for n in PARAM_NAMES}
        views[view] = {
            "ridge_alpha": float(alpha),
            "ridge_r2": {n: r_ridge[n] for n in PARAM_NAMES},
            "mlp_r2": {n: r_mlp[n] for n in PARAM_NAMES},
            "best_param_r2": per_view_best[view],
            "min_param_r2": float(min(per_view_best[view].values())),
            "mean_param_r2": float(np.mean(list(per_view_best[view].values()))),
        }
        del Xtr, Xval, Xte, pred_r, pred_m
    best = {n: max(per_view_best["native"][n], per_view_best["normass"][n])
            for n in PARAM_NAMES}
    return {
        "views": views,
        "best_param_r2": best,
        "min_param_r2": float(min(best.values())),
        "mean_param_r2": float(np.mean(list(best.values()))),
        "n_params_ge_090": int(sum(v >= KEEP_BAR for v in best.values())),
    }


def main() -> None:
    t0 = time.time()
    scene = ThermalScene()
    log(f"[t1-iid] thermal scene: 2 x-ordered heat sources, stage {scene.w}x{scene.h}, "
        f"canonical V={BASE_DENSITY} ({BASE_DENSITY}x{int(BASE_DENSITY / (4 / 3))} chars)")

    log(f"[t1-iid] generating frames: {N_TRAIN}/{N_VAL}/{N_TEST} (train/val/test)")
    splits = []
    for name, n, seed in (("train", N_TRAIN, 4101), ("val", N_VAL, 4202),
                          ("test", N_TEST, 4303)):
        frames, y = scene.sequence(n, seed=seed)
        splits.append({"name": name, "frames": frames, "y": y})
        log(f"  {name}: {n} frames, y{y.shape}")
    ytr, yval, yte = splits[0]["y"], splits[1]["y"], splits[2]["y"]
    ymu, ysd = ytr.mean(0), ytr.std(0)
    ytr_z, yval_z, yte_z = ((y - ymu) / ysd for y in (ytr, yval, yte))

    # ---- dial landscape at V=32 -------------------------------------------
    grid = build_grid()
    log(f"[t1-iid] dial landscape at V={BASE_DENSITY}: {len(grid)} settings x 2 feature views ...")
    landscape = []
    for k, (name, st) in enumerate(grid):
        ts = time.time()
        res = run_setting(scene, st, splits, ymu, ysd, ytr_z, yval_z, yte_z, yte)
        res["name"] = name
        res["dials"] = {k2: st.as_dict()[k2] for k2 in
                        ("density", "aspect", "ramp", "tone_amt", "edge_amt",
                         "edge_thresh", "edge_gain", "edge_style", "relief",
                         "contrast", "gamma", "invert")}
        res["n_text_features"] = 4 * st.rows_for() * st.density
        landscape.append(res)
        log(f"  [{k + 1:2d}/{len(grid)}] {name:20s} min={res['min_param_r2']:+.3f} "
            f"mean={res['mean_param_r2']:+.3f} ge090={res['n_params_ge_090']}/6 "
            f"[nat {res['views']['native']['mean_param_r2']:+.2f} / "
            f"nrm {res['views']['normass']['mean_param_r2']:+.2f}] "
            f"({time.time() - ts:.1f}s)")

    # ---- resolution sweep on the strongest dial ---------------------------
    top = max(landscape, key=lambda e: e["mean_param_r2"])
    st_top = next(s for n, s in grid if n == top["name"])
    log(f"[t1-iid] resolution sweep on '{top['name']}' over densities {DENSITIES_SWEEP} ...")
    sweep = []
    for dens in DENSITIES_SWEEP:
        ts = time.time()
        st = Settings(**{**st_top.as_dict(), "density": dens})
        rows = st.rows_for()
        # encode cost: wall time of the glyph encoder itself
        t_enc = time.perf_counter()
        for sp in splits:
            for fr in sp["frames"]:
                render(fr, st)
        enc_ms = (time.perf_counter() - t_enc) * 1000.0 / sum(len(s["frames"]) for s in splits)
        res = run_setting(scene, st, splits, ymu, ysd, ytr_z, yval_z, yte_z, yte)
        entry = {
            "density_cols": dens, "rows": rows,
            "chars_per_frame": int(rows * dens),
            "n_text_features": int(4 * rows * dens),
            "encode_ms_per_frame": round(enc_ms, 3),
            "all_params_ge_090": bool(res["min_param_r2"] >= KEEP_BAR),
            "views": res["views"],
            **{k2: res[k2] for k2 in ("best_param_r2", "min_param_r2",
                                      "mean_param_r2", "n_params_ge_090")},
        }
        sweep.append(entry)
        log(f"  d={dens:3d} ({rows}x{dens}={rows * dens} chars, {enc_ms:.2f} ms/frame) "
            f"min={entry['min_param_r2']:+.3f} mean={entry['mean_param_r2']:+.3f} "
            f"ge090={entry['n_params_ge_090']}/6 ({time.time() - ts:.1f}s)")

    # ---- knee: smallest density where ALL params hold >= 0.90 -------------
    ok = [e for e in sweep if e["all_params_ge_090"]]
    knee = None
    if len(ok) >= 2:
        k = min(ok, key=lambda e: e["density_cols"])
        knee = {"density_cols": k["density_cols"],
                "chars_per_frame": k["chars_per_frame"],
                "encode_ms_per_frame": k["encode_ms_per_frame"],
                "min_param_r2": k["min_param_r2"],
                "rationale": f"smallest density with all 6 params >= {KEEP_BAR}; "
                             f"holds at {len(ok)}/{len(sweep)} swept densities "
                             f"({[e['density_cols'] for e in ok]})"}

    # ---- empirical chance floor on the strongest setting ------------------
    top_view = max(("native", "normass"),
                   key=lambda v: top["views"][v]["mean_param_r2"])
    log(f"[t1-iid] permutation null ({N_SHUFFLES} shuffles) on '{top['name']}' "
        f"view={top_view} ...")
    Xtr = standardize(featurize_all(scene, st_top, splits[0]["frames"], view=top_view))[0]
    Xval = standardize(featurize_all(scene, st_top, splits[1]["frames"], view=top_view))[0]
    Xte = standardize(featurize_all(scene, st_top, splits[2]["frames"], view=top_view))[0]
    _, _, cache = dual_ridge(Xtr, ytr_z, Xval, yval_z, Xte, yte_z)
    rng = np.random.default_rng(7)
    perm = [per_param_r2(dual_ridge_from_cache(ytr_z[rng.permutation(len(ytr_z))],
                                               cache) * ysd + ymu, yte)
            for _ in range(N_SHUFFLES)]
    chance = {n: float(np.mean([p[n] for p in perm])) for n in PARAM_NAMES}

    # ---- verdict ------------------------------------------------------------
    best_min = max(e["min_param_r2"] for e in landscape + sweep)
    best_mean = max(e["mean_param_r2"] for e in landscape + sweep)
    if best_mean <= KILL_BAR and best_min <= KILL_BAR:
        verdict = "KILL"
        detail = (f"no setting recovers any parameter above R^2 {KILL_BAR} "
                  f"(best min {best_min:.3f}) — the field does not carry through glyphs")
    elif knee is not None and best_min >= KEEP_BAR:
        verdict = "KEEP"
        detail = (f"all-params R^2 >= {KEEP_BAR} reached (best min {best_min:.3f}) "
                  f"and the knee exists at {knee['density_cols']} cols "
                  f"({knee['chars_per_frame']} chars/frame, "
                  f"{knee['encode_ms_per_frame']} ms/frame)")
    else:
        verdict = "MARGINAL"
        detail = (f"signal present (best min {best_min:.3f}) but the reflex bar "
                  f"(all params >= {KEEP_BAR} at >= 2 resolutions) not met")

    # human-eye sample: the field, as text (canonical V=32, top dial)
    fr0 = scene.to_rgb(scene.field(*splits[0]["y"][0]))
    sample = render(fr0, st_top).text[:10]

    out = {
        "experiment": "t1-thermal-glyph-fields-iid",
        "repo": "chiaroscuro-embedding",
        "companion_run": "thermal_glyph.py + results/t1_thermal.json (temporal-split 3-source run, verdict KILL); this controlled i.i.d. run isolates identity-labeling as the suspected failure cause",
        "question": "can a text-only reader recover a thermal scene (source x/y, intensity) from chiaroscuro glyph text — at what encoding cost?",
        "directive": "casey 2026-09-28: same 2ASCII for more than visible light — thermal embedding at any resolution for creature reflexes/motion ML",
        "scene": "2 x-ordered heat sources (gaussian sigma 20px fixed [v2: v1 sigma 9px failed a raw-field control — even un-glyphed fields capped at R2~0.4; template too small for the linear reader class], amp 18-80C over 12C ambient, diffusion blur + 0.8C sensor noise); identity = leftmost-first, deterministic; i.i.d. uniform per frame",
        "canonical": f"V={BASE_DENSITY} ({BASE_DENSITY}x{int(BASE_DENSITY / (4 / 3))} = {int(BASE_DENSITY * (3 / 4) * BASE_DENSITY)} chars/frame, G1 reference density; aspect 4/3)",
        "data": {"n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
                 "targets": list(PARAM_NAMES),
                 "sampling": "i.i.d. uniform per frame, disjoint seeds (house style c1)"},
        "readers": {"ridge": "exact dual ridge on native text features, alpha in [10,100,1000] on val",
                    "mlp": "PCA-100 -> StandardScaler -> sklearn MLPRegressor(64)",
                    "views": "native = c1 lossless read; normass = same grid losslessly re-read as (cell/mass, mass) — removes the intensity-amplitude confound that caps linear position readout (raw-field control: even un-glyphed fields capped ~0.5 for the native view)",
                    "per_param": "max over {ridge, mlp} x {native, normass}"},
        "scoring": {"per_param": "max(ridge, mlp) test R^2",
                    "knee": f"smallest density with ALL params >= {KEEP_BAR}",
                    "gate": f"KILL iff nothing > {KILL_BAR}; KEEP iff all params >= {KEEP_BAR} somewhere AND knee holds at >= 2 densities; else MARGINAL"},
        "chance_empirical": {"method": f"ridge, permuted train labels, {N_SHUFFLES} shuffles, top setting ({top_view} view)",
                             "mean_test_r2": chance},
        "verdict": verdict,
        "verdict_detail": detail,
        "resolution_knee": knee,
        "landscape_density32": sorted(landscape, key=lambda e: -e["mean_param_r2"]),
        "resolution_sweep": sweep,
        "sample_text_frame_top_dial": sample,
        "runtime_s": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=2))  # THE verdict — stdout stays pure JSON
    log(f"[t1-iid] done in {out['runtime_s']}s -> verdict {verdict}")


if __name__ == "__main__":
    main()
