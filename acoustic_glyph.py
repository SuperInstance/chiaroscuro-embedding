"""
acoustic_glyph.py — experiment a1: acoustic glyph fields.

Second modality in Casey's directive (2026-09-28): "we can do this same 2ASCII
for more than just visible light..." — here the 2D scalar field is a
SPECTROGRAM (time x frequency, intensity = energy) of a moving chirping
source. The standard glyph pipeline (downsample -> tone/edge codes) turns the
sonogram into text; the c1 question: can a *text-only* reader recover the
acoustic scene parameters from the glyph text alone?

Scene (ground truth by construction):
  spectrogram stage 240x180 (time x freq), frequencies normalized to Nyquist
  source moves along a line: x(t) = x0 + vel * t        (vel in [-0.35, 0.35])
  its chirp is position-tagged:  f(t) = f0 + 0.25 * x(t)  (rising pitch = right)
  six chirp bursts (gaussian envelopes, width 0.035) per clip; a fundamental
  ridge (sigma_f 0.028) + a 0.45-amplitude harmonic at 2f (falls off-frame at
  high f0, as real harmonics do)
  DISTANCE cue disambiguates (x0, f0) — the freq intercept alone is the combo
  f0 + 0.25*x0, so amplitude also tracks proximity: A(t) = amp * (0.35 + 0.65*(1 - x(t)))
  (source louder on the left = closer to the mic). With the amplitude trajectory
  pinning x(t), all four params are identifiable.
  background noise floor 0.05 + sensor noise, gaussian smear (sigma 2.2 px)
  = window smearing / small-room reverb.

  params per frame: (x0, vel, f0, inten); i.i.d. uniform, disjoint seeds.

Canonical density V=32 (32x24 chars) as in t1; the same resolution sweep is
run so the per-modality gate (including the knee clause) is checkable.

GATE (pre-registered, same as t1):
  KILL   iff no setting recovers any parameter at test R^2 > 0.5
  KEEP   iff some setting recovers ALL parameters at test R^2 >= 0.90 AND the
         resolution knee exists (all-params >= 0.90 holds at >= 2 densities)
  else   MARGINAL

Usage:
  python acoustic_glyph.py                 # full sweep, ONE JSON verdict on stdout
  A1_SMOKE=1 python acoustic_glyph.py      # ~2-min smoke

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

SMOKE = bool(os.environ.get("A1_SMOKE"))
N_TRAIN, N_VAL, N_TEST = (60, 12, 24) if SMOKE else (280, 40, 80)
DENSITIES_SWEEP = (24, 32, 48) if SMOKE else (24, 32, 48, 96, 192)
N_SHUFFLES = 5 if SMOKE else 20
BASE_DENSITY = 32            # V=32, the G1 reference
KEEP_BAR = 0.90
KILL_BAR = 0.50
ALPHAS = (10.0, 100.0, 1000.0)
PCA_DIMS = 100
PARAM_NAMES = ("x0", "vel", "f0", "inten")

# chirp code (fixed by convention, known to the scene, unknown to the reader)
KF = 0.25                    # position -> frequency coupling
BURST_T = (0.07, 0.22, 0.38, 0.53, 0.69, 0.85)
BURST_W = 0.05               # v2: was 0.035 (~0.8 time cells at V=32 — sub-cell)
SIGMA_F = 0.05               # v2: was 0.028 (~0.7 freq cells at V=32 — sub-cell)
HARMONIC = 0.45


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# the acoustic scene: spectrogram -> gray RGB for the glyph pipeline
# ---------------------------------------------------------------------------
class AcousticScene:
    """Moving chirping source -> spectrogram field (time x freq, energy)."""

    def __init__(self, width: int = 240, height: int = 180,
                 smear_sigma: float = 3.0, noise: float = 0.008,
                 floor: float = 0.05):
        self.w, self.h = int(width), int(height)   # w = time, h = freq
        self.smear_sigma = smear_sigma
        self.noise = noise
        self.floor = floor
        self.t = np.linspace(0.0, 1.0, self.w, dtype=np.float32)
        self.f = np.linspace(0.0, 1.0, self.h, dtype=np.float32)
        env = np.zeros_like(self.t)
        for tb in BURST_T:
            env += np.exp(-((self.t - tb) / BURST_W) ** 2)
        self._env = env

    def field(self, x0: float, vel: float, f0: float, inten: float,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
        x_t = x0 + vel * self.t
        f_t = f0 + KF * x_t                       # position-tagged pitch
        amp = (0.30 + 0.65 * inten) * (0.35 + 0.65 * (1.0 - x_t))
        burst = amp[None, :] * self._env[None, :]  # (1, T)
        df = self.f[:, None] - f_t[None, :]        # (F, T)
        E = self.floor \
            + burst * np.exp(-(df / SIGMA_F) ** 2) \
            + HARMONIC * burst * np.exp(-((self.f[:, None] - 2.0 * f_t[None, :]) / SIGMA_F) ** 2)
        E = cv2.GaussianBlur(E, (0, 0), self.smear_sigma)  # window smear / reverb
        if rng is not None and self.noise > 0:
            E = E + rng.normal(0.0, self.noise, E.shape).astype(np.float32)
        return E

    def to_rgb(self, E: np.ndarray) -> np.ndarray:
        g = np.clip(E, 0.0, 1.0) * 255.0
        return np.repeat(g[:, :, None], 3, axis=2).astype(np.uint8)

    def sample_params(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.stack([
            rng.uniform(0.25, 0.75, n),      # x0
            rng.uniform(-0.35, 0.35, n),     # vel
            rng.uniform(0.20, 0.60, n),      # f0
            rng.uniform(0.0, 1.0, n),        # inten
        ], axis=1).astype(np.float32)

    def sequence(self, n: int, seed: int) -> Tuple[List[np.ndarray], np.ndarray]:
        rng = np.random.default_rng(seed)
        P = self.sample_params(n, rng)
        frames = [self.to_rgb(self.field(*P[k], rng=rng)) for k in range(n)]
        return frames, P


# ---------------------------------------------------------------------------
# readers (c1 house style, verbatim pattern from sweep.py)
# ---------------------------------------------------------------------------
def featurize_all(scene: AcousticScene, s: Settings, frames: List[np.ndarray],
                  view: str = "native") -> np.ndarray:
    """Render every frame -> (n, D) float32 text features.

    view="native": per cell (tone-ramp index normalized, edge level, edge
    dir x, edge dir y) — the c1 lossless read of the glyph grid.

    view="normass": the same grid re-read LOSSLESSLY as (cell / total ink
    mass, mass) — removes the absolute-amplitude confound for LINEAR readers
    (same lesson as t1-iid; relative shape over time/freq — the chirp ridge's
    slope and envelope — survives normalization; loudness moves to the mass
    feature). Both views reported everywhere; gate uses the per-param max.
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
# dial landscape + resolution sweep (thermal/sonar-adapted thresholds)
# ---------------------------------------------------------------------------
def build_grid() -> List[Tuple[str, Settings]]:
    """9 dials at the canonical V=32. Acoustic adaptation: chirp ridges are
    thin + smooth, so low edge thresholds, high gain/relief. Polarity NOTE
    (same lesson as t1-iid): invert=True blanks the noise floor via the tone
    mask (only ridge cells render); invert=False keeps every cell inked, so
    the tone dials default to it — one 'ridgeink' variant quantifies the
    blanking cost."""

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
    g.append(("tone|waves|ridgeink", mix(edge_amt=0, ramp="waves", invert=True)))
    g.append(("edge|thr=1", mix(tone_amt=0, edge_amt=100, edge_thresh=1,
                                edge_gain=14, relief=75)))
    g.append(("edge|emboss", mix(tone_amt=0, edge_amt=100, edge_thresh=1,
                                 edge_style="emboss", edge_gain=14, relief=75)))
    return g


def run_setting(scene: AcousticScene, st: Settings, splits, ymu, ysd,
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
    scene = AcousticScene()
    log(f"[a1] acoustic scene: moving chirping source -> spectrogram "
        f"{scene.w}x{scene.h} (time x freq), canonical V={BASE_DENSITY}")

    log(f"[a1] generating frames: {N_TRAIN}/{N_VAL}/{N_TEST} (train/val/test)")
    splits = []
    for name, n, seed in (("train", N_TRAIN, 5101), ("val", N_VAL, 5202),
                          ("test", N_TEST, 5303)):
        frames, y = scene.sequence(n, seed=seed)
        splits.append({"name": name, "frames": frames, "y": y})
        log(f"  {name}: {n} frames, y{y.shape}")
    ytr, yval, yte = splits[0]["y"], splits[1]["y"], splits[2]["y"]
    ymu, ysd = ytr.mean(0), ytr.std(0)
    ytr_z, yval_z, yte_z = ((y - ymu) / ysd for y in (ytr, yval, yte))

    # ---- dial landscape at V=32 -------------------------------------------
    grid = build_grid()
    log(f"[a1] dial landscape at V={BASE_DENSITY}: {len(grid)} settings x 2 feature views ...")
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
            f"mean={res['mean_param_r2']:+.3f} ge090={res['n_params_ge_090']}/4 "
            f"[nat {res['views']['native']['mean_param_r2']:+.2f} / "
            f"nrm {res['views']['normass']['mean_param_r2']:+.2f}] "
            f"({time.time() - ts:.1f}s)")

    # ---- resolution sweep on the strongest dial ---------------------------
    top = max(landscape, key=lambda e: e["mean_param_r2"])
    st_top = next(s for n, s in grid if n == top["name"])
    log(f"[a1] resolution sweep on '{top['name']}' over densities {DENSITIES_SWEEP} ...")
    sweep = []
    for dens in DENSITIES_SWEEP:
        ts = time.time()
        st = Settings(**{**st_top.as_dict(), "density": dens})
        rows = st.rows_for()
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
            f"ge090={entry['n_params_ge_090']}/4 ({time.time() - ts:.1f}s)")

    # ---- knee ---------------------------------------------------------------
    ok = [e for e in sweep if e["all_params_ge_090"]]
    knee = None
    if len(ok) >= 2:
        k = min(ok, key=lambda e: e["density_cols"])
        knee = {"density_cols": k["density_cols"],
                "chars_per_frame": k["chars_per_frame"],
                "encode_ms_per_frame": k["encode_ms_per_frame"],
                "min_param_r2": k["min_param_r2"],
                "rationale": f"smallest density with all 4 params >= {KEEP_BAR}; "
                             f"holds at {len(ok)}/{len(sweep)} swept densities "
                             f"({[e['density_cols'] for e in ok]})"}

    # ---- empirical chance floor ---------------------------------------------
    top_view = max(("native", "normass"),
                   key=lambda v: top["views"][v]["mean_param_r2"])
    log(f"[a1] permutation null ({N_SHUFFLES} shuffles) on '{top['name']}' "
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

    fr0 = scene.to_rgb(scene.field(*splits[0]["y"][0]))
    sample = render(fr0, st_top).text[:10]

    out = {
        "experiment": "a1-acoustic-glyph-fields",
        "repo": "chiaroscuro-embedding",
        "question": "can a text-only reader recover an acoustic scene (start pos, velocity, pitch, loudness) from a chirp spectrogram pushed through chiaroscuro glyph text — at what encoding cost?",
        "directive": "casey 2026-09-28: same 2ASCII for more than visible light — spatial/acoustic fields for creature reflexes/motion ML",
        "scene": "moving chirping source -> 240x180 spectrogram (time x freq): 6 bursts, position-tagged pitch f=f0+0.25x, proximity-coded amplitude, harmonic, noise floor + gaussian smear [v2: burst width 0.05, sigma_f 0.05 — v1's 0.035/0.028 were sub-cell at V=32, the raw-field lesson from t1]; i.i.d. uniform per frame",
        "canonical": f"V={BASE_DENSITY} ({BASE_DENSITY}x{int(BASE_DENSITY * (3 / 4))} = {int(BASE_DENSITY * (3 / 4) * BASE_DENSITY)} chars/frame, G1 reference density; aspect 4/3)",
        "data": {"n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
                 "targets": list(PARAM_NAMES),
                 "sampling": "i.i.d. uniform per frame, disjoint seeds (house style c1)"},
        "readers": {"ridge": "exact dual ridge on native text features, alpha in [10,100,1000] on val",
                    "mlp": "PCA-100 -> StandardScaler -> sklearn MLPRegressor(64)",
                    "views": "native = c1 lossless read; normass = same grid losslessly re-read as (cell/mass, mass) — removes the absolute-amplitude confound for linear readers (t1-iid raw-field lesson)",
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
    log(f"[a1] done in {out['runtime_s']}s -> verdict {verdict}")


if __name__ == "__main__":
    main()
