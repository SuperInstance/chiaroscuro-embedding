# c1 — renderer sweep vs synthetic ground truth

First experiment for the chiaroscuro-embedding seed (see `SEED.md`).

**Question:** when chiaroscuro crushes a frame into a grid of characters, which
render settings leave enough information in the *text* for a text-only reader
to recover the scene?

**Method:** re-implement the chiaroscuro glyph pipeline in numpy/opencv
(`renderer.py`), render a synthetic `?mock=1`-style scene with known ground
truth (`scene.py`: shaded sphere position + rotating bar angle, static
checkerboard/light as no-info distractors), sweep 34 render settings, and train
two text-only readers (exact dual kernel ridge + PCA→small MLP) to recover the
scene parameters from the rendered character grid alone. Chance floor is
measured by training the same ridge on permuted labels.

## Files
- `renderer.py` — the chiaroscuro core (glyph engine path of `studio.html`):
  1 cell = 1 char downsample → 0.299R+0.587G+0.114B luminance → dial chain
  (contrast/gamma/black/white/invert/posterize) → Sobel/emboss/laplace →
  glyph choice (edge slope family × relief level **or** tone ramp) → text.
  Not ported (change pixels or color, not glyph choice): pixel/braille/match/
  halftone engines, jitter, kaleido, glitch, scanlines, grain, vignette, bloom,
  trail, temporal smooth, dither.
- `scene.py` — the mock scene. Deviations from the real `?mock=1` (on purpose,
  to avoid confounds): key light and checkerboard parity are frozen; sphere
  params are sampled i.i.d. uniform instead of sinusoidal.
- `sweep.py` — the sweep. `python3 sweep.py` → one JSON verdict on stdout,
  progress on stderr. `C1_SMOKE=1` for a 15-second version.
- `results/c1-sweep.json` — the full verdict of the run below.

## Result (2026-09-28): **KEEP**

- 400 frames (280/40/80 train/val/test), 34 settings.
- **All 34 settings recover all 3 scene params at test R² ≥ 0.5** — in fact
  ridge R² ≥ 0.94 on every param at every setting (mean-of-best-per-param
  ranges 0.936–0.982). Permuted-label chance is ≈ −0.4 to −0.6.
- Gate (≥ 2/3 params at R² ≥ 0.5 in ≥ 1 setting): passed 3/3 × 34 settings.
- Top settings: `edge|style=emboss`, `preset|green-matrix`, `preset|sculptor`;
  weakest (still passing easily): `edge|style=laplace`, `mix|density=40`,
  `tone|posterize=3`.

**Honest reading of the landscape:** it is nearly FLAT. This mock scene is
easy — bright objects on a dark radial background with strong luminance
signatures — and chiaroscuro's text preserves the geometry under every tested
ramp, density (40–130 cols), edge threshold, edge mix, contrast, posterize, and
all shipped presets. Even a 5-glyph `binary` ramp or edge-only laplace keeps
R² ≳ 0.94. The information bottleneck is nowhere near binding here. The
landscape will only differentiate on harder scenes (smaller objects, lower
contrast, clutter, or real footage) — that is exactly what c2/c3 are for.

**Fidelity note:** verified line-by-line against `studio.html` (manual Sobel ==
`cv2.Sobel ksize=3` with replicate borders; tone index `(1−t)(rl−1)`; edge
families/levels; DIRS glyph table; dial chain order). Do not modify
`SuperInstance/chiaroscuro` — this repo is the re-implementation home.

## Next
- c2: same sweep, labels from a VLM on real footage.
- c3: framerate vs pixel loss — where does text-carried info stop matching pixels?
