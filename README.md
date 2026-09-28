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
- `c2_feed.py` / `c2_clip.py` / `c2_sweep.py` / `c2_run.sh` — experiment c2
  (VLM as ground truth). `./c2_run.sh` → one JSON verdict on stdout. Feed
  frames land in `c2_frames/` (gitignored). Needs system python3 (cv2/sklearn)
  for feed+sweep and `~/venvs/elephant-gpu` (torch+transformers) for CLIP;
  `C2_SMOKE=1` for the smoke version of every stage.
- `results/c2-sweep.json` — the full verdict of the c2 run below.
- `c2_sources_contact_sheet.png` — the five lavfi sources at t=0/4.9s/9.9s.

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
- c2: same sweep, labels from a VLM on real footage. → done, below.
- c3: framerate vs pixel loss — where does text-carried info stop matching pixels?

---

# c2 — VLM as ground truth (the GAN-ish loop)

**Question:** a real vision model interprets each frame; can a text-only reader
recover that interpretation from chiaroscuro's rendered TEXT of the same frame?
The discriminator is a real interpreter; the generator sweeps.

**Method:** five ffmpeg lavfi feeds (`testsrc2`, `smptebars`, `mandelbrot`,
`life`, `gradients`; 10 s × 10 fps × 320×240 = 500 frames) → CLIP ViT-B/32
zero-shot (cached `openai/clip-vit-base-patch32`, fp16 GPU) scores every frame
against 12 captions (5 source + 7 distractors); label = argmax caption; frames
with top-1/top-2 softmax margin < 0.10 dropped (4/500; margin median 0.86).
The same frames go through the c1 renderer at the same 34 settings, and two
text-only readers (dual ridge on native glyph features, PCA-100→MLP) predict
the caption index. **Splits are temporal per source (70/15/15, no shuffle)** —
adjacent frames are near-duplicates, so random splits would leak the future.
Pixels control = the SAME ridge-one-hot reader on CLIP image embeddings.

## Result (2026-09-28): **KEEP**

- Test majority-class prior = 0.400; pixels control = 1.000.
- **34/34 settings beat the prior by ≥15 points.** 30/34 hit 100% test
  accuracy (+60 pts over prior, 100% of the pixels control). Below the ceiling:
  `tone|posterize=3` (0.80), `edge|style=emboss` / `mix|contrast=10` (0.89),
  `edge|style=laplace` (0.93). Ridge alone ≥ 0.80 everywhere, exactly 1.000 at
  30/34 settings.

**Honest reading:** even flatter than c1 — because the classes are visually
EXTREME (fractal vs cellular automaton vs gradient vs color bars) and every
one has a distinct global luminance/texture signature. The bottleneck is again
nowhere near binding. Two findings that do matter:
1. **The discriminator's interpretation ≠ the generator's intent.** CLIP
   (single-frame) merges testsrc2 into the smptebars caption 96/100 — a still
   frame of a test pattern cannot reveal motion — and reads 3 `life` frames as
   "a night sky full of stars". Labels were kept as CLIP saw them.
2. Text-only readers match the CLIP-embedding control at every setting here:
   whatever CLIP's own features carry about its label, the glyph text carries too.

**Infra notes (future-us):** transformers 5.x `get_*_features` returns
`BaseModelOutputWithPooling` — the projected embeddings are in
`.pooler_output`; and the HF cache for `openai/clip-vit-base-patch32` was
missing tokenizer files, which made the offline tokenizer emit garbage ids
(silent! everything decoded to the same caption) — fixed by letting the hub
download `tokenizer.json` et al. Check tokenizer output before trusting
zero-shot labels.
