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

---

# c3 — framerate vs pixel loss at a fixed character budget

**Question (the seed claim, quantified):** *"chiaroscuro can crank framerate to
make up for pixel loss."* At a fixed TOKEN BUDGET (chars/sec), is information
better carried by **MORE frames at LOWER resolution** or **FEWER frames at
HIGHER resolution**?

**Method:** one 20 fps master timeline (c2's five lavfi sources, 10 s each,
1000 frames, 320x240) labeled by CLIP at the master rate. Twelve cells:
framerate {2.5, 5, 10, 20} x density {40, 80, 160} cols — each cell samples the
master at its stride and renders through the glyph pipeline at c2's default mix
dials. Budget = fps x cols x rows (fixed-width text frame, blanks count). The
grid contains **four exact matched-budget pairs** where the high-fps cell has
4x the framerate and 1/4 the glyph area: 12k c/s (2.5/80 vs 10/40), 24k c/s
(5/80 vs 20/40), 48k c/s (2.5/160 vs 10/80), 96k c/s (5/160 vs 20/80). The
reader is IDENTICAL for every cell — 8x6 block-pooled histograms over the 23-
glyph vocabulary (dim 1104, constant) — so capacity never confounds the
comparison; c2's exact readers, blocked 5-fold temporal CV. Stacked variant
concatenates the last k=4 frames' text (low-fps cells get a longer temporal
reach per stack — their fair shot; budget 4x, pairs stay matched). **No margin
filter** (paired comparison, identical label noise). Gates (pre-registered):
FRAMERATE-WINS iff >= 2/4 matched pairs favor the high-fps cell by >= 5 acc
points (and more pairs favor it); RESOLUTION-WINS the reverse; else TIE.
Headline = stacked variant.

## Result (2026-09-28): **TIE**

| fps | cols | chars/frame | chars/sec | n | single | stacked |
|----:|-----:|------------:|----------:|---:|-------:|--------:|
| 2.5 | 40 | 1,200 | 3,000 | 125 | 0.960 | 0.960 |
| 5.0 | 40 | 1,200 | 6,000 | 250 | 0.960 | 0.980 |
| 2.5 | 80 | 4,800 | 12,000 | 125 | 0.928 | 0.936 |
| 10.0 | 40 | 1,200 | 12,000 | 500 | 0.970 | 0.986 |
| 5.0 | 80 | 4,800 | 24,000 | 250 | 0.952 | 0.944 |
| 20.0 | 40 | 1,200 | 24,000 | 1000 | 0.956 | **0.992** |
| 2.5 | 160 | 19,200 | 48,000 | 125 | 0.904 | 0.936 |
| 10.0 | 80 | 4,800 | 48,000 | 500 | 0.944 | 0.950 |
| 5.0 | 160 | 19,200 | 96,000 | 250 | 0.932 | 0.940 |
| 20.0 | 80 | 4,800 | 96,000 | 1000 | 0.922 | 0.954 |
| 10.0 | 160 | 19,200 | 192,000 | 500 | 0.918 | 0.926 |
| 20.0 | 160 | 19,200 | 384,000 | 1000 | 0.914 | 0.930 |

Matched-pair deltas (stacked, high-fps minus low-fps, acc pts): **−5.0,
−4.8, −1.4, −1.4** → 0 pairs favor high-fps, 1 favor low-fps → TIE (single-
frame ablation: −4.2, −0.4, −4.0, +1.0 → TIE).

**Honest reading of the frontier:** neither frontier dominates — the curve is
essentially FLAT from 3,000 to 384,000 chars/sec (all cells 0.90–0.99 vs a
0.38–0.40 majority prior; constant-class sources again). The seed claim is not
confirmed at these budgets — but neither is it refuted, because the bottleneck
is STILL not binding (third experiment in a row with saturation): even 2.5 fps
at 1,200 chars/frame holds 96%. If anything, the one pair to clear the gate
favored low-fps/high-density (2.5/80 vs 10/40, stacked: −5.0) while the best
cell overall is the high-fps budget cell (20 fps/40 cols, 0.992 stacked) — the
frontiers interleave within noise of each other. Framerate does NOT need
cranking to make up for pixel loss on this task; framerate and glyph resolution
are near-interchangeable at matched char budgets. The claim can only be tested
where the budget actually binds: harder content (motion-differentiated classes —
note CLIP itself merged the two test-pattern sources 98/100 at 20 fps, same as
c2 at 10 fps), harsher budgets (sub-1,000 chars/sec), or confusable classes.
That is a c3b (or c4-adjacent) question.

## Files
- `c3_feed.py` — 20 fps master timeline (rebuilds c2's lavfi graphs at
  `rate=20`, mandelbrot `end_pts` rescaled; reuses `c2_feed.read_source`).
  `C3_SMOKE=1` for the 2 s version.
- `c3_clip.py` — CLIP zero-shot labels on every master frame (c2's caption set
  verbatim; runs in `~/venvs/elephant-gpu`, fp16 GPU, serialized; the ONLY GPU
  step). Writes `c3_frames/clip_labels.npz` + summary.
- `c3_sweep.py` — the 12-cell sweep, matched-budget gate, ONE JSON verdict on
  stdout. `C3_SMOKE=1` for the smoke version.
- `c3_run.sh` — driver: feed → CLIP → sweep. `./c3_run.sh > results/c3-sweep.json`.
- `results/c3-sweep.json` — the full verdict of the run above.
- `c3_frames/` — gitignored (230 MB npy + labels).
