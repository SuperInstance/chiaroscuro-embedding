# chiaroscuro-embedding

A chiaroscuro-embedding agent: **perception → generation → higher abstractions, without ever leaving a text-only base model.**

## The seed (Casey, 2026-09-28)

- [chiaroscuro](https://github.com/SuperInstance/chiaroscuro) renders webcam → text. *"Characters are shapes, not pixels."* Five glyph engines, 45+ dials, ~60fps.
- We **know what's happening in video** because a VLM / multimodal model can interpret it. That gives us a **GAN**: the VLM is the ground truth (discriminator), the renderer is the generator.
- The renderer can **sweep its settings**, hunting for the text representations that carry the information a text-only ML can learn.
- chiaroscuro's edge: it can **crank framerate to make up for pixel loss** (text is nearly free).
- The goal: fine-tune the renderer into an **a2a (agent-to-agent) translation layer** — an embedding agent that starts with perception, then moves to generation and higher abstractions, staying text-only the whole way.

## The cellular thread

Treat every **cell of a quilt like a character** in chiaroscuro. A quilt = a grid of cells = a text frame. The ML learns **inter-relational weighting** — how cells relate (neighbors, slopes, motion) — developing *intuition* and *interpretation* from relational structure, not just ink.

## First experiments (pre-registered, build order)

### c1 — renderer sweep vs synthetic ground truth (no VLM needed)
Render a synthetic scene (chiaroscuro's `?mock=1`: moving light, shaded sphere, rotated bars, checkerboard) through a **Python re-implementation** of the chiaroscuro pipeline at N settings (engine × ramp × threshold × density). Ground truth = the known scene parameters. Train a text-only reader (ridge + small MLP) to recover scene params from the rendered text. **KEEP iff some settings preserve the info (reader >> chance); report the informativeness LANDSCAPE over settings.**

### c2 — VLM as ground truth
Same sweep, but the label is a VLM's interpretation of real footage. The discriminator is a real interpreter; the generator sweeps.

### c3 — framerate vs pixel loss
Quantify: at what text framerate does text-carried information match raw pixels?

### c4 — the a2a layer
A text-only agent (local Liquid/Granite) whose *eyes* are the chiaroscuro text. Perception first; generation later.

## Rules
- **Do NOT modify SuperInstance/chiaroscuro.** Read it; re-implement the pipeline here.
- Seeds, not constraints. Kill what dies; keep the gold (archive-by-rename).
- **Train novel models**, not just frozen probes. Long season, always running.

---

## Wave 2 seeds (Casey, 2026-09-28 11:06)

### Video = free supervision
Any feed of people or animations can be ML'd directly — **the video is the supervision rendering that way.** No labels needed; the content provides them (via VLM or known scene structure). The renderer sweeps over the SAME frames and we measure what survives into text.

### Oscillation & fabulation
The chiaroscuro dials don't have to be static — they can **oscillate, fabulate, and change stochastically** during rendering. Randomness source: **MOTHquantum API (true random)** once keyed; numpy RNG as fallback until then. A render becomes a trajectory through dial-space, not a point.

### MicroMoth-quilt, lower level
[MicroMoth-quilt](https://github.com/SuperInstance/MicroMoth-quilt) is the smallest quantum framework (pure Python, read-in-an-afternoon) wearing the quilt receipt-ledger lane: every gate a BIND cell, every measurement a sealed collapse, fnv1a-64 chained, replayable. Seed: design it **even lower level to distribute better over more cells**.

### cudaclaw-Moth
A **GPU-native, fully-accelerated MicroMoth** — what cudaclaw is to the cell kernel, done for the quantum micro-framework. For scale.

### Meta
**Keep building and using your tools to make and sharpen other tools.** Iron sharpens iron.

## Pending / blocked
- `key.txt` (MOTHquantum.com + Typesafe.ai links): **not found** — searched ~, ~/projects, workspace. Need its location from Casey.
- mothquantum.com is JS-rendered; API docs need a browser pass once the key lands.

---

## Wave 3 seed (Casey, 2026-09-28 16:47) — the CV for PLATO: spatial acoustics as room-shape

- **Spatial-acoustic is its own long-running lane, later.** When it opens:
  - sonar-vision's rendering couples to audio → **ASCII spatial images of sound**.
  - The deeper half: **ray-tracing / path-tracing equations EMBED THE SHAPE OF THE ROOM** — the acoustic ray paths and resonances are a multimodal expression of room geometry. Read the field, read the room.
  - **Scaling law: more transmitters + receivers → resolution of spatial understanding scales easily** in ASCII-first perception (the lattice composes with the array).
  - This is **PLATO's CV**: the creature whose senses include seeing rooms through sound.
- t1 (thermal) proceeds now — quick-encoding heat fields for reflexes (creature motion).
