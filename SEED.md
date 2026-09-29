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

### t2 seed (Casey, 2026-09-28 16:47) — motion signatures for cross-camera identification
- **Thermal at high glyph framerate gives objects a MOTION SIGNATURE** — and motion signatures enable **relative identification ACROSS CAMERAS** (and are far more general-purpose than that, of course).
- The hypothesis that inverts X2: the STATIC visible-light read was renderer-bound (didn't transfer), but the MOTION signature (the glyph-delta stream — G4's changed-cells as the carrier) may transfer across sensors because it's about the OBJECT, not the sensor's parameterization.
- t2 (fires when t1's artifacts exist): same moving heat sources rendered from TWO cameras (different position/scale/pipeline) → extract motion signatures (delta streams) → test cross-camera object matching. Gate: matching >> chance. KILL: signatures are camera-bound like statics were.

### t3 seed (Casey, 2026-09-28 16:49) — the unsteady camera abstracts 4D
- **When the camera isn't steady, high framerate helps abstract 4D** (3D world + time). The unsteady camera is a FEATURE: its ego-motion parallax reveals depth — near objects swing fast in frame, far ones drift slow; the parallax gradient IS the depth map.
- The 4D abstraction from the glyph stream: separate ego-motion from object-motion in the delta stream → recover depth ordering. High framerate makes the correspondence tractable (tiny inter-frame deltas).
- t3: a 3D thermal scene (heat sources at multiple depths) + a camera with known ego-motion (drift/shake), rendered at high fps. Gate: depth ordering recovered from the glyph delta stream >> chance. Same principle as the ray-traced room shape: geometry lives in the dynamics.

## Wave 4 seed (Casey, 2026-09-28 16:52) — the simulation-first paradigm: confirm until surprised, then earn the moment

- **4D in a time-first, SIMULATION-FIRST paradigm** (the tminus / t-minus repos): the creature runs a forward simulation of its world; **framerate becomes a waveform** — each frame a confirmation of the prior calculated simulation, **until surprised**.
- **The arc of surprise**: primitive creatures minimize surprise; the highest form SEEKS it — **jam sessions and sports games, trying for EARNED moments** (deliberate high-surprise states with skin in the game). We are reaching that.
- Today's program already embodies it without having named it: G5's composite = the confirm-until-surprised loop (persistence IS the prior simulation; the diff-head IS the surprise handler). G4's diff-mask = learning only from surprise. t3's unsteady camera = deliberate surprise injection to reveal geometry (the saccade doctrine). The falsifier campaign = surprise-seeking as method. The Tap's agents jamming = the social form.
- t4+ design implication: the predictor and the simulator merge — the glyph model's job is not prediction but CONFIRMATION (cheap, most frames) + surprise detection (the changed cells) + re-simulation on surprise. Framerate is the sampling rate of the confirmation waveform.

## Wave 5 seed (Casey, 2026-09-28 18:44) — the 5TH DIMENSION: resolution-to-framerate as a cognitive axis

- **High-res + sitting still = reference frames** (the map: precision, dial-reading, the elephant's mode). **Low-res + high framerate = 4D extrapolation while moving** (the leap: parallax, motion signatures, t3's depth-from-shake).
- **The 5th dimension is the sensory budget allocation itself** — and modulating it is a BEHAVIOR, not a rendering parameter. The insect's compound eye lives at one fixed point (low-res/high-temporal); the vertebrate fovea at another (high-res during fixation). OUR creature can TRAVERSE the axis at will — a shape in (x,y,z,t,r) space.
- **The simulation-first connection**: still → confirm at high-res (cheap, precise, low-surprise); moving → extrapolate at high-fps (the waveform, surprise-rich). The 5th dimension IS the attention dial between confirm-mode and predict-mode.
- t5 seed: **the dial controller** — an agent that chooses its (resolution, framerate) point from motion energy (moving → 24cols/high-fps; still → 96cols/low-fps), measured on a task where both regimes matter vs every fixed point. Gate: adaptive beats ALL fixed allocations.
