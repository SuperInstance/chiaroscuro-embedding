"""
renderer.py — chiaroscuro core pipeline, re-implemented in numpy + opencv.

"Do NOT modify SuperInstance/chiaroscuro. Read it; re-implement the pipeline here."

This is a faithful port of the GLYPH engine path in `chiaroscuro/studio.html`
(the `P.engine === 'glyph'` branch) plus the `mirror.html` luminance/contrast
logic, reduced to the parts that produce *text*:

    frame -> downsample (1 cell = 1 char) -> luminance (0.299R+0.587G+0.114B)
          -> image dials (contrast, gamma, black/white point, invert, posterize)
          -> Sobel edge (angle + magnitude)
          -> glyph choice  (edge slope family  OR  tone ramp)
          -> text frame (rows x cols of characters)
          -> [optional] cell color (colMode)

Deliberately NOT ported (out of scope for c1, documented in README):
  pixel/braille/match/halftone engines, braille-detail branch, jitter,
  kaleidoscope, glitch, scanlines, grain, vignette, bloom, trail, temporal
  smooth, dither. None of them change which CHARACTERS are chosen, except the
  braille branch — that one is skipped entirely (brailleAmt = 0).

Everything here is pure numpy/cv2: no GPU, no torch.

Text-frame ground truth for the feature reader lives in the GLYPH->feature map
below (`RenderResult.channels`). See sweep.py for how it is consumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ramps + directional glyphs — copied verbatim from studio.html DEFS.engine
# ---------------------------------------------------------------------------
RAMPS: Dict[str, str] = {
    "classic": " .:-=+*#%@@",
    "fine": ''' .'`^",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$''',
    "blocks": " ░▒▓█",
    "minimal": " .:*#",
    "stars": "·∴∗✦✱✳❋█",
    "runes": "·ᚠᚢᚦᚨᚱᚲᚷᚹᚺᚾᛁᛃ█",
    "binary": " 01",
    "code": ''' {}[]()<>/\\|=+*''',
    "waves": "·≈≋▒▓█",
}

# studio.html:  DIRS = [['·','─','━','═','█'], ['·','╱','╱','▚','▓'],
#                       ['·','│','┃','║','█'], ['·','╲','╲','▞','▓']]
EDGE_DIRS: Tuple[Tuple[str, ...], ...] = (
    ("·", "─", "━", "═", "█"),   # family 0: ~0 deg   (horizontal)
    ("·", "╱", "╱", "▚", "▓"),   # family 1: ~45 deg
    ("·", "│", "┃", "║", "█"),   # family 2: ~90 deg  (vertical)
    ("·", "╲", "╲", "▞", "▓"),   # family 3: ~135 deg
)
EDGE_DIR_ANGLE_DEG = (0.0, 45.0, 90.0, 135.0)
_EDGE_FLAT = np.array([c for row in EDGE_DIRS for c in row], dtype="<U1")


@dataclass
class Settings:
    """The studio dials that matter for the text frame (names kept 1:1)."""

    density: int = 100          # P.density  -> columns of chars
    aspect: float = 4.0 / 3.0   # source w/h, used for rows (studio: cols/aspect)
    ramp: str = "classic"
    ramp_custom: str = ""
    tone_amt: float = 100.0     # P.toneAmt
    edge_amt: float = 0.0       # P.edgeAmt
    edge_gain: float = 8.0      # P.edgeGain
    edge_style: str = "sobel"   # sobel | emboss | laplace
    edge_thresh: float = 4.0    # P.edgeThresh (studio compares mag > thr/100)
    relief: float = 55.0        # P.relief
    light_angle: float = 315.0  # P.lightAngle (only used for color boost)
    brightness: float = 0.0     # P.brightness
    contrast: float = 25.0      # P.contrast
    gamma: float = 100.0        # P.gamma
    black_pt: float = 0.0       # P.blackPt
    white_pt: float = 255.0     # P.whitePt
    invert: bool = False
    posterize: int = 0
    # color (text frame does not depend on these; kept for fidelity)
    col_mode: str = "original"  # original | phosphor | duotone | heatmap | ink
    hue: float = 120.0
    sat: float = 100.0
    temp: float = 0.0
    shadow_col: str = "#0a1030"
    high_col: str = "#ffd166"

    def rows_for(self) -> int:
        return max(2, int(self.density / self.aspect))

    def as_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class RenderResult:
    """One rendered frame."""

    text: List[str]                 # the text frame: rows of chars (the "pixels")
    chars: np.ndarray               # (rows, cols) '<U1'
    channels: np.ndarray            # (4, rows, cols) float32 feature fields
    lum: np.ndarray                 # (rows, cols) float32, 0..255 post-dials
    mag: np.ndarray                 # (rows, cols) float32, sobel magnitude/1020
    kind: np.ndarray                # (rows, cols) int8: 0 blank, 1 tone, 2 edge

    @property
    def shape(self) -> Tuple[int, int]:
        return self.chars.shape


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def active_ramp(s: Settings) -> str:
    if s.ramp == "custom" and s.ramp_custom:
        return s.ramp_custom
    return RAMPS.get(s.ramp, RAMPS["classic"])


def hex_to_rgb(hx: str) -> Tuple[float, float, float]:
    hx = hx.lstrip("#")
    return tuple(float(int(hx[i:i + 2], 16)) for i in (0, 2, 4))  # type: ignore


def downsample(frame_rgb: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """1 cell = 1 character. INTER_AREA == the browser's box-ish downscale."""
    return cv2.resize(frame_rgb, (int(cols), int(rows)), interpolation=cv2.INTER_AREA)


def to_luminance(frame_rgb: np.ndarray) -> np.ndarray:
    """0.299R + 0.587G + 0.114B (studio: standard human perception luminance)."""
    f = frame_rgb.astype(np.float32)
    return 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]


def tone_field(frame_rgb: np.ndarray, s: Settings) -> np.ndarray:
    """mirror.html contrast stretch + studio loop's dial chain. Returns 0..255."""
    L = to_luminance(frame_rgb) / 255.0
    con = s.contrast / 25.0
    bri = s.brightness * 2.2
    L = np.clip((L - 0.5) * con + 0.5 + bri / 2550.0, 0.0, 1.0)
    L = np.power(L, s.gamma / 100.0)
    L = np.clip((L * 255.0 - s.black_pt) / max(1.0, s.white_pt - s.black_pt), 0.0, 1.0)
    if s.invert:
        L = 1.0 - L
    if s.posterize > 1:
        lv = s.posterize - 1
        L = np.round(L * lv) / lv
    return (L * 255.0).astype(np.float32)


def gradient_fields(lum: np.ndarray, edge_style: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sobel / emboss / laplacian. studio indexes are index-clamped (replicate)."""
    if edge_style == "emboss":
        # gxv = g22-g00 ; gyv = g20-g02  (the "one-side" emboss)
        kx = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float32)
        ky = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]], dtype=np.float32)
        gx = cv2.filter2D(lum, cv2.CV_32F, kx, borderType=cv2.BORDER_REPLICATE)
        gy = cv2.filter2D(lum, cv2.CV_32F, ky, borderType=cv2.BORDER_REPLICATE)
    elif edge_style == "laplace":
        gx = cv2.filter2D(lum, cv2.CV_32F, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
                          borderType=cv2.BORDER_REPLICATE)
        gy = np.zeros_like(gx)
    else:  # sobel — identical to cv2.Sobel ksize=3 kernels
        gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
        gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    mag = np.sqrt(gx * gx + gy * gy) / 1020.0
    return gx, gy, mag


def _hsl_to_rgb(h_deg, s01, l01):
    """vectorized hsl2rgb (mirrors studio's hsl2rgb)."""
    h = (np.asarray(h_deg, dtype=np.float32) % 360.0) / 60.0
    c = (1 - np.abs(2 * l01 - 1)) * s01
    x = c * (1 - np.abs(h % 2 - 1))
    m = l01 - c / 2
    r = np.zeros_like(h); g = np.zeros_like(h); b = np.zeros_like(h)
    for lo, hi, (rc, gc, bc) in (
        (0, 1, (c, x, 0)), (1, 2, (x, c, 0)), (2, 3, (0, c, x)),
        (3, 4, (0, x, c)), (4, 5, (x, 0, c)), (5, 6, (c, 0, x)),
    ):
        sel = (h >= lo) & (h < hi)
        r = np.where(sel, rc, r); g = np.where(sel, gc, g); b = np.where(sel, bc, b)
    return np.stack([(r + m) * 255.0, (g + m) * 255.0, (b + m) * 255.0], axis=-1)


def paint_colors(frame_rgb_small: np.ndarray, lum: np.ndarray, s: Settings) -> np.ndarray:
    """studio cellColor() for the cell background/source color (H,W,3 float)."""
    f = frame_rgb_small.astype(np.float32)
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    sat = s.sat / 100.0
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    r = gray + (r - gray) * sat
    g = gray + (g - gray) * sat
    b = gray + (b - gray) * sat
    if s.temp:
        t = s.temp * 0.35
        r = np.minimum(255.0, r + t); b = np.maximum(0.0, b - t)
    L = lum
    if s.col_mode == "phosphor":
        return _hsl_to_rgb(s.hue, np.minimum(100.0, sat * 80.0) / 100.0,
                           np.minimum(92.0, L * 0.85 + 4.0) / 100.0)
    if s.col_mode == "duotone":
        a = np.array(hex_to_rgb(s.shadow_col), dtype=np.float32).reshape(1, 1, 3)
        c = np.array(hex_to_rgb(s.high_col), dtype=np.float32).reshape(1, 1, 3)
        t = np.clip(L / 255.0, 0.0, 1.0)[:, :, None]
        return a + (c - a) * t
    if s.col_mode == "heatmap":
        return _hsl_to_rgb((300 - L / 255.0 * 300 + 360) % 360, 0.85,
                           np.minimum(70.0, 20.0 + L * 0.22) / 100.0)
    if s.col_mode == "ink":
        v = 255.0 - L
        return np.stack([v, v, v], axis=-1)
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# the core: frame -> text
# ---------------------------------------------------------------------------
def render(frame_rgb: np.ndarray, s: Settings) -> RenderResult:
    """The chiaroscuro GLYPH pipeline: image -> text frame + feature fields."""
    cols, rows = int(s.density), s.rows_for()
    small = downsample(frame_rgb, cols, rows)
    lum = tone_field(small, s)
    gx, gy, mag = gradient_fields(lum, s.edge_style)
    ramp = active_ramp(s)
    rl = len(ramp)
    ramp_arr = np.array(list(ramp), dtype="<U1")
    H, W = lum.shape

    # --- edge branch: glyph chosen by slope family + magnitude bucket ------
    edge_mask = (s.edge_amt > 0) & (mag > s.edge_thresh / 100.0)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    fam = np.zeros((H, W), dtype=np.int32)
    fam[(ang >= 22.5) & (ang < 67.5)] = 1
    fam[(ang >= 67.5) & (ang < 112.5)] = 2
    fam[(ang >= 112.5) & (ang < 157.5)] = 3
    relief = s.relief / 100.0
    level = np.clip(np.floor(mag * 9.0 * (s.edge_gain / 8.0) * (0.5 + relief)), 0, 4).astype(np.int32)
    edge_code = fam * 5 + level

    # --- tone branch: ramp index from (1 - L/255) --------------------------
    t = np.clip(lum / 255.0, 0.0, 0.999999)
    tone_idx = np.clip(np.floor((1.0 - t) * (rl - 1)).astype(np.int32), 0, rl - 1)
    tone_mask = (s.tone_amt > 5.0) & (~edge_mask)   # studio: toneMix=toneAmt/100 > 0.05
    tone_mask &= (ramp_arr[tone_idx] != " ")          # studio: ' ' -> blank

    chars = np.full((H, W), " ", dtype="<U1")
    chars[tone_mask] = ramp_arr[tone_idx[tone_mask]]
    chars[edge_mask] = _EDGE_FLAT[edge_code[edge_mask]]

    kind = np.zeros((H, W), dtype=np.int8)
    kind[tone_mask] = 1
    kind[edge_mask] = 2

    # --- text-equivalent feature fields (what a text reader gets to see) ---
    tone_scalar = np.where(tone_mask, (tone_idx + 1.0) / rl, 0.0).astype(np.float32)
    edge_scalar = np.where(edge_mask, (level + 1.0) / 5.0, 0.0).astype(np.float32)
    fam_deg = np.take(np.array(EDGE_DIR_ANGLE_DEG, dtype=np.float32), fam)
    dir_x = np.where(edge_mask, np.cos(np.radians(fam_deg)), 0.0).astype(np.float32)
    dir_y = np.where(edge_mask, np.sin(np.radians(fam_deg)), 0.0).astype(np.float32)
    channels = np.stack([tone_scalar, edge_scalar, dir_x, dir_y], axis=0)

    return RenderResult(
        text=["".join(row) for row in chars],
        chars=chars,
        channels=channels,
        lum=lum,
        mag=mag,
        kind=kind,
    )


def render_text(frame_rgb: np.ndarray, s: Settings) -> List[str]:
    return render(frame_rgb, s).text


if __name__ == "__main__":  # tiny smoke test: python renderer.py
    from scene import MockScene

    sc = MockScene()
    frame, params = sc.frame(0.30, 0.60, 35.0)
    for name, st in (
        ("tone only", Settings(ramp="classic", density=80, tone_amt=100, edge_amt=0)),
        ("edge only", Settings(ramp="classic", density=80, tone_amt=0, edge_amt=100,
                               edge_thresh=4, relief=70, edge_gain=12)),
    ):
        out = render(frame, st)
        print(f"--- {name} {out.shape} params={params} ---")
        print("\n".join(out.text))
