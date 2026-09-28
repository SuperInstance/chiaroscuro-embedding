"""
scene.py — the synthetic scene generator for c1.

Mirrors chiaroscuro's `?mock=1` scene (studio.html, the MOCK branch):

    dark background + radial key light
    a smoothly SHADED SPHERE (radial highlight gradient) that moves
    two ROTATING BARS (one light, one mid-grey, +45 deg apart)
    a STATIC CHECKERBOARD patch (a distractor: it carries no param info)

Ground truth is known by construction: we choose the params, we render, we
hand the renderer only pixels.

Params (all normalized to [0,1] except the angle):
    sphere_x   -> sphere horizontal position
    sphere_y   -> sphere vertical position
    bar_angle  -> rotation of the light bar, degrees in [0, 180)

Sampling is i.i.d. uniform per frame (fills the target space; the c1 question
is "can text carry the info at all", not "can it track a trajectory").
The key light is held STATIC (mock animates it; a moving light would confound
sphere position by changing the whole background gradient frame to frame).
The checkerboard parity is also held static for the same reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# 1x1 px operations below are pure numpy (no cv2 needed here).


def _radial_ramp(dist: np.ndarray, stops: Tuple[float, ...],
                 colors: Tuple[Tuple[float, float, float], ...]) -> np.ndarray:
    """Piecewise-linear radial gradient, like canvas createRadialGradient."""
    out = np.zeros(dist.shape + (3,), dtype=np.float32)
    for i in range(len(stops) - 1):
        lo, hi = stops[i], stops[i + 1]
        c0 = np.array(colors[i], dtype=np.float32)
        c1 = np.array(colors[i + 1], dtype=np.float32)
        t = np.clip((dist - lo) / max(1e-6, hi - lo), 0.0, 1.0)[:, :, None]
        seg = c0 + (c1 - c0) * t
        m = (dist >= lo) & (dist <= hi)
        out = np.where(m[:, :, None], seg, out)
    # clamp outside
    out = np.where(dist[:, :, None] < stops[0], np.array(colors[0], dtype=np.float32), out)
    out = np.where(dist[:, :, None] > stops[-1], np.array(colors[-1], dtype=np.float32), out)
    return out


@dataclass
class SceneParams:
    sphere_x: float
    sphere_y: float
    bar_angle: float

    def vector(self) -> np.ndarray:
        """Targets for the reader: 4 numbers that describe the 3 params.

        bar_angle is circular (period 180 deg), so it is encoded as
        (sin 2a, cos 2a); see sweep.py for the R^2 aggregation.
        """
        a = np.radians(self.bar_angle)
        return np.array([self.sphere_x, self.sphere_y, np.sin(2 * a), np.cos(2 * a)],
                        dtype=np.float32)

    def as_dict(self) -> Dict[str, float]:
        return {"sphere_x": self.sphere_x, "sphere_y": self.sphere_y,
                "bar_angle": self.bar_angle}


TARGET_NAMES = ("sphere_x", "sphere_y", "bar_angle_sin2", "bar_angle_cos2")


class MockScene:
    """Fixed-size synthetic stage. frame() is fully vectorized."""

    def __init__(self, width: int = 320, height: int = 240,
                 sphere_amp_x: float = 0.25, sphere_amp_y: float = 0.25):
        self.w, self.h = int(width), int(height)
        self.sphere_amp_x = sphere_amp_x
        self.sphere_amp_y = sphere_amp_y
        self.scale = self.w / 640.0
        Y, X = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.X, self.Y = X, Y

        # static key light at stage center (mock: createRadialGradient '#f5eede'->'#0a0a0a')
        lx, ly = self.w * 0.5, self.h * 0.5
        bg_d = np.sqrt((X - lx) ** 2 + (Y - ly) ** 2)
        self._bg = _radial_ramp(bg_d, (10.0 * self.scale, self.h * 0.85),
                                ((245, 238, 222), (10, 10, 10)))

        # static checkerboard (mock: 8x8 cells of 16px at (470,300), parity frozen)
        cb = np.zeros((self.h, self.w), dtype=bool)
        cx0, cy0 = int(470 * self.scale), int(300 * self.scale)
        cell = max(2, int(16 * self.scale))
        for cy in range(8):
            for cx in range(8):
                if (cx + cy) % 2:
                    y0, x0 = cy0 + cy * cell, cx0 + cx * cell
                    if y0 < self.h and x0 < self.w:
                        cb[y0:y0 + cell, x0:x0 + cell] = True
        self._checker = cb

    # -- geometry -----------------------------------------------------------
    def params_to_px(self, p: SceneParams) -> Tuple[float, float]:
        bx = self.w / 2.0 + (p.sphere_x - 0.5) * 2.0 * self.sphere_amp_x * self.w
        by = self.h / 2.0 + (p.sphere_y - 0.5) * 2.0 * self.sphere_amp_y * self.h
        return bx, by

    def _bar_mask(self, cx: float, cy: float, ang_deg: float,
                  half_len: float, half_thick: float) -> np.ndarray:
        a = np.radians(ang_deg)
        ca, sa = np.cos(a), np.sin(a)
        u = (self.X - cx) * ca + (self.Y - cy) * sa
        v = -(self.X - cx) * sa + (self.Y - cy) * ca
        return (np.abs(u) <= half_len) & (np.abs(v) <= half_thick)

    # -- rendering ----------------------------------------------------------
    def frame(self, sphere_x: float, sphere_y: float, bar_angle: float
              ) -> Tuple[np.ndarray, SceneParams]:
        """Returns (RGB uint8 HxWx3, ground-truth params)."""
        p = SceneParams(float(sphere_x), float(sphere_y), float(bar_angle) % 180.0)
        img = self._bg.copy()

        # shaded sphere (mock: radial highlight gradient at (bx-30,by-30), r=90)
        bx, by = self.params_to_px(p)
        r = 90.0 * self.scale
        off = 30.0 * self.scale
        d_center = np.sqrt((self.X - bx) ** 2 + (self.Y - by) ** 2)
        inside = d_center <= r
        d_hi = np.sqrt((self.X - (bx - off)) ** 2 + (self.Y - (by - off)) ** 2)
        sphere = _radial_ramp(np.clip(d_hi / r, 0.0, 1.4),
                              (0.05, 0.6, 1.0),
                              ((255, 255, 255), (154, 122, 85), (16, 16, 16)))
        img = np.where(inside[:, :, None], sphere, img)

        # bars (mock: rotate(t/90) bar 460x28 '#e8e8e8'; then +45deg bar 540x20 '#989898')
        m1 = self._bar_mask(self.w / 2.0, self.h / 2.0, p.bar_angle,
                            230.0 * self.scale, 14.0 * self.scale)
        img = np.where(m1[:, :, None], np.array([232, 232, 232], dtype=np.float32), img)
        m2 = self._bar_mask(self.w / 2.0, self.h / 2.0, p.bar_angle + 45.0,
                            270.0 * self.scale, 10.0 * self.scale)
        img = np.where(m2[:, :, None], np.array([152, 152, 152], dtype=np.float32), img)

        # checkerboard patch on top
        img = np.where(self._checker[:, :, None], np.array([216, 216, 216], dtype=np.float32), img)

        return np.clip(img, 0, 255).astype(np.uint8), p

    # -- data generation ----------------------------------------------------
    def sample_params(self, n: int, rng: Optional[np.random.Generator] = None) -> List[SceneParams]:
        rng = rng or np.random.default_rng(0)
        sx = rng.uniform(0.0, 1.0, n)
        sy = rng.uniform(0.0, 1.0, n)
        ba = rng.uniform(0.0, 180.0, n)
        return [SceneParams(float(a), float(b), float(c)) for a, b, c in zip(sx, sy, ba)]

    def sequence(self, n: int, seed: int = 0) -> Tuple[List[np.ndarray], np.ndarray, List[SceneParams]]:
        """n frames + (n,4) target matrix + the params themselves."""
        params = self.sample_params(n, np.random.default_rng(seed))
        frames, Y = [], []
        for p in params:
            f, _ = self.frame(p.sphere_x, p.sphere_y, p.bar_angle)
            frames.append(f)
            Y.append(p.vector())
        return frames, np.stack(Y, axis=0), params


if __name__ == "__main__":  # python scene.py -> writes a contact sheet
    import cv2

    sc = MockScene()
    tiles = []
    for i, (sx, sy, ba) in enumerate([(0.2, 0.5, 0), (0.5, 0.5, 30), (0.8, 0.5, 60), (0.5, 0.3, 120)]):
        f, p = sc.frame(sx, sy, ba)
        cv2.putText(f, f"{sx},{sy},{ba}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 80), 1)
        tiles.append(f)
    sheet = np.concatenate([np.concatenate(tiles[:2], axis=1),
                            np.concatenate(tiles[2:], axis=1)], axis=0)
    out = "scene_contact_sheet.png"
    cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"wrote {out} {sheet.shape}")
