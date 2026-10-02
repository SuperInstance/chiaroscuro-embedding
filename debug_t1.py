"""debug_t1.py — throwaway sanity checks for the t1 pipeline. NOT an artifact."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from thermal_glyph_flash import (make_sequence, field_to_rgb, make_codes_fn,
                                 featurize_split, W, H, K_SOURCES)
from renderer import Settings
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# 1) scene<->label alignment: argmax of field should sit at (x_k, y_k)
fields, params = make_sequence(20, 11)
f = 7
for k in range(K_SOURCES):
    x, y = params[f, k, 0], params[f, k, 1]
    px, py = x * W, y * H
    x0, x1 = max(0, int(px) - 40), min(W, int(px) + 40)
    y0, y1 = max(0, int(py) - 40), min(H, int(py) + 40)
    win = fields[f][y0:y1, x0:x1]
    iy, ix = np.unravel_index(win.argmax(), win.shape)
    print(f"src{k+1}: label=({x:.2f},{y:.2f}) local-argmax=({(x0+ix)/W:.2f},{(y0+iy)/H:.2f}) val={win.max():.2f}")

y = params.reshape(20, -1)

def ridge_probe(X, y, ntr, tag, alphas=(1e-2, 1e-1, 1, 10, 100, 1000, 1e4)):
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0); sd[sd < 1e-6] = 1
    Xa = (X - mu) / sd
    best = (None, -np.inf)
    for a in alphas:
        r = Ridge(alpha=a).fit(Xa[:ntr], y[:ntr])
        v = r2_score(y[ntr:], r.predict(Xa[ntr:]), multioutput="uniform_average")
        if v > best[1]: best = (a, v, r)
    a, v, r = best
    per = np.round(r2_score(y[ntr:], r.predict(Xa[ntr:]), multioutput="raw_values"), 2)
    print(f"{tag}: alpha={a} test_r2_uniform={v:.3f} per-param={per}")

# 2) raw field (decimated), WITHIN-seq (seq11 first 14 train / last 6 test)
Xraw = fields.reshape(20, -1)[:, ::16]
ridge_probe(Xraw, y, 14, "raw-field within-seq (20fr)")

# 3) glyph codes (d48), same within-seq
s = Settings(density=48, aspect=4/3, ramp="classic", tone_amt=100.0, edge_amt=60.0,
             edge_thresh=4.0, edge_gain=10.0, relief=70.0)
cf = make_codes_fn(s, 48, 36)
Xg = featurize_split(fields, cf)
ridge_probe(Xg, y, 14, "glyph d48 within-seq (20fr)")

# 4) full-scale: train seq11 (100fr) -> test seq12 (100fr), raw field vs glyphs
f11, p11 = make_sequence(100, 11)
f12, p12 = make_sequence(100, 12)
y12 = p12.reshape(100, -1); y11 = p11.reshape(100, -1)
Xa = f11.reshape(100, -1)[:, ::16]; Xb = f12.reshape(100, -1)[:, ::16]
ridge_probe(np.vstack([Xa, Xb]), np.vstack([y11, y12]), 100, "raw-field CROSS-seq 11->12")
Xga = featurize_split(f11, cf); Xgb = featurize_split(f12, cf)
ridge_probe(np.vstack([Xga, Xgb]), np.vstack([y11, y12]), 100, "glyph d48 CROSS-seq 11->12")

# 5) what does the field actually look like — is signal saturation/clipping hiding position?
print("field stats seq11 frame0: min %.3f max %.3f mean %.3f" % (f11[0].min(), f11[0].max(), f11[0].mean()))
print("cells >0.95 of max:", int((f11[0] > 0.95 * f11[0].max()).sum()), "/", f11[0].size)
