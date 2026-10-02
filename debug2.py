"""debug2.py — why is src2 (middle slot) suppressed? Oracle + data-scaling probes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from thermal_glyph_flash import (make_sequence, field_to_rgb, make_codes_fn,
                                 codes_to_channels, W, H)

s48 = dict(density=48, aspect=4/3)

def feat(fields, density=48):
    from renderer import Settings
    s = Settings(density=density, aspect=4/3, ramp="classic", tone_amt=100.0,
                 edge_amt=60.0, edge_thresh=4.0, edge_gain=10.0, relief=70.0)
    cf = make_codes_fn(s, density, s.rows_for())
    return np.stack([codes_to_channels(cf(field_to_rgb(f))) for f in fields])

# data: 6 train seqs + 2 test seqs (same as the run), plus a 3000-frame pool
Ftr = np.concatenate([make_sequence(50, sd)[0] for sd in (1,2,3,4,5,6)])
Ptr = np.concatenate([make_sequence(50, sd)[1].reshape(50,-1) for sd in (1,2,3,4,5,6)])
Fte = np.concatenate([make_sequence(50, sd)[0] for sd in (9,10)])
Pte = np.concatenate([make_sequence(50, sd)[1].reshape(50,-1) for sd in (9,10)])

Yg, Xg_ = np.mgrid[0:H, 0:W].astype(np.float32)

def oracle_src2_x(fields):
    """center-of-mass of field within the middle band [0.38,0.62]"""
    out = []
    mask = (Xg_ >= 0.38*W) & (Xg_ <= 0.62*W)
    for f in fields:
        m = np.where(mask, f, 0.0).astype(np.float32)
        out.append((m.sum(axis=0) * Xg_[0]).sum() / max(m.sum(), 1e-6) / W)
    return np.array(out)

i_x2 = 3 + 1  # params order: src1(x,y,i) src2(x,y,i)... -> src2_x = index 3
print("== oracle: band-COM -> src2_x ==")
for tag, F, P in (("train", Ftr, Ptr), ("test", Fte, Pte)):
    r2 = r2_score(P[:, i_x2], oracle_src2_x(F))
    print(f"  {tag}: R2 = {r2:.4f}")

def ridge_probe(Xtr, ytr, Xte, yte, alphas=(1e-3, 1e-2, 0.1, 1, 10, 100), tag=""):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-8] = 1
    A, B = (Xtr-mu)/sd, (Xte-mu)/sd
    best = (-np.inf, None)
    for a in alphas:
        m = Ridge(alpha=a).fit(A, ytr)
        v = r2_score(yte, m.predict(B), multioutput="uniform_average")
        if v > best[0]: best = (v, a, m)
    v, a, m = best
    per = r2_score(yte, m.predict(B), multioutput="raw_values")
    print(f"  {tag}: alpha={a} test_r2={v:.3f}")
    print(f"    per-param: {np.round(per, 2)}")
    print(f"    src2_x={per[3]:.2f} src2_y={per[4]:.2f} src2_i={per[5]:.2f}")

print("== glyph d48, n_train=300 ==")
Gtr, Gte = feat(Ftr), feat(Fte)
ridge_probe(Gtr, Ptr, Gte, Pte, tag="glyph d48")

print("== more data: 3000 train frames (30 seqs x 100, seeds 100..129) ==")
Fs = [make_sequence(100, sd)[0] for sd in range(100, 122)]
Ps = [make_sequence(100, sd)[1].reshape(100, -1) for sd in range(100, 122)]
Fbig, Pbig = np.concatenate(Fs), np.concatenate(Ps)
Gbig = feat(Fbig)
ridge_probe(Gbig, Pbig, Gte, Pte, tag="glyph d48 n=2200")
