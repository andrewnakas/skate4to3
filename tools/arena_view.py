#!/usr/bin/env python3
"""Top-down check of assembled tile arenas: decodes our own cPres (render, light shading) and cSim
(collision, red where there is collision without render) arenas back with sk3's decoders.

    python tools/arena_view.py <arenas dir> out.png [--px 0.5]
"""
import argparse, glob, os, sys
import numpy as np
from PIL import Image, ImageDraw
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sk3 import rw4, render, collision as co

ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('out'); ap.add_argument('--px', type=float, default=0.5)
a = ap.parse_args()
tris_r, tris_c = [], []
for f in sorted(glob.glob(os.path.join(a.dir, 'cPres_*.arena'))):
    for p in render.decode(open(f, 'rb').read()).parts():
        P, I = p.positions, np.asarray(p.indices).reshape(-1, 3)
        tris_r.append(P[I])
for f in sorted(glob.glob(os.path.join(a.dir, 'cSim_*.arena'))):
    ar = rw4.parse(open(f, 'rb').read())
    for o in ar.of_type(0x80006):
        V, T = co.parse_mesh(o.data).triangles()[:2]
        tris_c.append(np.asarray(V)[np.asarray(T)])
R = np.concatenate(tris_r); C = np.concatenate(tris_c)
lo = np.minimum(R.reshape(-1, 3).min(0), C.reshape(-1, 3).min(0)); hi = np.maximum(R.reshape(-1, 3).max(0), C.reshape(-1, 3).max(0))
W, H = int((hi[0] - lo[0]) / a.px) + 1, int((hi[2] - lo[2]) / a.px) + 1
img = Image.new('RGB', (W, H), (20, 20, 30)); d = ImageDraw.Draw(img)
def xy(t): return [((v[0] - lo[0]) / a.px, (v[2] - lo[2]) / a.px) for v in t]
for t in C:
    d.polygon(xy(t), fill=(150, 30, 30))
for t in R:
    n = np.cross(t[1] - t[0], t[2] - t[0]); n = n / (np.linalg.norm(n) + 1e-9)
    s = int(60 + 180 * abs(n[1])); h = int(np.clip((t[:, 1].mean() - lo[1]) / max(hi[1] - lo[1], 1) * 80, 0, 80))
    d.polygon(xy(t), fill=(s - 40 + h // 2, s - 20, s - h))
img.save(a.out)
print(f'render tris {len(R):,}  collision tris {len(C):,}  image {W}x{H}')
