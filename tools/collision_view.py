#!/usr/bin/env python3
"""Top-down picture of Skate 3 collision: height-tinted, normal-shaded triangles of every 0x80006 in a
cSim stream, grind splines (eb0004) drawn on top, optional KD leaf boxes.

    python tools/collision_view.py <World> <cSim file prefix> out.png [--px 0.05] [--leaves] [--surface]
    e.g. python tools/collision_view.py BlackBoxPark cSim_50_50 work/coll_bbp_50_50.png
"""
import argparse, os, sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sk3 import big, sfil, rw4, collision as co, splines as sp  # noqa: E402

DATA = os.environ.get('SK3_DATA', r'C:\dev\skate3recomp\out\build\windows-release\game\data')


def load(world, prefix):
    a = big.read(os.path.join(DATA, 'content', f'worldDIST_{world}.big'))
    meshes, spl = [], []
    for e in a.entries:
        b = os.path.basename(e.name)
        if b.endswith('.xsf') and b.startswith(prefix):
            for x in sfil.read(a.read(e))[1]:
                ar = rw4.parse(x.arena)
                meshes += [co.parse_mesh(o.data) for o in ar.of_type(co.T_MESH)]
                spl += [sp.parse(o.data) for o in ar.of_type(sp.T_SPLINES)]
    return meshes, spl


def raster(V, T, lo, px, W, H, values):
    """z-buffer (max height) rasteriser over x/z; returns (height, per-pixel value)"""
    zb = np.full((H, W), -np.inf)
    vb = np.zeros((H, W, values.shape[1]))
    P = V[T]                                   # (m, 3, 3)
    X = (P[:, :, 0] - lo[0]) / px
    Z = (P[:, :, 2] - lo[1]) / px
    for t in range(len(T)):
        x, z, y = X[t], Z[t], P[t, :, 1]
        x0, x1 = max(int(np.floor(x.min())), 0), min(int(np.ceil(x.max())), W - 1)
        z0, z1 = max(int(np.floor(z.min())), 0), min(int(np.ceil(z.max())), H - 1)
        if x0 > x1 or z0 > z1:
            continue
        gx, gz = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(z0, z1 + 1) + 0.5)
        d = (x[1] - x[0]) * (z[2] - z[0]) - (x[2] - x[0]) * (z[1] - z[0])
        if abs(d) < 1e-12:
            continue
        l1 = ((gx - x[0]) * (z[2] - z[0]) - (x[2] - x[0]) * (gz - z[0])) / d
        l2 = ((x[1] - x[0]) * (gz - z[0]) - (gx - x[0]) * (z[1] - z[0])) / d
        l0 = 1 - l1 - l2
        m = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        h = l0 * y[0] + l1 * y[1] + l2 * y[2]
        sub = zb[z0:z1 + 1, x0:x1 + 1]
        upd = m & (h > sub)
        sub[upd] = h[upd]
        vb[z0:z1 + 1, x0:x1 + 1][upd] = values[t]
    return zb, vb


def render(meshes, spl, out, px=0.05, leaves=False, surface=False, ymax=np.inf):
    Vs, Ts, Ss, base = [], [], [], 0
    boxes = []
    for m in meshes:
        V, T, S, _E, _U = m.triangles()
        Vs.append(V); Ts.append(T + base); Ss.append(S); base += len(V)
        if leaves:
            boxes += [b for _c, _f, b in m.leaves()]
    V = np.concatenate(Vs).astype(np.float64); T = np.concatenate(Ts); S = np.concatenate(Ss)
    lo = V[:, [0, 2]].min(0) - 1; hi = V[:, [0, 2]].max(0) + 1
    W, H = (np.ceil((hi - lo) / px).astype(int) + 1).tolist()
    N = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N /= np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]
    keep = (N[:, 1] > -0.05) & (V[T][:, :, 1].min(1) < ymax)      # drop ceilings / roofs
    T, S, N = T[keep], S[keep], N[keep]
    if surface:
        ids = {s: k for k, s in enumerate(sorted(set(S.tolist())))}
        rng = np.random.default_rng(3)
        pal = rng.integers(60, 255, (len(ids), 3))
        val = np.concatenate([pal[[ids[s] for s in S.tolist()]], N], 1)
    else:
        val = N
    zb, vb = raster(V, T, lo, px, W, H, val)
    hit = np.isfinite(zb)
    y0, y1 = V[:, 1].min(), V[:, 1].max()
    hn = np.where(hit, (zb - y0) / max(y1 - y0, 1e-6), 0)
    n = vb[..., -3:]
    light = np.array([-0.4, 0.8, -0.45]); light /= np.linalg.norm(light)
    shade = np.clip(np.abs(n @ light), 0.15, 1.0)
    if surface:
        rgb = vb[..., :3] * shade[..., None]
    else:
        tint = np.stack([0.35 + 0.65 * hn, 0.45 + 0.35 * (1 - np.abs(hn - 0.5) * 2), 1.0 - 0.6 * hn], -1)
        rgb = 255 * tint * shade[..., None]
    rgb[~hit] = (20, 20, 24)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)[::-1])     # +z up in the picture
    dr = ImageDraw.Draw(img)
    to_px = lambda p: ((p[0] - lo[0]) / px, H - 1 - (p[2] - lo[1]) / px)
    for b0, b1 in boxes:
        a0, a1 = to_px(b0), to_px(b1)
        dr.rectangle([min(a0[0], a1[0]), min(a0[1], a1[1]), max(a0[0], a1[0]), max(a0[1], a1[1])], outline=(90, 90, 90))
    nseg = 0
    for ss in spl:
        for s in ss.splines:
            pts = [to_px(p) for p in s.points(6)]
            nseg += len(s.segments)
            if len(pts) > 1:
                dr.line(pts, fill=(255, 40, 40), width=3)
            for g in s.segments:
                q = to_px(g.d)
                dr.ellipse([q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2], fill=(255, 230, 0))
    img.save(out)
    return dict(tris=len(T), size=(W, H), splines=sum(len(s.splines) for s in spl), segments=nseg,
                xz=(lo.round(2).tolist(), hi.round(2).tolist()), y=(round(float(y0), 2), round(float(y1), 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world'); ap.add_argument('prefix'); ap.add_argument('out')
    ap.add_argument('--px', type=float, default=0.05)
    ap.add_argument('--ymax', type=float, default=np.inf)
    ap.add_argument('--leaves', action='store_true'); ap.add_argument('--surface', action='store_true')
    a = ap.parse_args()
    meshes, spl = load(a.world, a.prefix)
    print(render(meshes, spl, a.out, a.px, a.leaves, a.surface, a.ymax))


if __name__ == '__main__':
    main()
