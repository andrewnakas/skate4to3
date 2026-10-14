"""Seams you can catch on: height steps where one rideable collision surface meets another.

The user's standing complaint about Summer Camp (2026-09-23): "i want everything to not have any
seams that could catch you and be smooth". A seam is the open edge of an up-facing collision
region with another up-facing surface just across it at a different height. Every open edge of
the up-facing collision is sampled every STEP m. At each sample the heights of all up-facing
surfaces a few cm across the edge are compared with the edge's own height:

    lip = (surface across) - (edge height), 1.5-10 cm either way, over a surface that also
    reaches this edge's side (so a curb top against the road 15 cm below is not a seam).

    python tools/seam_scan.py <glb_dir> [--min 0.015] [--max 0.10] [--attribute] [--box x0,z0,x1,z1]

Reports metres of seam per step band, and the worst 5 m cells with the nearest placed meshes
(`--attribute`, needs SKATE_WORK and recentre.json beside the GLBs).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STEP = 0.10      # m between samples along an edge
ACROSS = 0.04    # m: probe distance across the edge
CELL = 4.0       # m: triangle hash cell


def load(glb_dir, box=None):
    import verify_build as vb
    vb.GLB = glb_dir
    mats = json.load(open(os.path.join(glb_dir, 'blenrose_materials.json')))
    A, B, C = [], [], []
    for _f, mn, P, _uv, T in vb.prims():
        if mats.get(mn, {}).get('exclude_collision', True) or not len(T):
            continue
        P = P.astype(np.float64)
        a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        n = np.cross(b - a, c - a)
        up = n[:, 1] > 0.7 * np.linalg.norm(n, axis=1)
        if box is not None:
            cx = (a[:, 0] + b[:, 0] + c[:, 0]) / 3
            cz = (a[:, 2] + b[:, 2] + c[:, 2]) / 3
            up &= (cx > box[0] - 20) & (cx < box[2] + 20) & (cz > box[1] - 20) & (cz < box[3] + 20)
        A.append(a[up]), B.append(b[up]), C.append(c[up])
    return np.concatenate(A), np.concatenate(B), np.concatenate(C)


def open_edges(A, B, C, q=1e-3, with_tri=False):
    """Edges used by exactly one up-facing triangle (positions quantised to 1 mm), with the
    triangle's third vertex (tells which side is 'inside')."""
    V = np.stack([A, B, C], 1)                       # (n,3,3)
    _, vid = np.unique(np.round(V.reshape(-1, 3) / q).astype(np.int64), axis=0, return_inverse=True)
    vid = vid.reshape(-1, 3)
    i0 = np.concatenate([vid[:, 0], vid[:, 1], vid[:, 2]])
    i1 = np.concatenate([vid[:, 1], vid[:, 2], vid[:, 0]])
    pair = np.minimum(i0, i1).astype(np.int64) * (int(vid.max()) + 1) + np.maximum(i0, i1)
    _, inv, cnt = np.unique(pair, return_inverse=True, return_counts=True)
    single = cnt[inv] == 1
    P0 = np.concatenate([V[:, 0], V[:, 1], V[:, 2]])
    P1 = np.concatenate([V[:, 1], V[:, 2], V[:, 0]])
    P2 = np.concatenate([V[:, 2], V[:, 0], V[:, 1]])
    if with_tri:
        tri = np.concatenate([np.arange(len(A))] * 3)
        return P0[single], P1[single], P2[single], tri[single]
    return P0[single], P1[single], P2[single]


class Heights:
    """Heights of the up-facing triangles over XZ points."""

    def __init__(self, A, B, C):
        self.A, self.B, self.C = A, B, C
        lo = np.floor(np.minimum(np.minimum(A, B), C)[:, [0, 2]] / CELL).astype(np.int64)
        hi = np.floor(np.maximum(np.maximum(A, B), C)[:, [0, 2]] / CELL).astype(np.int64)
        self.grid = collections.defaultdict(list)
        for i in range(len(A)):
            for x in range(lo[i, 0], hi[i, 0] + 1):
                for z in range(lo[i, 1], hi[i, 1] + 1):
                    self.grid[(x, z)].append(i)
        self.grid = {k: np.array(v, np.int64) for k, v in self.grid.items()}

    def above(self, pts, y, eps=0.01, reach=0.5):
        """True where some up-facing surface lies eps..reach m above y at the XZ point (the
        point is covered: an edge under another surface is not something a wheel meets)."""
        h = self.query(pts, y + (eps + reach) / 2, window=(reach - eps) / 2)
        return np.isfinite(h)

    def query(self, pts, ref_y, window=0.5):
        """For each XZ point: the height of the up-facing surface closest to ref_y within
        `window` m (nan if none)."""
        out = np.full(len(pts), np.nan)
        win = np.broadcast_to(np.asarray(window, np.float64), (len(pts),))
        cells = np.floor(pts[:, [0, 2]] / CELL).astype(np.int64)
        order = np.lexsort((cells[:, 1], cells[:, 0]))
        cs = cells[order]
        brk = np.nonzero((np.diff(cs, axis=0) != 0).any(1))[0] + 1
        for grp in np.split(order, brk):
            tri = self.grid.get(tuple(cells[grp[0]].tolist()))
            if tri is None:
                continue
            A, B, C = self.A[tri], self.B[tri], self.C[tri]
            p = pts[grp][:, [0, 2]]
            v0, v1 = (B - A)[:, [0, 2]], (C - A)[:, [0, 2]]
            den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            den = np.where(np.abs(den) < 1e-12, np.nan, den)
            v2 = p[:, None, :] - A[None, :, [0, 2]]
            w1 = (v2[..., 0] * v1[None, :, 1] - v1[None, :, 0] * v2[..., 1]) / den[None]
            w2 = (v0[None, :, 0] * v2[..., 1] - v2[..., 0] * v0[None, :, 1]) / den[None]
            ins = (w1 >= -1e-6) & (w2 >= -1e-6) & (w1 + w2 <= 1 + 1e-6)
            y = A[None, :, 1] + w1 * (B - A)[None, :, 1] + w2 * (C - A)[None, :, 1]
            d = np.where(ins, np.abs(y - ref_y[grp][:, None]), np.inf)
            j = d.argmin(axis=1)
            best = d[np.arange(len(grp)), j]
            ok = best <= win[grp]
            out[grp[ok]] = y[np.arange(len(grp)), j][ok]
        return out


def scan(glb_dir, lo_step=0.015, hi_step=0.10, box=None):
    A, B, C = load(glb_dir, box)
    P0, P1, P2 = open_edges(A, B, C)
    L = np.linalg.norm((P1 - P0)[:, [0, 2]], axis=1)
    keep = L > 1e-3
    P0, P1, P2, L = P0[keep], P1[keep], P2[keep], L[keep]
    ns = np.maximum(1, np.floor(L / STEP).astype(np.int64))
    rep = np.repeat(np.arange(len(L)), ns)
    k = np.concatenate([(np.arange(n) + 0.5) / n for n in ns.tolist()])
    S = P0[rep] + (P1 - P0)[rep] * k[:, None]
    seg = (P1 - P0)[rep][:, [0, 2]]
    nrm = np.stack([-seg[:, 1], seg[:, 0]], 1) / np.linalg.norm(seg, axis=1)[:, None]
    inside = (P2[rep][:, [0, 2]] - S[:, [0, 2]])
    flip = (nrm * inside).sum(1) > 0
    nrm[flip] = -nrm[flip]                           # nrm points OUT of the edge's own face
    if box is not None:
        m = (S[:, 0] > box[0]) & (S[:, 0] < box[2]) & (S[:, 2] > box[1]) & (S[:, 2] < box[3])
        S, nrm, rep = S[m], nrm[m], rep[m]
    H = Heights(A, B, C)
    inward = S.copy()
    inward[:, 0] -= nrm[:, 0] * 0.02
    inward[:, 2] -= nrm[:, 1] * 0.02
    vis = ~H.above(inward, S[:, 1])
    S, nrm = S[vis], nrm[vis]
    Q = S.copy()
    Q[:, 0] += nrm[:, 0] * ACROSS
    Q[:, 2] += nrm[:, 1] * ACROSS
    across = H.query(Q, S[:, 1], window=hi_step + 0.02)
    step = across - S[:, 1]
    lip = np.isfinite(step) & (np.abs(step) >= lo_step) & (np.abs(step) <= hi_step)
    return S, step, lip, len(ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('glb_dir')
    ap.add_argument('--min', type=float, default=0.015)
    ap.add_argument('--max', type=float, default=0.10)
    ap.add_argument('--box', default=None)
    ap.add_argument('--attribute', action='store_true')
    ap.add_argument('--top', type=int, default=40)
    a = ap.parse_args()
    box = tuple(float(v) for v in a.box.split(',')) if a.box else None
    S, step, lip, nedge = scan(a.glb_dir, a.min, a.max, box)
    print(f"open up-facing edges {nedge:,}, samples {len(S):,} every {STEP} m")
    tot = lip.sum() * STEP
    print(f"seams {a.min * 100:.1f}-{a.max * 100:.0f} cm: {tot:,.0f} m")
    for b0, b1 in ((0.015, 0.03), (0.03, 0.05), (0.05, 0.075), (0.075, 0.10)):
        m = lip & (np.abs(step) >= b0) & (np.abs(step) < b1)
        print(f"  {b0 * 100:4.1f}-{b1 * 100:4.1f} cm: {m.sum() * STEP:8,.0f} m")
    cell = collections.Counter()
    ex = {}
    for p, s in zip(S[lip], step[lip]):
        k = (int(np.floor(p[0] / 5) * 5), int(np.floor(p[2] / 5) * 5))
        cell[k] += STEP
        ex.setdefault(k, []).append((p, s))
    inst = None
    if a.attribute:
        import export_glb as eg
        shift = np.array(json.load(open(os.path.join(a.glb_dir, 'recentre.json')))['shift'])
        inst = [(m.rsplit('/', 1)[-1], t + shift) for m, _R, t, _ in eg.load_instances(eg.CROP)
                if not eg.DROP.search(m)]
        IP = np.array([p for _, p in inst])
    print(f"worst 5 m cells (m of seam, median step cm, height):")
    for k, v in cell.most_common(a.top):
        pts = np.array([p for p, _ in ex[k]])
        med = np.median([s for _, s in ex[k]]) * 100
        line = f"  {v:6.1f} m  x={k[0]} z={k[1]} y={np.median(pts[:, 1]):.1f}  step {med:+.1f} cm"
        if inst is not None:
            c = pts.mean(0)
            d = np.hypot(IP[:, 0] - c[0], IP[:, 2] - c[2]) + 3 * np.abs(IP[:, 1] - c[1])
            near = []
            for i in np.argsort(d)[:12]:
                if inst[i][0] not in near:
                    near.append(inst[i][0])
            line += '  ' + ', '.join(near[:3])
        print(line)


if __name__ == '__main__':
    main()
