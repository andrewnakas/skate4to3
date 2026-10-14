"""Collision you cannot see: collision surface with no drawn surface within R metres.

The user's first complaint about Summer Camp (2026-09-22) was rideable areas with nothing drawn
and walls you hit but cannot see. `work/scratch/walls.py` measures the opposite (drawn walls with
no collision). This measures collision area whose nearest drawn point is more than R away, sampling
both sides by AREA (render every ~0.35 m, collision every ~0.7 m), because vertex/centroid samples
of a 13,000 m2 lake triangle say nothing about its middle.

    python tools/invisible_collision.py <glb_dir> [--radius 0.5] [--attribute]

`--attribute` names the skate. physics asset nearest to each invisible sample (needs SKATE_WORK
pointing at the level's work dir and the export's recentre shift, read from `recentre.json` beside
the GLBs or parsed from `$SKATE_WORK/export.log`). Samples near no placed physics are collision the
pipeline made itself (reduced, seated, welded).

Baseline, shipped Summer Camp of 2026-09-22 20:30: floor 12.7% (142,649 of 1,122,778 m2),
walls 5.3% (30,231 of 572,190 m2); 98k m2 of the invisible area matched no placed skate. physics.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RENDER_SPACING = 0.12     # m2 per render sample (~0.35 m apart)
COLL_SPACING = 0.5        # m2 per collision sample (~0.7 m apart)


def _sample(a, b, c, spacing, cap, rng):
    """Area-uniform points on triangles (a, b, c): (points, weight m2 each, triangle index)."""
    ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    ns = np.clip(np.ceil(ar / spacing).astype(np.int64), 1, cap)
    rep = np.repeat(np.arange(len(ar)), ns)
    u, v = rng.random(len(rep)), rng.random(len(rep))
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    q = a[rep] + (b[rep] - a[rep]) * u[:, None] + (c[rep] - a[rep]) * v[:, None]
    return q, (ar / ns)[rep], rep


def load(glb_dir):
    """-> (render sample points float32 (n,3), collision samples (q, w, ny))."""
    import verify_build as vb
    vb.GLB = glb_dir
    mats = json.load(open(os.path.join(glb_dir, 'blenrose_materials.json')))
    rng = np.random.default_rng(0)
    vis, cq, cw, cny = [], [], [], []
    for _f, mn, P, _uv, T in vb.prims():
        m = mats.get(mn, {})
        a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        if not m.get('exclude_pres', False):
            q, _, _ = _sample(a, b, c, RENDER_SPACING, 10**7, rng)
            vis += [P.astype(np.float32), q.astype(np.float32)]
        if not m.get('exclude_collision', True):
            keep = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-4
            a, b, c = a[keep], b[keep], c[keep]
            q, w, rep = _sample(a, b, c, COLL_SPACING, 10**6, rng)
            n = np.cross(b - a, c - a)[rep]
            cq.append(q)
            cw.append(w)
            cny.append(n[:, 1] / np.maximum(np.linalg.norm(n, axis=1), 1e-12))
    return (np.concatenate(vis) if vis else np.zeros((0, 3), np.float32),
            (np.concatenate(cq), np.concatenate(cw), np.concatenate(cny)))


def measure(glb_dir, radius=0.5, verbose=True):
    """-> dict with total/invisible m2 per kind and the invisible sample points."""
    from scipy.spatial import cKDTree
    vis, (q, w, ny) = load(glb_dir)
    tree = cKDTree(vis)
    del vis
    d, _ = tree.query(q, distance_upper_bound=radius, workers=-1)
    bad = ~np.isfinite(d)
    kind = np.where(ny > 0.5, 'floor', np.where(ny < -0.5, 'ceiling', 'wall'))
    out = {'radius': radius, 'kinds': {}, 'bad_q': q[bad], 'bad_w': w[bad], 'bad_ny': ny[bad]}
    for k in ('floor', 'wall', 'ceiling'):
        sel = kind == k
        out['kinds'][k] = (float(w[sel].sum()), float(w[sel & bad].sum()))
    out['total'] = float(w.sum())
    out['invisible'] = float(w[bad].sum())
    if verbose:
        print(f"collision with NO drawn surface within {radius} m:")
        for k, (t, b) in out['kinds'].items():
            print(f"  {k:8s} {b:>10,.0f} m2 of {t:>10,.0f} m2 ({100 * b / max(t, 1):.1f}%)")
        qb, wb, nyb = out['bad_q'], out['bad_w'], out['bad_ny']
        tile = collections.Counter()
        tk = zip((np.floor(qb[:, 0] / 100) * 100 + 50).astype(int).tolist(),
                 (np.floor(qb[:, 2] / 100) * 100 + 50).astype(int).tolist())
        for key, ww in zip(tk, wb.tolist()):
            tile[key] += ww
        print('  by tile (m2): ' + ', '.join(f'{a}_{b}:{v:,.0f}' for (a, b), v in tile.most_common(12)))
        cell = collections.Counter()
        ex = {}
        ck = zip((np.floor(qb[:, 0] / 10) * 10).astype(int).tolist(),
                 (np.floor(qb[:, 2] / 10) * 10).astype(int).tolist(),
                 np.where(nyb > 0.5, 'floor', 'wall').tolist())
        for i, (key, ww) in enumerate(zip(ck, wb.tolist())):
            cell[key] += ww
            ex.setdefault(key, i)
        print('  worst 10 m cells:')
        for key, v in cell.most_common(15):
            print(f"    {v:>8,.0f} m2  x={key[0]} z={key[1]} {key[2]:5s} e.g. {qb[ex[key]].round(1).tolist()}")
    return out


def recentre_shift(glb_dir):
    p = os.path.join(glb_dir, 'recentre.json')
    if os.path.exists(p):
        return np.array(json.load(open(p))['shift'], float)
    log = os.path.join(os.environ.get('SKATE_WORK', ''), 'export.log')
    m = re.search(r'recentre shift \[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\]', open(log).read())
    return np.array([float(m[1]), float(m[2]), float(m[3])])


def attribute(res, glb_dir, top=40):
    """Name the skate. physics asset nearest to each invisible sample (within 0.6 m)."""
    from scipy.spatial import cKDTree
    import export_glb as eg
    from cas import decompress_many
    shift = recentre_shift(glb_dir)
    inst = eg.load_instances(eg.CROP)
    G = json.load(open(os.path.join(eg.WORK, 'global_assets.json')))
    RES = G['res']
    del G
    inst = [i for i in inst if not eg.DROP.search(i[0]) and i[0] in RES]
    pn = sorted({eg.physics_name(i[0]) for i in inst} & set(RES))
    cas = eg.Cas()
    phys = {}
    for n, b in zip(pn, decompress_many([cas.read(*RES[n][1:5]) for n in pn])):
        try:
            phys[n] = eg.physics.parse(b)
        except ValueError:
            pass
    rng = np.random.default_rng(1)
    pts, lab = [], []
    for mesh, R, t, _src in inst:
        for part in phys.get(eg.physics_name(mesh), ()):
            if not len(part.triangles):
                continue
            W = part.positions @ R + t + shift
            a, b, c = (W[part.triangles[:, j]] for j in range(3))
            q, _, _ = _sample(a, b, c, 0.25, 400000, rng)
            pts.append(q.astype(np.float32))
            lab += [mesh] * len(q)
    lab = np.array(lab)
    tree = cKDTree(np.concatenate(pts))
    qb, wb, nyb = res['bad_q'], res['bad_w'], res['bad_ny']
    d, i = tree.query(qb, distance_upper_bound=0.6, workers=-1)
    hit = np.isfinite(d)
    by, byf, ex = collections.Counter(), collections.Counter(), {}
    for j in np.nonzero(hit)[0]:
        m = lab[i[j]]
        by[m] += float(wb[j])
        if nyb[j] > 0.5:
            byf[m] += float(wb[j])
        ex.setdefault(m, qb[j].round(1).tolist())
    print(f"attributed to placed skate. physics: {wb[hit].sum():,.0f} m2; "
          f"pipeline-made (no skate. physics within 0.6 m): {wb[~hit].sum():,.0f} m2")
    print('  invisible m2 | floor m2 | example | mesh')
    for m, v in by.most_common(top):
        print(f"  {v:>10,.0f} | {byf[m]:>8,.0f} | {ex[m]} | {m.rsplit('/', 1)[-1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('glb_dir')
    ap.add_argument('--radius', type=float, default=0.5)
    ap.add_argument('--attribute', action='store_true')
    a = ap.parse_args()
    res = measure(a.glb_dir, a.radius)
    if a.attribute:
        attribute(res, a.glb_dir)


if __name__ == '__main__':
    main()
